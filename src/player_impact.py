import re


DEFAULT_RATING = 78.0
BASELINE_STARTER_RATING = 80.0

TEAM_BASE_PLAYER_RATING = {
    "Brasil": 83.5,
    "Argentina": 83.5,
    "Franca": 84.0,
    "Espanha": 83.5,
    "Inglaterra": 83.5,
    "Alemanha": 82.5,
    "Portugal": 83.0,
    "Holanda": 81.5,
    "Belgica": 81.5,
    "Croacia": 80.5,
    "Uruguai": 80.5,
    "Colombia": 80.0,
    "Italia": 81.0,
    "EUA": 78.5,
    "Mexico": 78.0,
    "Japao": 78.5,
    "Marrocos": 78.5,
    "Senegal": 78.0,
    "Suica": 78.0,
}


def clean_player_name(name):
    value = re.sub(r"\s*\([^)]*\)", "", str(name or ""))
    return value.strip()


def _clip(value, low=0.94, high=1.06):
    return float(max(low, min(high, float(value))))


def _players_from_lineup(lineup, key):
    if not lineup or not isinstance(lineup, dict):
        return []
    players = lineup.get(key, [])
    if not players:
        return []
    return [clean_player_name(player) for player in players if clean_player_name(player)]


def _rating_for(player, player_ratings):
    if not player_ratings:
        return DEFAULT_RATING
    clean = clean_player_name(player)
    return float(player_ratings.get(clean, player_ratings.get(player, DEFAULT_RATING)))


def _sector_for_index(index, player_name):
    lowered = str(player_name).lower()
    if "gk" in lowered or "goleiro" in lowered:
        return "goleiro"
    if index <= 4:
        return "defesa"
    if index <= 7:
        return "meio"
    return "ataque"


def _ascii_key(value):
    replacements = {
        "Ã¡": "a",
        "Ã£": "a",
        "Ã¢": "a",
        "Ã©": "e",
        "Ãª": "e",
        "Ã­": "i",
        "Ã³": "o",
        "Ã´": "o",
        "Ãº": "u",
        "Ã§": "c",
        "Ã¼": "u",
    }
    text = str(value or "")
    for old, new in replacements.items():
        text = text.replace(old, new)
    return re.sub(r"[^A-Za-z0-9 ]+", "", text).strip()


def estimate_player_ratings(team_name, lineup=None):
    """Builds a conservative offline rating proxy from the probable lineup."""
    starters = _players_from_lineup(lineup, "titulares")
    bench = _players_from_lineup(lineup, "banco")
    base = TEAM_BASE_PLAYER_RATING.get(_ascii_key(team_name), DEFAULT_RATING)
    ratings = {}

    for idx, player in enumerate(starters + bench, start=1):
        sector = _sector_for_index(idx if idx <= 11 else min(idx - 11, 11), player)
        sector_bonus = {
            "goleiro": 0.2,
            "defesa": 0.0,
            "meio": 0.3,
            "ataque": 0.5,
        }.get(sector, 0.0)
        reserve_penalty = -1.5 if idx > 11 else 0.0
        ratings[player] = float(base + sector_bonus + reserve_penalty)

    return ratings


def calculate_team_player_impact(team_name, lineup=None, player_ratings=None, unavailable_players=None):
    starters = _players_from_lineup(lineup, "titulares")
    bench = _players_from_lineup(lineup, "banco")
    unavailable = {clean_player_name(player) for player in (unavailable_players or [])}

    if not starters:
        return {
            "team": team_name,
            "has_player_data": False,
            "average_starter_rating": DEFAULT_RATING,
            "bench_depth": 0.0,
            "absence_penalty": 0.0,
            "player_adjustment": 1.0,
            "sector_strengths": {},
            "reasons": [f"{team_name}: sem dados de jogadores suficientes; ajuste neutro."],
        }

    available_starters = [player for player in starters if player not in unavailable]
    starter_ratings = [_rating_for(player, player_ratings) for player in available_starters]
    if not starter_ratings:
        starter_ratings = [DEFAULT_RATING]
    average_starter_rating = sum(starter_ratings) / len(starter_ratings)

    bench_ratings = [_rating_for(player, player_ratings) for player in bench]
    bench_depth = 0.0
    if bench_ratings:
        bench_depth = max(0.0, min(1.0, (sum(bench_ratings) / len(bench_ratings) - 76.0) / 10.0))

    absence_penalty = 0.0
    unavailable_relevant = []
    for player in unavailable:
        if player in starters:
            rating = _rating_for(player, player_ratings)
            unavailable_relevant.append(player)
            absence_penalty += max(0.0, rating - 80.0) / 220.0
    absence_penalty = min(0.08, absence_penalty)

    quality_bonus = (average_starter_rating - BASELINE_STARTER_RATING) / 140.0
    depth_bonus = bench_depth * 0.012
    adjustment = _clip(1.0 + quality_bonus + depth_bonus - absence_penalty)

    sectors = {}
    for idx, player in enumerate(available_starters, start=1):
        sector = _sector_for_index(idx, player)
        sectors.setdefault(sector, []).append(_rating_for(player, player_ratings))
    sector_strengths = {
        sector: float(sum(values) / len(values))
        for sector, values in sectors.items()
        if values
    }

    reasons = []
    if average_starter_rating > BASELINE_STARTER_RATING + 2:
        reasons.append(f"{team_name}: titulares acima da media elevam levemente o potencial.")
    if bench_depth > 0:
        reasons.append(f"{team_name}: banco oferece profundidade para sustentar o nivel.")
    if unavailable_relevant:
        reasons.append(f"{team_name}: desfalque de titular relevante reduz o ajuste.")
    if not reasons:
        reasons.append(f"{team_name}: impacto de jogadores neutro.")

    return {
        "team": team_name,
        "has_player_data": True,
        "average_starter_rating": float(average_starter_rating),
        "bench_depth": float(bench_depth),
        "absence_penalty": float(absence_penalty),
        "player_adjustment": float(adjustment),
        "sector_strengths": sector_strengths,
        "reasons": reasons,
    }


def _for_team(mapping, team_name, default=None):
    if isinstance(mapping, dict):
        return mapping.get(team_name, default)
    return default


def build_match_player_impact(
    mandante,
    visitante,
    lineups=None,
    player_ratings=None,
    unavailable_players=None,
):
    mandante_impact = calculate_team_player_impact(
        mandante,
        lineup=_for_team(lineups, mandante),
        player_ratings=_for_team(player_ratings, mandante, {}),
        unavailable_players=_for_team(unavailable_players, mandante, []),
    )
    visitante_impact = calculate_team_player_impact(
        visitante,
        lineup=_for_team(lineups, visitante),
        player_ratings=_for_team(player_ratings, visitante, {}),
        unavailable_players=_for_team(unavailable_players, visitante, []),
    )
    used = mandante_impact["player_adjustment"] != 1.0 or visitante_impact["player_adjustment"] != 1.0
    return {
        "mandante": mandante_impact,
        "visitante": visitante_impact,
        "modelo_com_jogadores": bool(used),
        "fator_jogadores_mandante": float(mandante_impact["player_adjustment"]),
        "fator_jogadores_visitante": float(visitante_impact["player_adjustment"]),
        "motivos_jogadores": mandante_impact["reasons"] + visitante_impact["reasons"],
    }


def build_player_impact_from_lineups(mandante, visitante, lineups=None, unavailable_players=None):
    lineups = lineups or {}
    player_ratings = {
        mandante: estimate_player_ratings(mandante, _for_team(lineups, mandante, {})),
        visitante: estimate_player_ratings(visitante, _for_team(lineups, visitante, {})),
    }
    return build_match_player_impact(
        mandante,
        visitante,
        lineups=lineups,
        player_ratings=player_ratings,
        unavailable_players=unavailable_players,
    )
