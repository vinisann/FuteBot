"""
Módulo de Web Scraping & Análise Pré-Jogo do FuteBot.
Inclui coleta de notícias (RSS), previsão do tempo (Open-Meteo),
cálculo de odds de mercado e escalações prováveis.
"""

import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from urllib.parse import quote
import urllib.parse
from scipy.stats import poisson
import numpy as np
from googlenewsdecoder import new_decoderv1
import re
import html

# ============================================================
# MAPEAMENTO DE CIDADES-SEDE DA COPA DO MUNDO 2026
# ============================================================

# Cada estádio da Copa 2026 mapeado com cidade, latitude, longitude e timezone
VENUE_COORDS = {
    # EUA
    "MetLife Stadium":        {"cidade": "Nova York/Nova Jersey", "lat": 40.8128, "lon": -74.0742, "tz": "America/New_York"},
    "SoFi Stadium":           {"cidade": "Los Angeles", "lat": 33.9535, "lon": -118.3392, "tz": "America/Los_Angeles"},
    "AT&T Stadium":           {"cidade": "Dallas", "lat": 32.7473, "lon": -97.0945, "tz": "America/Chicago"},
    "Hard Rock Stadium":      {"cidade": "Miami", "lat": 25.958, "lon": -80.2389, "tz": "America/New_York"},
    "Lumen Field":            {"cidade": "Seattle", "lat": 47.5952, "lon": -122.3316, "tz": "America/Los_Angeles"},
    "Lincoln Financial Field":{"cidade": "Filadélfia", "lat": 39.9008, "lon": -75.1675, "tz": "America/New_York"},
    "Arrowhead Stadium":      {"cidade": "Kansas City", "lat": 39.0489, "lon": -94.4839, "tz": "America/Chicago"},
    "NRG Stadium":            {"cidade": "Houston", "lat": 29.6847, "lon": -95.4107, "tz": "America/Chicago"},
    "Mercedes-Benz Stadium":  {"cidade": "Atlanta", "lat": 33.7554, "lon": -84.4010, "tz": "America/New_York"},
    "Levi's Stadium":         {"cidade": "São Francisco/Bay Area", "lat": 37.4033, "lon": -121.9694, "tz": "America/Los_Angeles"},
    "Gillette Stadium":       {"cidade": "Boston", "lat": 42.0909, "lon": -71.2643, "tz": "America/New_York"},
    # México
    "Estadio Azteca":         {"cidade": "Cidade do México", "lat": 19.3029, "lon": -99.1505, "tz": "America/Mexico_City"},
    "Estadio BBVA":           {"cidade": "Monterrey", "lat": 25.6698, "lon": -100.2455, "tz": "America/Monterrey"},
    "Estadio Akron":          {"cidade": "Guadalajara", "lat": 20.6821, "lon": -103.4628, "tz": "America/Monterrey"},
    # Canadá
    "BMO Field":              {"cidade": "Toronto", "lat": 43.6336, "lon": -79.4186, "tz": "America/Toronto"},
    "BC Place":               {"cidade": "Vancouver", "lat": 49.2768, "lon": -123.1117, "tz": "America/Vancouver"},
}

# Mapeamento simplificado: grupo -> cidade principal onde os jogos acontecem
GROUP_VENUES = {
    "A": {"cidade": "Cidade do México", "lat": 19.3029, "lon": -99.1505, "tz": "America/Mexico_City"},
    "B": {"cidade": "Vancouver", "lat": 49.2768, "lon": -123.1117, "tz": "America/Vancouver"},
    "C": {"cidade": "Los Angeles", "lat": 33.9535, "lon": -118.3392, "tz": "America/Los_Angeles"},
    "D": {"cidade": "Houston", "lat": 29.6847, "lon": -95.4107, "tz": "America/Chicago"},
    "E": {"cidade": "Dallas", "lat": 32.7473, "lon": -97.0945, "tz": "America/Chicago"},
    "F": {"cidade": "Seattle", "lat": 47.5952, "lon": -122.3316, "tz": "America/Los_Angeles"},
    "G": {"cidade": "Boston", "lat": 42.0909, "lon": -71.2643, "tz": "America/New_York"},
    "H": {"cidade": "Miami", "lat": 25.958, "lon": -80.2389, "tz": "America/New_York"},
    "I": {"cidade": "Monterrey", "lat": 25.6698, "lon": -100.2455, "tz": "America/Monterrey"},
    "J": {"cidade": "Filadélfia", "lat": 39.9008, "lon": -75.1675, "tz": "America/New_York"},
    "K": {"cidade": "Kansas City", "lat": 39.0489, "lon": -94.4839, "tz": "America/Chicago"},
    "L": {"cidade": "Atlanta", "lat": 33.7554, "lon": -84.4010, "tz": "America/New_York"},
}

# Fallback para mata-mata (MetLife Stadium, NY)
DEFAULT_VENUE = {"cidade": "Nova York/Nova Jersey", "lat": 40.8128, "lon": -74.0742, "tz": "America/New_York"}


# ============================================================
# 1. NOTÍCIAS VIA GOOGLE NEWS RSS
# ============================================================

def fetch_news_rss(team_name, max_results=4):
    """
    Busca notícias recentes relacionadas à seleção da Copa do Mundo 2026
    via Google News RSS.
    Retorna lista de dicts: [{title, link, pub_date}, ...]
    """
    query = quote(f"{team_name} Copa do Mundo 2026")
    url = f"https://news.google.com/rss/search?q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    
    try:
        resp = requests.get(url, timeout=5, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        resp.raise_for_status()
        
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        
        news = []
        for item in items[:max_results]:
            title = item.findtext("title", "")
            link = item.findtext("link", "")
            pub_date_raw = item.findtext("pubDate", "")
            
            # Formatar a data de publicação
            pub_date = ""
            if pub_date_raw:
                try:
                    dt = datetime.strptime(pub_date_raw, "%a, %d %b %Y %H:%M:%S %Z")
                    pub_date = dt.strftime("%d/%m/%Y %H:%M")
                except ValueError:
                    pub_date = pub_date_raw[:16]
            
            # Limpar o título (Google News adiciona " - Fonte" no final)
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title_clean = parts[0]
                source = parts[1] if len(parts) > 1 else ""
            else:
                title_clean = title
                source = ""
            
            news.append({
                "title": title_clean,
                "source": source,
                "link": link,
                "pub_date": pub_date
            })
        
        return news
    except Exception:
        return []


def scrape_globo_lineup(url):
    """
    Scrapes a G1/GE article to extract paragraphs containing/near probable lineups keywords.
    """
    try:
        resp = requests.get(url, timeout=5, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        if resp.status_code != 200:
            return None
        
        content = resp.text
        # Find paragraphs
        paras = re.findall(r'<p[^>]*class="[^"]*content-text__container[^"]*"[^>]*>(.*?)</p>', content, re.DOTALL)
        if not paras:
            paras = re.findall(r'<p[^>]*>(.*?)</p>', content, re.DOTALL)
            
        clean_paras = []
        for p in paras:
            clean = re.sub(r'<[^>]+>', '', p)
            clean = html.unescape(clean).strip()
            if clean:
                clean_paras.append(clean)
                
        lineup_info = []
        found_lineup_intro = False
        
        keywords_intro = ["formação deve ter", "provável escalação", "deve ir a campo", "deve começar", "escalações do jogo", "prováveis times"]
        keywords_contains = ["no gol", "lateral", "zaga", "meio-campo", "ataque", "escalação", "titular"]
        
        for idx, p in enumerate(clean_paras):
            p_lower = p.lower()
            if any(k in p_lower for k in keywords_intro):
                lineup_info.append(p)
                found_lineup_intro = True
                for offset in range(1, 3):
                    if idx + offset < len(clean_paras):
                        lineup_info.append(clean_paras[idx + offset])
                break
            
        if not found_lineup_intro:
            for idx, p in enumerate(clean_paras):
                p_lower = p.lower()
                hits = sum(1 for k in keywords_contains if k in p_lower)
                if hits >= 3:
                    lineup_info.append(p)
                    if idx > 0:
                        lineup_info.insert(0, clean_paras[idx - 1])
                    if idx + 1 < len(clean_paras):
                        lineup_info.append(clean_paras[idx + 1])
                    break
                    
        return "\n\n".join(lineup_info) if lineup_info else None
    except Exception:
        return None


def fetch_match_specific_news(team_a, team_b, max_results=3):
    """
    Searches Google News RSS for matchup news (e.g. "team_a x team_b escalacao"),
    decodes the intermediate links using googlenewsdecoder, and extracts probable lineup details
    if it's a Globo/G1/GE page.
    """
    query = urllib.parse.quote(f"{team_a} x {team_b} escalação")
    url = f"https://news.google.com/rss/search?q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
    
    try:
        resp = requests.get(url, timeout=5, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        resp.raise_for_status()
        
        root = ET.fromstring(resp.content)
        items = root.findall(".//item")
        
        news = []
        for item in items[:max_results]:
            title = item.findtext("title", "")
            gnews_link = item.findtext("link", "")
            pub_date_raw = item.findtext("pubDate", "")
            
            # format pubDate
            pub_date = ""
            if pub_date_raw:
                try:
                    dt = datetime.strptime(pub_date_raw, "%a, %d %b %Y %H:%M:%S %Z")
                    pub_date = dt.strftime("%d/%m/%Y %H:%M")
                except ValueError:
                    pub_date = pub_date_raw[:16]
            
            # Clean title
            if " - " in title:
                parts = title.rsplit(" - ", 1)
                title_clean = parts[0]
                source = parts[1] if len(parts) > 1 else ""
            else:
                title_clean = title
                source = ""
                
            # Decode link
            decoded_link = gnews_link
            try:
                decoded_resp = new_decoderv1(gnews_link, interval=1)
                if decoded_resp.get("status"):
                    decoded_link = decoded_resp["decoded_url"]
            except Exception:
                pass
                
            # Try to scrape probable lineup if it's G1/GE
            parsed_lineup = None
            if "globo.com" in decoded_link or "ge.globo" in decoded_link:
                parsed_lineup = scrape_globo_lineup(decoded_link)
                
            news.append({
                "title": title_clean,
                "source": source,
                "link": decoded_link,
                "pub_date": pub_date,
                "parsed_lineup": parsed_lineup
            })
            
        return news
    except Exception:
        return []



# ============================================================
# 2. PREVISÃO DO TEMPO VIA OPEN-METEO
# ============================================================

def fetch_weather_forecast(lat, lon, date_str, tz="America/New_York"):
    """
    Busca a previsão do tempo para uma localização e data específicas
    usando a API gratuita Open-Meteo.
    
    Parâmetros:
        lat: Latitude da cidade-sede
        lon: Longitude da cidade-sede
        date_str: Data no formato 'YYYY-MM-DD HH:MM' ou 'YYYY-MM-DDTHH:MM:SS'
        tz: Timezone IANA da cidade-sede
    
    Retorna:
        dict com temperatura_c, precipitacao_mm, umidade_pct, vento_kmh, descricao
    """
    try:
        # Extrair a data para a API (formato YYYY-MM-DD)
        if "T" in date_str:
            dt = datetime.strptime(date_str[:19], "%Y-%m-%dT%H:%M:%S")
        else:
            dt = datetime.strptime(date_str[:16], "%Y-%m-%d %H:%M")
        
        date_only = dt.strftime("%Y-%m-%d")
        target_hour = dt.hour
        
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}"
            f"&hourly=temperature_2m,relative_humidity_2m,precipitation_probability,wind_speed_10m,weather_code"
            f"&start_date={date_only}&end_date={date_only}"
            f"&timezone={tz}"
        )
        
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        
        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        precip = hourly.get("precipitation_probability", [])
        wind = hourly.get("wind_speed_10m", [])
        weather_codes = hourly.get("weather_code", [])
        
        # Encontrar o índice da hora mais próxima
        idx = min(target_hour, len(temps) - 1) if temps else 0
        
        # Mapear weather_code para descrição textual em português
        wmo_descriptions = {
            0: "Céu limpo ☀️",
            1: "Predominantemente limpo 🌤️",
            2: "Parcialmente nublado ⛅",
            3: "Nublado ☁️",
            45: "Neblina 🌫️",
            48: "Neblina com geada 🌫️",
            51: "Garoa leve 🌦️",
            53: "Garoa moderada 🌦️",
            55: "Garoa intensa 🌧️",
            61: "Chuva leve 🌧️",
            63: "Chuva moderada 🌧️",
            65: "Chuva forte 🌧️",
            71: "Neve leve ❄️",
            73: "Neve moderada ❄️",
            75: "Neve forte ❄️",
            80: "Pancadas de chuva leves 🌦️",
            81: "Pancadas de chuva moderadas 🌧️",
            82: "Pancadas de chuva fortes ⛈️",
            95: "Tempestade ⛈️",
            96: "Tempestade com granizo ⛈️",
            99: "Tempestade severa com granizo ⛈️"
        }
        
        wc = weather_codes[idx] if idx < len(weather_codes) else 0
        
        return {
            "temperatura_c": round(temps[idx], 1) if idx < len(temps) else None,
            "umidade_pct": humidity[idx] if idx < len(humidity) else None,
            "precipitacao_pct": precip[idx] if idx < len(precip) else None,
            "vento_kmh": round(wind[idx], 1) if idx < len(wind) else None,
            "descricao": wmo_descriptions.get(wc, "Indisponível"),
            "weather_code": wc
        }
    except Exception:
        return None


# ============================================================
# 3. ODDS DE MERCADO (CALCULADAS PELO MODELO POISSON/ELO)
# ============================================================

def calculate_match_odds(prob_m, prob_e, prob_v, margin=0.05):
    """
    Converte probabilidades reais em odds decimais com margem de mercado (vig).
    
    Parâmetros:
        prob_m: Probabilidade de vitória do mandante (0.0 a 1.0)
        prob_e: Probabilidade de empate (0.0 a 1.0)
        prob_v: Probabilidade de vitória do visitante (0.0 a 1.0)
        margin: Margem da casa de apostas (default 5%)
    
    Retorna:
        dict com odds para 3 "casas": betano, bet365, betfair
    """
    total = prob_m + prob_e + prob_v
    if total == 0:
        return None
    
    # Normalizar probabilidades
    prob_m /= total
    prob_e /= total
    prob_v /= total
    
    def to_odds(prob, vig):
        """Converte probabilidade em odds decimais com margem."""
        if prob <= 0:
            return 99.0
        raw = 1.0 / prob
        return round(raw * (1.0 - vig), 2)
    
    # Cada "casa" tem uma margem ligeiramente diferente para criar variação realista
    houses = {
        "Betano": {
            "mandante": to_odds(prob_m, margin),
            "empate": to_odds(prob_e, margin),
            "visitante": to_odds(prob_v, margin),
        },
        "bet365": {
            "mandante": to_odds(prob_m, margin + 0.01),
            "empate": to_odds(prob_e, margin - 0.005),
            "visitante": to_odds(prob_v, margin + 0.005),
        },
        "Betfair": {
            "mandante": to_odds(prob_m, margin - 0.02),
            "empate": to_odds(prob_e, margin + 0.01),
            "visitante": to_odds(prob_v, margin - 0.01),
        }
    }
    
    return {
        "houses": houses,
        "prob_mandante": round(prob_m * 100, 1),
        "prob_empate": round(prob_e * 100, 1),
        "prob_visitante": round(prob_v * 100, 1),
    }


# ============================================================
# 4. ESCALAÇÕES PROVÁVEIS (PLANTEL ESTÁTICO + FORMAÇÃO)
# ============================================================

PROBABLE_LINEUPS = {
    "Brasil": {
        "formacao": "4-3-3",
        "titulares": [
            "Alisson (GK)", "Danilo", "Marquinhos", "Gabriel Magalhães", "Douglas Santos",
            "Casemiro", "Bruno Guimarães", "Lucas Paquetá",
            "Luiz Henrique", "Matheus Cunha", "Vinícius Jr."
        ],
        "tecnico": "Carlo Ancelotti"
    },
    "Argentina": {
        "formacao": "4-3-3",
        "titulares": [
            "E. Martínez (GK)", "Molina", "Romero", "Lisandro Martínez", "Acuña",
            "De Paul", "Enzo Fernández", "Mac Allister",
            "Messi", "Julián Álvarez", "Nico González"
        ],
        "tecnico": "Lionel Scaloni"
    },
    "França": {
        "formacao": "4-3-3",
        "titulares": [
            "Maignan (GK)", "Koundé", "Upamecano", "Saliba", "Theo Hernández",
            "Kanté", "Tchouaméni", "Griezmann",
            "Dembélé", "Mbappé", "Thuram"
        ],
        "tecnico": "Didier Deschamps"
    },
    "Espanha": {
        "formacao": "4-3-3",
        "titulares": [
            "Unai Simón (GK)", "Carvajal", "Le Normand", "Laporte", "Cucurella",
            "Rodri", "Pedri", "Dani Olmo",
            "Lamine Yamal", "Morata", "Nico Williams"
        ],
        "tecnico": "Luis de la Fuente"
    },
    "Inglaterra": {
        "formacao": "4-2-3-1",
        "titulares": [
            "Pickford (GK)", "Alexander-Arnold", "Stones", "Guehi", "Shaw",
            "Rice", "Bellingham",
            "Saka", "Foden", "Palmer",
            "Kane"
        ],
        "tecnico": "Thomas Tuchel"
    },
    "Alemanha": {
        "formacao": "4-2-3-1",
        "titulares": [
            "Ter Stegen (GK)", "Kimmich", "Rüdiger", "Tah", "Raum",
            "Andrich", "Wirtz",
            "Musiala", "Havertz", "Sané",
            "Füllkrug"
        ],
        "tecnico": "Julian Nagelsmann"
    },
    "Portugal": {
        "formacao": "4-3-3",
        "titulares": [
            "Diogo Costa (GK)", "Cancelo", "Rúben Dias", "António Silva", "Nuno Mendes",
            "Vitinha", "Bernardo Silva", "Bruno Fernandes",
            "Rafael Leão", "Cristiano Ronaldo", "Diogo Jota"
        ],
        "tecnico": "Roberto Martínez"
    },
    "Holanda": {
        "formacao": "4-3-3",
        "titulares": [
            "Verbruggen (GK)", "Dumfries", "De Vrij", "Van Dijk", "Aké",
            "Schouten", "F. de Jong", "Reijnders",
            "Xavi Simons", "Depay", "Gakpo"
        ],
        "tecnico": "Ronald Koeman"
    },
    "Bélgica": {
        "formacao": "4-2-3-1",
        "titulares": [
            "Casteels (GK)", "Castagne", "Faes", "Theate", "De Cuyper",
            "Onana", "Tielemans",
            "Doku", "De Bruyne", "Trossard",
            "Openda"
        ],
        "tecnico": "Domenico Tedesco"
    },
    "Croácia": {
        "formacao": "4-3-3",
        "titulares": [
            "Livaković (GK)", "Juranović", "Šutalo", "Gvardiol", "Sosa",
            "Modrić", "Brozović", "Kovačić",
            "Kramarić", "Petković", "Perišić"
        ],
        "tecnico": "Zlatko Dalić"
    },
    "Uruguai": {
        "formacao": "4-3-3",
        "titulares": [
            "Rochet (GK)", "Nández", "Giménez", "Olivera", "Viña",
            "Valverde", "Bentancur", "Ugarte",
            "Pellistri", "Núñez", "Araújo"
        ],
        "tecnico": "Marcelo Bielsa"
    },
    "EUA": {
        "formacao": "4-3-3",
        "titulares": [
            "Turner (GK)", "Dest", "Richards", "Tim Ream", "Antonee Robinson",
            "McKennie", "Musah", "Reyna",
            "Weah", "Pulisic", "Balogun"
        ],
        "tecnico": "Mauricio Pochettino"
    },
    "México": {
        "formacao": "4-3-3",
        "titulares": [
            "Ochoa (GK)", "Jorge Sánchez", "Montes", "Vásquez", "Gallardo",
            "Edson Álvarez", "Romo", "Chávez",
            "Lozano", "Giménez", "Vega"
        ],
        "tecnico": "Javier Aguirre"
    },
    "Colômbia": {
        "formacao": "4-2-3-1",
        "titulares": [
            "Camilo Vargas (GK)", "Muñoz", "Davinson Sánchez", "Lucumí", "Mojica",
            "Richard Ríos", "Lerma",
            "Arias", "James Rodríguez", "Luis Díaz",
            "Córdoba"
        ],
        "tecnico": "Néstor Lorenzo"
    },
    "Japão": {
        "formacao": "4-2-3-1",
        "titulares": [
            "Suzuki (GK)", "Itakura", "Tomiyasu", "Board", "Mitoma",
            "Endo", "Morita",
            "Kubo", "Kamada", "Doan",
            "Ueda"
        ],
        "tecnico": "Hajime Moriyasu"
    },
    "Marrocos": {
        "formacao": "4-3-3",
        "titulares": [
            "Bounou (GK)", "Hakimi", "Aguerd", "Saiss", "Mazraoui",
            "Amrabat", "Ounahi", "Ziyech",
            "Diaz", "En-Nesyri", "Boufal"
        ],
        "tecnico": "Walid Regragui"
    },
    "Senegal": {
        "formacao": "4-3-3",
        "titulares": [
            "E. Mendy (GK)", "Sabaly", "Koulibaly", "Diallo", "Jakobs",
            "N. Mendy", "Gueye", "Pape Sarr",
            "Ismaïla Sarr", "Dia", "Diatta"
        ],
        "tecnico": "Aliou Cissé"
    },
    "Suíça": {
        "formacao": "3-4-3",
        "titulares": [
            "Sommer (GK)", "Schär", "Akanji", "Ricardo Rodríguez",
            "Widmer", "Xhaka", "Freuler", "Aebischer",
            "Shaqiri", "Embolo", "Vargas"
        ],
        "tecnico": "Murat Yakin"
    },
    "Coreia do Sul": {
        "formacao": "4-2-3-1",
        "titulares": [
            "Kim S. (GK)", "Kim M.", "Kim Y.", "Kim J.", "Park J.",
            "Hwang I.", "Jung W.",
            "Lee K.", "Lee J.", "Son H.",
            "Cho G."
        ],
        "tecnico": "Hong Myung-bo"
    },
    "Austrália": {
        "formacao": "4-4-2",
        "titulares": [
            "Ryan (GK)", "Atkinson", "Souttar", "Rowles", "Behich",
            "McGree", "Mooy", "Irvine", "Goodwin",
            "Duke", "Maclaren"
        ],
        "tecnico": "Graham Arnold"
    },
    "Itália": {
        "formacao": "3-5-2",
        "titulares": [
            "Donnarumma (GK)", "Bastoni", "Buongiorno", "Calafiori",
            "Cambiaso", "Barella", "Locatelli", "Tonali", "Dimarco",
            "Retegui", "Raspadori"
        ],
        "tecnico": "Luciano Spalletti"
    },
    "Escócia": {
        "formacao": "5-4-1",
        "titulares": [
            "Angus Gunn (GK)", "Ralston", "Hendry", "Hanley", "McKenna", "Robertson",
            "McTominay", "McGinn", "Christie", "Gilmour",
            "Ché Adams"
        ],
        "tecnico": "Steve Clarke"
    },
    "Haiti": {
        "formacao": "4-4-2",
        "titulares": [
            "Josué Duverger (GK)", "Alex Christian Jr.", "Carlens Arcus", "Ricardo Adé", "Frantzdy Pierrot",
            "Derrick Etienne Jr.", "Bryan Alceus", "Duckens Nazon", "Leverton Pierre",
            "Frantz Pangop", "Mélchie Dumornay"
        ],
        "tecnico": "Marc Collat"
    },
    "Arábia Saudita": {
        "formacao": "4-3-3",
        "titulares": [
            "Al-Owais (GK)", "Al-Ghannam", "Al-Amri", "Al-Bulayhi", "Al-Shahrani",
            "Kanno", "Al-Malki", "Al-Dawsari",
            "Al-Shehri", "Al-Buraikan", "Al-Ghamdi"
        ],
        "tecnico": "Roberto Mancini"
    },
    "Equador": {
        "formacao": "4-3-3",
        "titulares": [
            "Galíndez (GK)", "Preciado", "Torres", "Hincapié", "Estupiñán",
            "Caicedo", "Franco", "Cifuentes",
            "Sarmiento", "Valencia", "Plata"
        ],
        "tecnico": "Sebastián Beccacece"
    },
    "Turquia": {
        "formacao": "4-2-3-1",
        "titulares": [
            "Altay Bayındır (GK)", "Zeki Çelik", "Merih Demiral", "Kaan Ayhan", "Ferdi Kadıoğlu",
            "Hakan Çalhanoğlu", "Orkun Kökçü",
            "Arda Güler", "Kenan Yıldız", "Barış Alper Yılmaz",
            "Cenk Tosun"
        ],
        "tecnico": "Vincenzo Montella"
    },
    "Irã": {
        "formacao": "4-2-3-1",
        "titulares": [
            "Beiranvand (GK)", "Moharrami", "Kanaani", "Hosseini", "Mohammadi",
            "Ezatolahi", "Noorafkan",
            "Jahanbakhsh", "Ghoddos", "Taremi",
            "Azmoun"
        ],
        "tecnico": "Amir Ghalenoei"
    },
}


def get_probable_lineup(team_name):
    """
    Retorna a escalação provável de uma seleção.
    Se não houver dados específicos, retorna um fallback genérico.
    """
    lineup = PROBABLE_LINEUPS.get(team_name)
    if lineup:
        return lineup
    
    # Fallback genérico
    return {
        "formacao": "4-4-2",
        "titulares": [
            "Goleiro", "Lateral Dir.", "Zagueiro 1", "Zagueiro 2", "Lateral Esq.",
            "Volante 1", "Volante 2", "Meia Dir.", "Meia Esq.",
            "Atacante 1", "Atacante 2"
        ],
        "tecnico": "A confirmar"
    }


def get_match_venue(grupo=None, fase=None):
    """
    Retorna as coordenadas e nome da cidade-sede com base no grupo/fase.
    """
    if grupo and grupo in GROUP_VENUES:
        return GROUP_VENUES[grupo]
    return DEFAULT_VENUE


# ============================================================
# 5. FUNÇÃO DE DIAGNÓSTICO / TESTE
# ============================================================

if __name__ == "__main__":
    print("=== Testando Módulo de Scraping ===\n")
    
    # 1. Notícias
    print("📰 Notícias sobre o Brasil:")
    news = fetch_news_rss("Brasil")
    for n in news:
        print(f"  • [{n['pub_date']}] {n['title']} ({n['source']})")
    print()
    
    # 1.1 Notícias do Confronto (Brasil x Escócia)
    print("📰 Notícias do Confronto Brasil x Escócia:")
    m_news = fetch_match_specific_news("Brasil", "Escócia")
    for mn in m_news:
        print(f"  • [{mn['pub_date']}] {mn['title']} ({mn['source']})")
        print(f"    Link: {mn['link']}")
        if mn['parsed_lineup']:
            print("    [Provável Escalação Extraída!]")
            print("    " + mn['parsed_lineup'].replace('\n', '\n    '))
    print()
    
    # 2. Previsão do tempo (Cidade do México, próxima data)
    print("🌤️ Previsão do tempo (Cidade do México, amanhã):")
    weather = fetch_weather_forecast(19.3029, -99.1505, "2026-06-25 14:00")
    if weather:
        print(f"  Temperatura: {weather['temperatura_c']}°C")
        print(f"  Umidade: {weather['umidade_pct']}%")
        print(f"  Chance de chuva: {weather['precipitacao_pct']}%")
        print(f"  Vento: {weather['vento_kmh']} km/h")
        print(f"  Condição: {weather['descricao']}")
    else:
        print("  Dados não disponíveis (offline ou fora do range de previsão)")
    print()
    
    # 3. Odds
    print("📊 Odds calculadas (Brasil 55% vs Empate 22% vs Adversário 23%):")
    odds = calculate_match_odds(0.55, 0.22, 0.23)
    if odds:
        for house, values in odds["houses"].items():
            print(f"  {house}: Mandante {values['mandante']} | Empate {values['empate']} | Visitante {values['visitante']}")
    print()
    
    # 4. Escalação
    print("⚽ Escalação provável do Brasil:")
    lineup = get_probable_lineup("Brasil")
    print(f"  Formação: {lineup['formacao']}")
    print(f"  Técnico: {lineup['tecnico']}")
    for i, p in enumerate(lineup['titulares'], 1):
        print(f"  {i}. {p}")
    print()
    
    print("✅ Todos os testes concluídos!")
