"""
Utilitários compartilhados para o FuteBot.
"""
import requests
import re
import urllib.parse

TEAM_FLAGS = {
    "Brasil": "🇧🇷",
    "Argentina": "🇦🇷",
    "França": "🇫🇷",
    "Bélgica": "🇧🇪",
    "Inglaterra": "🏴\U0001f3f4\U000e0067\U000e0062\U000e0065\U000e006e\U000e0067\U000e007f",
    "Holanda": "🇳🇱",
    "Croácia": "🇭🇷",
    "Itália": "🇮🇹",
    "Portugal": "🇵🇹",
    "Espanha": "🇪🇸",
    "Marrocos": "🇲🇦",
    "Suíça": "🇨🇭",
    "EUA": "🇺🇸",
    "Alemanha": "🇩🇪",
    "Uruguai": "🇺🇾",
    "Senegal": "🇸🇳",
    "Japão": "🇯🇵",
    "Polônia": "🇵🇱",
    "Coreia do Sul": "🇰🇷",
    "Austrália": "🇦🇺",
    "Equador": "🇪🇨",
    "Camarões": "🇨🇲",
    "Arábia Saudita": "🇸🇦",
    "Catar": "🇶🇦",
    "Gana": "🇬🇭",
    "Canadá": "🇨🇦",
    "Costa Rica": "🇨🇷",
    "Sérvia": "🇷🇸",
    "Dinamarca": "🇩🇰",
    "Tunísia": "🇹🇳",
    "México": "🇲🇽",
    "Irã": "🇮🇷",
    "País de Gales": "🏴\U0001f3f4\U000e0067\U000e0062\U000e0077\U000e006c\U000e0073\U000e007f",
    "Áustria": "🇦🇹",
    "Iraque": "🇮🇶",
    "Noruega": "🇳🇴",
    "Jordânia": "🇯🇴",
    "Argélia": "🇩🇿",
    "Haiti": "🇭🇹",
    "Escócia": "🏴\U0001f3f4\U000e0067\U000e0062\U000e0073\U000e0063\U000e0074\U000e007f",
    "África do Sul": "🇿🇦",
    "Chéquia": "🇨🇿",
    "Bósnia e Herzegovina": "🇧🇦",
    "Paraguai": "🇵🇾",
    "Turquia": "🇹🇷",
    "Curaçao": "🇨🇼",
    "Costa do Marfim": "🇨🇮",
    "Suécia": "🇸🇪",
    "Egito": "🇪🇬",
    "Nova Zelândia": "🇳🇿",
    "Cabo Verde": "🇨🇻",
    "Colômbia": "🇨🇴",
    "RD Congo": "🇨🇩",
    "Uzbequistão": "🇺🇿",
    "Panamá": "🇵🇦"
}

def get_flag(team_name):
    """Retorna o emoji da bandeira da seleção informada."""
    return TEAM_FLAGS.get(team_name, "⚽")

def format_fase(fase, grupo=None):
    """
    Formata a fase para exibição no frontend.
    Se for rodada de grupos (1, 2, 3), exibe como 'Xª Rodada' e opcionalmente adiciona o grupo.
    Se for mata-mata, exibe o nome amigável da fase.
    """
    if fase in ["1", "2", "3"]:
        suffix = "ª Rodada"
        if grupo:
            return f"{fase}{suffix} • Grupo {grupo}"
        return f"{fase}{suffix}"
    elif grupo:
        return f"{fase} • Grupo {grupo}"
    return fase

def format_fase_option(f):
    """Mapeia os valores de fase no banco para opções legíveis no selectbox."""
    if f == "1": return "1ª Rodada"
    if f == "2": return "2ª Rodada"
    if f == "3": return "3ª Rodada"
    return f


TEAM_CODES = {
    "Brasil": "br",
    "Argentina": "ar",
    "França": "fr",
    "Bélgica": "be",
    "Inglaterra": "gb-eng",
    "Holanda": "nl",
    "Croácia": "hr",
    "Itália": "it",
    "Portugal": "pt",
    "Espanha": "es",
    "Marrocos": "ma",
    "Suíça": "ch",
    "EUA": "us",
    "Alemanha": "de",
    "Uruguai": "uy",
    "Senegal": "sn",
    "Japão": "jp",
    "Polônia": "pl",
    "Coreia do Sul": "kr",
    "Austrália": "au",
    "Equador": "ec",
    "Camarões": "cm",
    "Arábia Saudita": "sa",
    "Catar": "qa",
    "Gana": "gh",
    "Canadá": "ca",
    "Costa Rica": "cr",
    "Sérvia": "rs",
    "Dinamarca": "dk",
    "Tunísia": "tn",
    "México": "mx",
    "Irã": "ir",
    "País de Gales": "gb-wls",
    "Áustria": "at",
    "Iraque": "iq",
    "Noruega": "no",
    "Jordânia": "jo",
    "Argélia": "dz",
    "Haiti": "ht",
    "Escócia": "gb-sct",
    "África do Sul": "za",
    "Chéquia": "cz",
    "Bósnia e Herzegovina": "ba",
    "Paraguai": "py",
    "Turquia": "tr",
    "Curaçao": "cw",
    "Costa do Marfim": "ci",
    "Suécia": "se",
    "Egito": "eg",
    "Nova Zelândia": "nz",
    "Cabo Verde": "cv",
    "Colômbia": "co",
    "RD Congo": "cd",
    "Uzbequistão": "uz",
    "Panamá": "pa"
}


def get_flag_html(team_name, width=28):
    """
    Retorna a tag HTML <img> com a bandeira oficial da seleção via FlagCDN.
    Se não mapeado, cai de volta no emoji de get_flag.
    """
    code = TEAM_CODES.get(team_name)
    if not code:
        return f'<span style="font-size: {width}px; margin-right: 6px; vertical-align: middle;">{get_flag(team_name)}</span>'
    return f'<img src="https://flagcdn.com/w80/{code}.png" width="{width}" style="vertical-align: middle; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.15); margin-right: 8px;">'


def clean_player_name(name):
    """Remove sufixos em parênteses ou números da string do nome do jogador."""
    name = re.sub(r'\s*\([^)]*\)', '', name)
    return name.strip()


_player_photo_cache = {}

def get_player_photo_url(player_name):
    """
    Tenta obter dinamicamente a foto do jogador via API da Wikipedia.
    Se não encontrar, retorna um avatar estilizado com as iniciais do jogador (DiceBear).
    Utiliza um cache local em memória para evitar chamadas duplicadas repetitivas.
    """
    if player_name in _player_photo_cache:
        return _player_photo_cache[player_name]
        
    clean_name = clean_player_name(player_name)
    
    # User-Agent necessário para evitar bloqueio da Wikipedia (HTTP 403)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    
    # 1. Tentar Wikipedia em Inglês
    url_en = f"https://en.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(clean_name)}&prop=pageimages&format=json&pithumbsize=150&redirects=1"
    try:
        resp = requests.get(url_en, headers=headers, timeout=2).json()
        pages = resp.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if "thumbnail" in page_data:
                url = page_data["thumbnail"]["source"]
                _player_photo_cache[player_name] = url
                return url
    except Exception:
        pass
        
    # 2. Tentar Wikipedia em Português
    url_pt = f"https://pt.wikipedia.org/w/api.php?action=query&titles={urllib.parse.quote(clean_name)}&prop=pageimages&format=json&pithumbsize=150&redirects=1"
    try:
        resp = requests.get(url_pt, headers=headers, timeout=2).json()
        pages = resp.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if "thumbnail" in page_data:
                url = page_data["thumbnail"]["source"]
                _player_photo_cache[player_name] = url
                return url
    except Exception:
        pass
        
    # 3. Fallback: DiceBear initials avatar
    encoded_name = urllib.parse.quote(clean_name)
    url = f"https://api.dicebear.com/9.x/initials/svg?seed={encoded_name}&radius=50&backgroundColor=039be5,00acc1,3949ab,5e35b1,d81b60,e53935,43a047,f4511e,ffb300"
    _player_photo_cache[player_name] = url
    return url



