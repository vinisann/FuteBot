import requests

def get_wikipedia_player_image(name):
    # Try English Wikipedia first
    url_en = f"https://en.wikipedia.org/w/api.php?action=query&titles={name}&prop=pageimages&format=json&pithumbsize=150&redirects=1"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url_en, headers=headers, timeout=5).json()
        pages = resp.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if "thumbnail" in page_data:
                return page_data["thumbnail"]["source"]
    except Exception:
        pass
        
    # Try Portuguese Wikipedia if English fails
    url_pt = f"https://pt.wikipedia.org/w/api.php?action=query&titles={name}&prop=pageimages&format=json&pithumbsize=150&redirects=1"
    try:
        resp = requests.get(url_pt, headers=headers, timeout=5).json()
        pages = resp.get("query", {}).get("pages", {})
        for page_id, page_data in pages.items():
            if "thumbnail" in page_data:
                return page_data["thumbnail"]["source"]
    except Exception:
        pass
        
    return None

players = ["Neymar", "Vinícius Júnior", "Lionel Messi", "Kylian Mbappé", "Julián Álvarez", "Alisson Becker", "Casemiro", "Rodri"]
for p in players:
    img = get_wikipedia_player_image(p)
    print(f"{p}: {img}")
