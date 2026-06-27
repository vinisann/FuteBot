import requests
import re
import html

def scrape_globo_lineup(url):
    try:
        resp = requests.get(url, timeout=10, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        if resp.status_code != 200:
            return None
        
        content = resp.text
        # Find paragraphs
        # G1/GE uses <p class="content-text__container ...">...</p>
        # Let's get both class containers and standard paragraphs
        paras = re.findall(r'<p[^>]*class="[^"]*content-text__container[^"]*"[^>]*>(.*?)</p>', content, re.DOTALL)
        if not paras:
            paras = re.findall(r'<p[^>]*>(.*?)</p>', content, re.DOTALL)
            
        clean_paras = []
        for p in paras:
            clean = re.sub(r'<[^>]+>', '', p)
            clean = html.unescape(clean).strip()
            if clean:
                clean_paras.append(clean)
                
        # Find paragraphs with lineup info
        lineup_info = []
        found_lineup_intro = False
        
        keywords_intro = ["formação deve ter", "provável escalação", "deve ir a campo", "deve começar", "escalações do jogo", "prováveis times"]
        keywords_contains = ["no gol", "lateral", "zaga", "meio-campo", "ataque", "escalação", "titular"]
        
        for idx, p in enumerate(clean_paras):
            p_lower = p.lower()
            # If paragraph contains intro keywords
            if any(k in p_lower for k in keywords_intro):
                lineup_info.append(p)
                found_lineup_intro = True
                # Grab the next 2 paragraphs too, as they usually contain the actual players
                for offset in range(1, 3):
                    if idx + offset < len(clean_paras):
                        lineup_info.append(clean_paras[idx + offset])
                break
            
        # If we didn't find a direct intro, search for paragraphs containing multiple position terms
        if not found_lineup_intro:
            for idx, p in enumerate(clean_paras):
                p_lower = p.lower()
                hits = sum(1 for k in keywords_contains if k in p_lower)
                if hits >= 3: # matches at least 3 position keywords
                    lineup_info.append(p)
                    # Grab surrounding paragraphs
                    if idx > 0:
                        lineup_info.insert(0, clean_paras[idx - 1])
                    if idx + 1 < len(clean_paras):
                        lineup_info.append(clean_paras[idx + 1])
                    break
                    
        return "\n\n".join(lineup_info) if lineup_info else None
        
    except Exception as e:
        return f"Error: {e}"

url = "https://g1.globo.com/hora1/noticia/2026/06/24/brasil-x-escocia-veja-a-provavel-escalacao-para-o-jogo-desta-quarta-feira-24.ghtml"
res = scrape_globo_lineup(url)
print("Parsed Lineup Section:\n", res)
