import xml.etree.ElementTree as ET
import urllib.parse
import requests

query = urllib.parse.quote("Brasil x Escócia escalação")
url = f"https://news.google.com/rss/search?q={query}&hl=pt-BR&gl=BR&ceid=BR:pt-419"

resp = requests.get(url, headers={
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
})
print("Status:", resp.status_code)
root = ET.fromstring(resp.content)
items = root.findall(".//item")

for i, item in enumerate(items[:5]):
    title = item.findtext("title", "")
    link = item.findtext("link", "")
    pub_date = item.findtext("pubDate", "")
    print(f"{i+1}: {title}\n   Link: {link}\n   Date: {pub_date}\n")
