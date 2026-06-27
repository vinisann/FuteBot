import re
import html

path = r"C:\Users\vinicius.nunes\.gemini\antigravity\brain\20c99236-d7a0-4eee-8fdf-e099510ec44e\.system_generated\steps\1506\content.md"

with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Let's find all text inside paragraphs or headers
# Or specifically inside class="content-text__container" or similar classes
# G1 typically uses <p class="content-text__container ...">...</p>
matches = re.findall(r'<p[^>]*class="[^"]*content-text__container[^"]*"[^>]*>(.*?)</p>', content, re.DOTALL)
if not matches:
    # try generic paragraphs
    matches = re.findall(r'<p[^>]*>(.*?)</p>', content, re.DOTALL)

for i, match in enumerate(matches):
    # remove html tags
    clean = re.sub(r'<[^>]+>', '', match)
    clean = html.unescape(clean).strip()
    if clean:
        print(f"{i}: {clean}")
