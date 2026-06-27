import os
import sys

# force utf-8 output
sys.stdout.reconfigure(encoding='utf-8')

for root, dirs, files in os.walk("."):
    if ".git" in root or "__pycache__" in root or ".agents" in root:
        continue
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(root, file)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()
                if "fetch_news" in content:
                    print(f"fetch_news found in: {path}")
                if "get_probable_lineup" in content:
                    print(f"get_probable_lineup found in: {path}")
                if "noticia" in content.lower():
                    print(f"'noticia' found in: {path}")
            except Exception as e:
                print(f"Error reading {path}: {e}")
