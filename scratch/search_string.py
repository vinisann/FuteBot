import os

def search_text(path, text):
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith('.py') or file.endswith('.md'):
                fpath = os.path.join(root, file)
                try:
                    with open(fpath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if text in content:
                            print(f"Found in: {fpath}")
                except Exception as e:
                    pass

print("Searching in workspace...")
search_text("C:/Users/vinicius.nunes/FuteBot", "calculate_real_group_standings")
print("Searching in appDataDir...")
search_text("C:/Users/vinicius.nunes/.gemini/antigravity/brain/20c99236-d7a0-4eee-8fdf-e099510ec44e", "calculate_real_group_standings")
