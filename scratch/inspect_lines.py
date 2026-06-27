with open("pages/2_🔮_Previsoes.py", "r", encoding="utf-8") as f:
    lines = f.readlines()
for i in range(139, 150):
    if i < len(lines):
        print(f"Line {i+1}: {repr(lines[i])}")
