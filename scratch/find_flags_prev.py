path = "./pages/2_🔮_Previsoes.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "get_flag" in line:
        print(f"{i+1}: {line.strip()}")
