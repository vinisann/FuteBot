path = "./pages/1_📊_Estatisticas.py"
with open(path, "r", encoding="utf-8") as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if "Classificação" in line or "classificacao" in line or "Grupo" in line or "standings" in line:
        if "get_flag" in line or "st.markdown" in line or "render" in line or "standings" in line.lower() or "tab" in line.lower():
            print(f"{i+1}: {line.strip()}")
