with open("pages/2_🔮_Previsoes.py", "r", encoding="utf-8") as f:
    for idx, line in enumerate(f):
        if '"""' in line:
            print(f"Line {idx+1}: {repr(line)}")
