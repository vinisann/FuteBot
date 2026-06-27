try:
    code = 'f"""<p>Gols Esperados do <strong>{mandante_nome}</strong>: <span style="color:#2563eb; font-size:22px; font-weight:600;">{xg_m:.2f}</span></p>"""'
    compile(code, "<string>", "eval")
    print("Compiled successfully!")
except Exception as e:
    import traceback
    traceback.print_exc()
