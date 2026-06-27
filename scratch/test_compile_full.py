import traceback
import sys

try:
    with open("pages/2_🔮_Previsoes.py", "r", encoding="utf-8") as f:
        content = f.read()
    compile(content, "pages/2_Previsoes.py", "exec")
    print("No syntax error found in compile()!")
except SyntaxError as e:
    # Safe printing to avoid unicode errors in Windows console
    sys.stdout.buffer.write(f"SyntaxError in file: {e.filename}\n".encode('utf-8'))
    sys.stdout.buffer.write(f"Line number: {e.lineno}\n".encode('utf-8'))
    sys.stdout.buffer.write(f"Offset: {e.offset}\n".encode('utf-8'))
    if e.text:
        sys.stdout.buffer.write(f"Text: {e.text}".encode('utf-8'))
        # Let's show where the offset is pointing in the text
        pointer = " " * (e.offset - 1) + "^\n"
        sys.stdout.buffer.write(pointer.encode('utf-8'))
        # Let's print the character at the offset
        char_idx = e.offset - 1
        if 0 <= char_idx < len(e.text):
            sys.stdout.buffer.write(f"Char at offset: {repr(e.text[char_idx])}\n".encode('utf-8'))
