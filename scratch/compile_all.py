import os
import sys

def check_all_files():
    has_error = False
    for root, dirs, files in os.walk("."):
        # Ignore virtual envs or git
        if "venv" in root or ".git" in root or ".gemini" in root:
            continue
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    compile(content, path, "exec")
                except SyntaxError as e:
                    sys.stdout.buffer.write(f"SyntaxError in {path} at line {e.lineno}:\n".encode("utf-8"))
                    sys.stdout.buffer.write(f"  {e.text}".encode("utf-8"))
                    has_error = True
                except Exception as e:
                    sys.stdout.buffer.write(f"Error reading {path}: {e}\n".encode("utf-8"))
                    has_error = True
    if not has_error:
        print("All Python files compiled successfully!")
    else:
        sys.exit(1)

if __name__ == "__main__":
    check_all_files()
