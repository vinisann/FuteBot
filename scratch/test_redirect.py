import requests
import re

url = "https://news.google.com/rss/articles/CBMixwFBVV95cUxQcWs4NUdHMHowc3FZeUJhQUVSWU05QnF1aWtGZUo0UWVNZWwyWlRHV2ZhMndmbUw1NjFXdl9VN2paQmJ3OVRpa1dYeTd3dWpPeWttVjhyZjZOQ3JlbjlBN3hMdWo4dU90OTBzekotRm5HS2NJd1E3OFI0XzkzV2JLQjJfRXFnY3FUNDFvd3FrT2RVbVFwRkkzZG5zbmhSemthRHVQRnRlbnJfbFdFbVFqTl9WUHJ3a2RFZFB5ODBFRFFxSC1qVV9v0gHWAUFVX3lxTE1vVEFydzk1T0RKVmRad3Z3SmVpaWtGcFRzN01Md1Fsd3gwV3FTeHg5djFuTnBLQzFodkxMU2VmOUpqVzRKZHlHcjc2UUVDWko2MGtFYXNLQW5MTEpqSjJuVXVtd2FEUkxka3AtLW1zMXh4MkF5S1Uxak1Ia3pob2lCMGNzcGx3aUpCUnFIb2NscHdPSXdJQXUtTjVCOGJHUGI5bGlCelRTSFlRQk9HSExRWFRuVVFzNjU3NmpZMjNvdHI0SGlGVlZoUVJjR0ZfVVYtNVhWZGc?oc=5"

resp = requests.get(url)

# Print all substrings that look like urls
urls = re.findall(r'https?://[^\s"\'<>]+', resp.text)
globo_urls = [u for u in urls if "globo" in u]
print("Globo URLs:", globo_urls)

# Search for the string "g1.globo.com" anywhere
g1_matches = [m.start() for m in re.finditer("g1.globo", resp.text)]
print("g1.globo occurrences at indexes:", g1_matches)
for idx in g1_matches:
    print("Context:", resp.text[idx-50:idx+150])
