import sys
import requests


url = f"http://localhost:{sys.argv[1]}/predict"
data = {"image": "/workspace/test.png", "gc": "none"}


response = requests.post(url, data=data)
print(response.status_code)
print(response.json())
