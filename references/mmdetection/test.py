import sys

import requests

args = sys.argv[1:]
url = f"http://localhost:{args[0]}/predict"
data = {"image": "/workspace/images/test.png"}
headers = {"content-type": "application/x-www-form-urlencoded"}

# python test.py 7000 5
for i in range(int(args[1])):
    response = requests.post(url, data=data, headers=headers)
    print(i, response.status_code)
    print(response.text)
