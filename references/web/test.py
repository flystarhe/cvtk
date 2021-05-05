# python test.py 500 /workspace/images/test.png 7000
import sys
import time

import requests

args = sys.argv[1:]
command = " ".join(args)


times = int(args[0])
data = {"image": [args[1]] * 2}
url = f"http://localhost:{args[2]}/predict"
headers = {"content-type": "application/x-www-form-urlencoded"}


oks = 0
start_time = time.time()
for _ in range(times):
    response = requests.post(url, data=data, headers=headers)
    if response.status_code == 200:
        x = response.json()

        if x["status"] == 0:
            oks += 1
total_time = time.time() - start_time


status = dict(
    command=command,
    times=times,
    oks=oks,
    total_time=total_time,
    latest_x=x,
)
print(str(status))
