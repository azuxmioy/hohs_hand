"""Minimal no-dependency ARCTIC/MANO downloader (POST auth, stream to disk).
Picks credentials by url-file name: 'mano'->MANO, 'smplx'->SMPLX, else ARCTIC.
Usage: python _arctic_dl.py <url_file> <out_folder>
"""
import os
import sys
import warnings

import requests

warnings.filterwarnings("ignore")

url_file, out_folder = sys.argv[1], sys.argv[2]
flag = "MANO" if "mano" in url_file else ("SMPLX" if "smplx" in url_file else "ARCTIC")
user = os.environ[flag + "_USERNAME"]
pw = os.environ[flag + "_PASSWORD"]
print("auth domain:", flag, "user:", user)

for url in open(url_file):
    url = url.strip()
    if not url:
        continue
    r = requests.post(url, data={"username": user, "password": pw},
                      stream=True, verify=False, allow_redirects=True)
    fn = url.split("/")[-1]
    if "mano_v1_2" in url:
        fn = "mano_v1_2.zip"
    elif "image" in url:
        fn = "/".join(url.split("/")[-2:])
    ctype = r.headers.get("content-type", "")
    if r.status_code != 200:
        print("HTTP", r.status_code, "for", fn, "ctype", ctype)
        continue
    out = os.path.join(out_folder, fn)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    total = 0
    with open(out, "wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
                total += len(chunk)
    print("%s: %.1f MB  http=%d  ctype=%s" % (fn, total / 1e6, r.status_code, ctype))
print("done")
