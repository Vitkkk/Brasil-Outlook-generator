from __future__ import annotations

from ftplib import FTP
import re
from urllib.request import Request, urlopen
from urllib.parse import urljoin

HOST = "ftp.inmet.gov.br"
ROOT = "/cosmo"
VIME = "https://vime.inmet.gov.br/"
API = "https://apivime.inmet.gov.br"


def fetch(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json,text/plain,*/*"})
    with urlopen(req, timeout=30) as r:
        data = r.read()
        print("HTTP", r.status, url, "CTYPE", r.headers.get("content-type"))
    return data.decode("utf-8", "ignore")


def probe_vime() -> None:
    print("=== VIME WEB PROBE ===")
    html = fetch(VIME)
    assets = re.findall(r'(?:src|href)=["\']([^"\']+\.(?:js|json))["\']', html)
    print("ASSETS", assets)
    for asset in assets[:20]:
        url = urljoin(VIME, asset)
        try:
            text = fetch(url)
        except Exception as exc:
            print("ASSET FAIL", url, exc)
            continue
        print("ASSET", url, "LEN", len(text))
        for needle in ["apivime", "modelos", "rodadas", "previs", "produto", "cosmo"]:
            pos = 0
            while True:
                i = text.lower().find(needle, pos)
                if i < 0:
                    break
                print("CTX", text[max(0, i-250): i+500].replace("\n", " "))
                pos = i + len(needle)


def probe_api() -> None:
    print("=== VIME API PROBE ===")
    for path in ["/", "/modelos", "/modelos/", "/api/modelos", "/produtos", "/rodadas"]:
        url = API + path
        try:
            text = fetch(url)
            print("BODY", path, text[:10000])
        except Exception as exc:
            print("API FAIL", path, repr(exc))


def probe_ftp() -> None:
    print("=== FTP PROBE ===")
    ftp = FTP(HOST, timeout=30)
    try:
        ftp.login()
        print("WELCOME", ftp.getwelcome())
        print("PWD", ftp.pwd())
        print("NLST", ftp.nlst(ROOT)[:100])
        ftp.quit()
    except Exception as exc:
        print("FTP FAILED", repr(exc))
        try:
            ftp.close()
        except Exception:
            pass


def main() -> None:
    probe_ftp()
    probe_vime()
    probe_api()


if __name__ == "__main__":
    main()
