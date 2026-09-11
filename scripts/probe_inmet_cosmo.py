from __future__ import annotations

from ftplib import FTP
import re
from urllib.request import Request, urlopen
from urllib.parse import urljoin

HOST = "ftp.inmet.gov.br"
ROOT = "/cosmo"
VIME = "https://vime.inmet.gov.br/"


def fetch(url: str) -> str:
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r:
        data = r.read()
    return data.decode("utf-8", "ignore")


def probe_vime() -> None:
    print("=== VIME WEB PROBE ===")
    html = fetch(VIME)
    print("INDEX", html[:3000])
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
        hits = sorted(set(re.findall(r'https?://[^"\'\\\s]+|/[A-Za-z0-9_./-]*(?:api|cosmo|modelo|model)[A-Za-z0-9_?=&./-]*', text, re.I)))
        for h in hits[:150]:
            print("HIT", h[:500])


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


if __name__ == "__main__":
    main()
