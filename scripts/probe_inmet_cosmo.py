from __future__ import annotations

from ftplib import FTP

HOST = "ftp.inmet.gov.br"
ROOT = "/cosmo"


def list_dir(ftp: FTP, path: str, depth: int = 0, max_depth: int = 2) -> None:
    indent = "  " * depth
    print(f"{indent}{path}")
    try:
        entries = list(ftp.mlsd(path))
    except Exception as exc:
        print(f"{indent}MLSD failed: {exc}")
        try:
            old = ftp.pwd()
            ftp.cwd(path)
            names = ftp.nlst()
            ftp.cwd(old)
            entries = [(n, {}) for n in names]
        except Exception as exc2:
            print(f"{indent}NLST failed: {exc2}")
            return

    for name, facts in entries[:120]:
        typ = facts.get("type", "?")
        size = facts.get("size", "")
        modify = facts.get("modify", "")
        print(f"{indent}- {name} type={typ} size={size} modify={modify}")

    if depth >= max_depth:
        return
    for name, facts in entries[:120]:
        if facts.get("type") == "dir" and name not in {".", ".."}:
            child = path.rstrip("/") + "/" + name
            list_dir(ftp, child, depth + 1, max_depth)


def main() -> None:
    ftp = FTP(HOST, timeout=30)
    ftp.login()
    print("WELCOME", ftp.getwelcome())
    print("PWD", ftp.pwd())
    list_dir(ftp, ROOT)
    ftp.quit()


if __name__ == "__main__":
    main()
