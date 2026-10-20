"""Save DrivenData pages (logged in) as text so a cloud session can read them.

Usage: DRIVENDATA_EMAIL=... DRIVENDATA_PASSWORD=... python scripts/fetch_pages.py <out_dir> <url> [...]
Each page is saved as <slug>.txt (visible text) and <slug>.links.txt (href + link text).
Paths like /competitions/306/... are resolved against https://www.drivendata.org.
Login is skipped when no credentials are set (public pages only).
"""

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "https://www.drivendata.org"


def slug(url: str) -> str:
    s = re.sub(r"^https?://", "", url).strip("/")
    return re.sub(r"[^A-Za-z0-9]+", "_", s)[:120] or "page"


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    urls = [u if u.startswith("http") else BASE + u for u in sys.argv[2:]]
    email, pw = os.environ.get("DRIVENDATA_EMAIL"), os.environ.get("DRIVENDATA_PASSWORD")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_context().new_page()
        if email and pw:
            page.goto(f"{BASE}/accounts/login/")
            page.fill("#id_login", email)
            page.fill("#id_password", pw)
            page.click('button[type="submit"]')
            page.wait_for_load_state("networkidle")
            print("login:", "ok" if "/accounts/login/" not in page.url else "FAILED")
        for url in urls:
            try:
                page.goto(url, timeout=60000)
                page.wait_for_load_state("networkidle")
                text = page.inner_text("body")
                links = page.eval_on_selector_all(
                    "a[href]", "els => els.map(e => e.href + '\\t' + e.innerText.trim().replace(/\\s+/g,' '))")
                # signed S3 links expire and are personal; keep only the path part
                links = [re.sub(r"\?X-Amz-[^\t]*", "?<signed>", l) for l in links]
                name = slug(url)
                (out / f"{name}.txt").write_text(f"URL: {url}\nFINAL: {page.url}\n\n{text}", encoding="utf-8")
                (out / f"{name}.links.txt").write_text("\n".join(links), encoding="utf-8")
                print(f"saved {name} ({len(text)} chars)")
            except Exception as e:  # keep going for the other pages
                print(f"FAILED {url}: {e}")
        browser.close()


if __name__ == "__main__":
    main()
