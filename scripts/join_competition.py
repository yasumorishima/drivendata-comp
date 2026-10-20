"""Register the DrivenData account for a competition (clicks "Compete!" and accepts the rules).

Usage: DRIVENDATA_EMAIL=... DRIVENDATA_PASSWORD=... python scripts/join_competition.py <out_dir> <comp_path>
  comp_path like /competitions/297/whats-up-docs/
Saves text + screenshot of each step to <out_dir> so a cloud session can see what happened.
Only run for competitions the user has decided to enter: joining accepts the competition rules.
"""

import os
import re
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "https://www.drivendata.org"


def snap(page, out: Path, name: str):
    page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    (out / f"{name}.txt").write_text(f"URL: {page.url}\n\n{page.inner_text('body')}", encoding="utf-8")
    print(f"[{name}] {page.url}")


def main():
    out, comp = Path(sys.argv[1]), sys.argv[2].rstrip("/") + "/"
    out.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        page = p.chromium.launch(headless=True).new_context().new_page()
        page.goto(f"{BASE}/accounts/login/")
        page.fill("#id_login", os.environ["DRIVENDATA_EMAIL"])
        page.fill("#id_password", os.environ["DRIVENDATA_PASSWORD"])
        page.click('button[type="submit"]')
        page.wait_for_load_state("networkidle")
        if "/accounts/login/" in page.url:
            sys.exit("login failed")

        page.goto(BASE + comp)
        page.wait_for_load_state("networkidle")
        snap(page, out, "1_home")
        btn = page.get_by_role("link", name=re.compile(r"^\s*(Compete|Register)!?\s*$", re.I))
        if btn.count() == 0:
            btn = page.get_by_role("button", name=re.compile(r"^\s*(Compete|Register)!?\s*$", re.I))
        if btn.count() == 0:
            print("no Compete!/Register! button: already joined?")
        else:
            btn.first.click()
            page.wait_for_load_state("networkidle")
            snap(page, out, "2_after_compete")
            # rules agreement page: tick every checkbox, then press the agree/submit button
            for cb in page.locator("input[type=checkbox]").all():
                if cb.is_visible() and not cb.is_checked():
                    cb.check()
            agree = page.get_by_role("button", name=re.compile(r"agree|accept|join|register|submit|continue", re.I))
            if agree.count():
                agree.first.click()
                page.wait_for_load_state("networkidle")
            snap(page, out, "3_after_agree")

        for sub in ("data/", "submissions/"):
            page.goto(BASE + comp + sub)
            page.wait_for_load_state("networkidle")
            snap(page, out, "4_" + sub.strip("/"))


if __name__ == "__main__":
    main()
