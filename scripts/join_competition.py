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
            page.wait_for_timeout(3000)
            page.wait_for_load_state("networkidle")
            snap(page, out, "2_after_compete")
            # the rules agreement opens in a modal dialog: act inside it when present
            modal = page.locator(".modal.show, [role=dialog]:visible, .modal:visible").first
            scope = modal if modal.count() else page
            (out / "2_modal.html").write_text(scope.inner_html() if modal.count() else page.content(), encoding="utf-8")
            for cb in scope.locator("input[type=checkbox]").all():
                try:
                    if not cb.is_checked():
                        cb.check(force=True)
                except Exception as e:
                    print("checkbox:", e)
            agree = scope.locator("button, input[type=submit]").filter(
                has_text=re.compile(r"agree|accept|join|register|compete|submit|continue", re.I))
            if agree.count() == 0:
                agree = scope.locator("input[type=submit]")
            print("agree buttons:", agree.count())
            if agree.count():
                agree.first.click()
                page.wait_for_timeout(3000)
                page.wait_for_load_state("networkidle")
            # an optional pre-registration survey may follow: leave the answers at their
            # defaults ("Prefer not to say") and press "Submit and join"
            for _ in range(2):
                join = page.locator(".modal.show button, .modal.show input[type=submit]").filter(
                    has_text=re.compile(r"submit and join|join|I agree", re.I))
                if not join.count():
                    break
                print("pressing:", join.first.inner_text())
                join.first.click()
                page.wait_for_timeout(3000)
                page.wait_for_load_state("networkidle")
            snap(page, out, "3_after_agree")

        for sub in ("data/", "submissions/"):
            page.goto(BASE + comp + sub)
            page.wait_for_load_state("networkidle")
            snap(page, out, "4_" + sub.strip("/"))


if __name__ == "__main__":
    main()
