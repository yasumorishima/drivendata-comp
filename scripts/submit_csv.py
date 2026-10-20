"""Upload a prediction file to a DrivenData competition and read back the score.

Usage: DRIVENDATA_EMAIL=... DRIVENDATA_PASSWORD=... \
       python scripts/submit_csv.py <comp_path> <file> <description> <out_dir>
  comp_path like /competitions/297/whats-up-docs/
Writes screenshots/text of each step to <out_dir>, and <out_dir>/result.txt with the
submissions page text after scoring (best score, submissions left today, the table).
Exit 2 when no submission is left today (nothing is uploaded).
"""

import os
import re
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE = "https://www.drivendata.org"


def snap(page, out, name):
    page.screenshot(path=str(out / f"{name}.png"), full_page=True)
    (out / f"{name}.txt").write_text(f"URL: {page.url}\n\n{page.inner_text('body')}", encoding="utf-8")
    print(f"[{name}] {page.url}")


def main():
    comp, path, desc, out = sys.argv[1].rstrip("/") + "/", Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4])
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

        subs = BASE + comp + "submissions/"
        page.goto(subs)
        page.wait_for_load_state("networkidle")
        snap(page, out, "1_submissions")
        body = page.inner_text("body")
        m = re.search(r"You have (\d+) of (\d+) submissions left today", body)
        print("left today:", m.group(0) if m else "unknown")
        if m and m.group(1) == "0":
            (out / "result.txt").write_text(body, encoding="utf-8")
            sys.exit(2)

        page.get_by_text(re.compile(r"Make new submission", re.I)).first.click()
        page.wait_for_timeout(3000)
        snap(page, out, "2_form")
        scope = page.locator(".modal.show").first if page.locator(".modal.show").count() else page
        scope.locator("input[type=file]").first.set_input_files(str(path))
        for sel in ("textarea", "input[name*=description i]", "input[type=text]"):
            box = scope.locator(sel)
            if box.count():
                try:
                    box.first.fill(desc[:250])
                    break
                except Exception as e:
                    print("description:", e)
        page.wait_for_timeout(1000)
        snap(page, out, "3_filled")
        btn = scope.locator("button, input[type=submit]").filter(has_text=re.compile(r"submit|upload", re.I))
        if not btn.count():
            btn = scope.locator("button[type=submit], input[type=submit]")
        btn.first.click()
        page.wait_for_timeout(5000)
        page.wait_for_load_state("networkidle")
        snap(page, out, "4_after_submit")

        # scoring a CSV usually takes seconds to a few minutes
        text = ""
        for _ in range(30):
            page.goto(subs)
            page.wait_for_load_state("networkidle")
            text = page.inner_text("body")
            if not re.search(r"pending|running|scoring|queued", text, re.I):
                break
            time.sleep(20)
        snap(page, out, "5_scored")
        (out / "result.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
