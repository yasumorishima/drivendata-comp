"""DrivenData code submission uploader using Playwright.

Usage:
    python scripts/submit_code.py --zip submission.zip --track phonetic
"""

import argparse
import os
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "https://www.drivendata.org"
COMPETITIONS = {
    "word": {"id": 308, "slug": "childrens-word-asr"},
    "phonetic": {"id": 309, "slug": "childrens-phonetic-asr"},
}


def login(page, email: str, password: str):
    """Log in to DrivenData."""
    page.goto(f"{BASE_URL}/accounts/login/")
    page.wait_for_selector("#id_login")
    page.fill("#id_login", email)
    page.fill("#id_password", password)
    page.click('button[type="submit"]')
    page.wait_for_load_state("networkidle")

    if "/accounts/login/" in page.url:
        print("ERROR: Login failed. Check credentials.")
        sys.exit(1)
    print(f"Logged in successfully. Redirected to: {page.url}")


def submit(page, comp_id: int, slug: str, zip_path: Path):
    """Upload submission ZIP to DrivenData."""
    submit_url = f"{BASE_URL}/competitions/{comp_id}/{slug}/submissions/"
    print(f"Navigating to: {submit_url}")
    page.goto(submit_url)
    page.wait_for_load_state("networkidle")
    time.sleep(3)

    # Screenshot before upload
    page.screenshot(path="submit_before.png", full_page=True)
    print(f"Page title: {page.title()}")

    if "Page not found" in page.title():
        print("ERROR: Submissions page returned 404. Is the competition joined?")
        return False

    # Find file input (may be hidden)
    file_input = page.query_selector('input[type="file"]')
    if not file_input:
        # Try looking in iframes or shadow DOM
        print("Looking for file upload input...")
        file_inputs = page.query_selector_all('input[type="file"]')
        print(f"Found {len(file_inputs)} file inputs")
        if not file_inputs:
            print("ERROR: No file input found on page")
            page.screenshot(path="submit_error.png", full_page=True)
            return False
        file_input = file_inputs[0]

    # Upload the ZIP
    print(f"Uploading: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    file_input.set_input_files(str(zip_path))
    time.sleep(2)

    # Screenshot after file selection
    page.screenshot(path="submit_file_selected.png", full_page=True)

    # Click submit button
    submit_btn = page.query_selector('button[type="submit"], input[type="submit"]')
    if not submit_btn:
        # Try finding by text
        submit_btn = page.get_by_role("button", name="Submit")
    if not submit_btn:
        submit_btn = page.query_selector('button:has-text("Submit"), button:has-text("submit")')

    if submit_btn:
        print("Clicking submit button...")
        submit_btn.click()
        time.sleep(5)
        page.wait_for_load_state("networkidle")
        page.screenshot(path="submit_after.png", full_page=True)
        print(f"After submit URL: {page.url}")
        print("Submission uploaded!")
        return True
    else:
        print("ERROR: Submit button not found")
        page.screenshot(path="submit_no_button.png", full_page=True)
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="Path to submission.zip")
    parser.add_argument("--track", required=True, choices=["word", "phonetic"])
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"ERROR: {zip_path} not found")
        sys.exit(1)

    email = os.environ.get("DRIVENDATA_EMAIL", "")
    password = os.environ.get("DRIVENDATA_PASSWORD", "")
    if not email or not password:
        print("ERROR: Set DRIVENDATA_EMAIL and DRIVENDATA_PASSWORD env vars")
        sys.exit(1)

    comp = COMPETITIONS[args.track]

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        login(page, email, password)
        success = submit(page, comp["id"], comp["slug"], zip_path)

        browser.close()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
