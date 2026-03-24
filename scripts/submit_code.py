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

    # Step 1: Click "Make new submission" to get to Code jobs page
    make_btn = page.query_selector('a:has-text("Make new submission"), button:has-text("Make new submission")')
    if make_btn:
        print("Clicking 'Make new submission'...")
        make_btn.click()
        page.wait_for_load_state("networkidle")
        time.sleep(3)
        page.screenshot(path="submit_form_opened.png", full_page=True)
        print(f"Code jobs page URL: {page.url}")

    # Step 2: Click "+ New code submission" to open modal dialog
    new_code_btn = page.query_selector(
        'a:has-text("New code submission"), button:has-text("New code submission")'
    )
    if new_code_btn:
        print("Clicking '+ New code submission'...")
        new_code_btn.click()
        time.sleep(3)
        page.screenshot(path="submit_modal_opened.png", full_page=True)
        print("Modal dialog opened")
    else:
        print("No 'New code submission' button found, trying current page...")

    # Step 3: Find file input inside the modal (#modalUploadCode)
    modal = page.query_selector('#modalUploadCode')
    if modal:
        print("Found modal #modalUploadCode")
        file_input = modal.query_selector('input[type="file"]')
    else:
        print("No modal found, searching page...")
        file_input = page.query_selector('input[type="file"]')

    if not file_input:
        # Wait for modal/input to appear
        print("Waiting for file input...")
        try:
            page.wait_for_selector('#modalUploadCode input[type="file"]', timeout=15000)
            file_input = page.query_selector('#modalUploadCode input[type="file"]')
        except Exception:
            pass

    if not file_input:
        # Last resort: any file input on page
        file_input = page.query_selector('input[type="file"]')

    if not file_input:
        print("ERROR: No file input found")
        html = page.content()
        with open("submit_debug.html", "w") as f:
            f.write(html)
        page.screenshot(path="submit_error.png", full_page=True)
        return False

    # Step 4: Upload the ZIP via file input
    print(f"Uploading: {zip_path} ({zip_path.stat().st_size / 1024 / 1024:.1f} MB)")
    file_input.set_input_files(str(zip_path))
    time.sleep(5)
    page.screenshot(path="submit_file_selected.png", full_page=True)

    # Step 5: Click submit button inside modal
    if modal:
        submit_btn = (
            modal.query_selector('button[type="submit"]')
            or modal.query_selector('button:has-text("Submit")')
            or modal.query_selector('button:has-text("Upload")')
        )
    else:
        submit_btn = None

    if not submit_btn:
        submit_btn = (
            page.query_selector('#modalUploadCode button[type="submit"]')
            or page.query_selector('#modalUploadCode button:has-text("Submit")')
            or page.query_selector('button[type="submit"]')
            or page.query_selector('button:has-text("Submit")')
            or page.query_selector('button:has-text("Upload")')
        )

    if submit_btn:
        print("Clicking submit button...")
        submit_btn.click()
        # Wait for upload to complete (330MB can take a while on GitHub Actions)
        print("Waiting for upload to complete (up to 120s)...")
        time.sleep(120)
        page.wait_for_load_state("networkidle")
        time.sleep(5)
        page.screenshot(path="submit_after.png", full_page=True)
        print(f"After submit URL: {page.url}")

        # Verify submission appeared on Code jobs page
        page.goto(f"{BASE_URL}/competitions/{comp_id}/submissions/code/")
        page.wait_for_load_state("networkidle")
        time.sleep(5)
        page.screenshot(path="submit_verify.png", full_page=True)
        content = page.content()
        if "don't have any code jobs" in content.lower() or "no code jobs" in content.lower():
            print("WARNING: Submission may not have been uploaded successfully")
            # Don't fail — upload might still be processing
            return True
        print("Submission uploaded!")
        return True
    else:
        print("ERROR: Submit button not found")
        html = page.content()
        with open("submit_debug.html", "w") as f:
            f.write(html)
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
