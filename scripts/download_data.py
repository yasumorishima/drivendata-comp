"""DrivenData competition data downloader using Playwright."""

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
DOWNLOAD_DIR = Path(os.environ.get("DOWNLOAD_DIR", "data"))


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


def join_competition(page, comp_id: int, slug: str):
    """Join a competition if not already joined."""
    url = f"{BASE_URL}/competitions/{comp_id}/{slug}/"
    page.goto(url)
    page.wait_for_load_state("networkidle")

    # Only click visible join buttons
    join_btn = page.query_selector(
        'button:visible:has-text("Join"), a:visible:has-text("Join")'
    )
    if join_btn and join_btn.is_visible():
        try:
            join_btn.click(timeout=10_000)
            page.wait_for_load_state("networkidle")
            # Handle any confirmation
            confirm = page.query_selector(
                'button:visible:has-text("Confirm"), button:visible:has-text("Accept"), '
                'button:visible:has-text("Agree"), input:visible[type="submit"]'
            )
            if confirm and confirm.is_visible():
                confirm.click(timeout=10_000)
                page.wait_for_load_state("networkidle")
            print(f"Joined competition: {slug} (ID: {comp_id})")
        except Exception as e:
            print(f"Join button found but click failed (likely already joined): {e}")
    else:
        print(f"Already joined or no join button: {slug} (ID: {comp_id})")


def download_data(page, comp_id: int, slug: str, track_dir: Path):
    """Download all data files from a competition's data page."""
    track_dir.mkdir(parents=True, exist_ok=True)

    data_url = f"{BASE_URL}/competitions/{comp_id}/{slug}/data/"
    page.goto(data_url)
    page.wait_for_load_state("networkidle")
    time.sleep(2)

    # Find all download links
    download_links = page.query_selector_all(
        'a[href*="download"], a:has-text("Download"), '
        'a[href*=".zip"], a[href*=".tar"], a[href*=".csv"], '
        'a[href*=".json"], a[href*=".jsonl"]'
    )

    if not download_links:
        # Try alternative: look for download buttons
        download_links = page.query_selector_all(
            'button:has-text("Download"), a.btn:has-text("Download")'
        )

    if not download_links:
        print(f"  WARNING: No download links found on {data_url}")
        print(f"  Page title: {page.title()}")
        print(f"  Current URL: {page.url}")
        # Save page and screenshot for debugging
        debug_file = track_dir / "debug_page.html"
        debug_file.write_text(page.content(), encoding="utf-8")
        print(f"  Saved page HTML to {debug_file}")
        screenshot_file = track_dir / "debug_screenshot.png"
        page.screenshot(path=str(screenshot_file), full_page=True)
        print(f"  Saved screenshot to {screenshot_file}")
        # List all links on page for debugging
        all_links = page.query_selector_all("a[href]")
        print(f"  All links on page ({len(all_links)}):")
        for link in all_links[:30]:
            href = link.get_attribute("href") or ""
            text = (link.inner_text() or "").strip()[:60]
            if text:
                print(f"    {text} -> {href}")
        return

    seen_urls = set()
    for i, link in enumerate(download_links):
        href = link.get_attribute("href") or ""
        text = (link.inner_text() or "").strip()

        # Deduplicate
        if href in seen_urls:
            continue
        seen_urls.add(href)

        print(f"  [{i+1}/{len(download_links)}] Downloading: {text or href}")

        try:
            with page.expect_download(timeout=300_000) as download_info:
                link.click()
            download = download_info.value
            filename = download.suggested_filename
            save_path = track_dir / filename
            download.save_as(str(save_path))
            size_mb = save_path.stat().st_size / (1024 * 1024)
            print(f"    Saved: {save_path} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"    Failed: {e}")
            # If click navigated instead of downloading, go back
            if data_url not in page.url:
                page.goto(data_url)
                page.wait_for_load_state("networkidle")
                time.sleep(1)


def main():
    email = os.environ.get("DRIVENDATA_EMAIL")
    password = os.environ.get("DRIVENDATA_PASSWORD")
    if not email or not password:
        print("ERROR: Set DRIVENDATA_EMAIL and DRIVENDATA_PASSWORD env vars.")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        # Login
        login(page, email, password)

        for track, info in COMPETITIONS.items():
            print(f"\n=== {track.upper()} TRACK (ID: {info['id']}) ===")
            track_dir = DOWNLOAD_DIR / track

            # Join competition
            join_competition(page, info["id"], info["slug"])

            # Download data
            print(f"Downloading data...")
            download_data(page, info["id"], info["slug"], track_dir)

        browser.close()

    # Summary
    print("\n=== DOWNLOAD SUMMARY ===")
    for track_dir in DOWNLOAD_DIR.iterdir():
        if track_dir.is_dir():
            files = list(track_dir.glob("*"))
            files = [f for f in files if f.name != "debug_page.html"]
            total = sum(f.stat().st_size for f in files if f.is_file())
            print(f"{track_dir.name}/: {len(files)} files, {total / (1024*1024):.1f} MB")
            for f in sorted(files):
                print(f"  {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")


if __name__ == "__main__":
    main()
