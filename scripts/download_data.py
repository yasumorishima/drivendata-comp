"""DrivenData competition data downloader using Playwright."""

import os
import re
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

# File extensions that indicate real downloads
DOWNLOAD_EXTENSIONS = re.compile(r"\.(zip|tar|gz|csv|json|jsonl|flac|wav|tsv)(\?|$)", re.I)


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


def download_data(page, comp_id: int, slug: str, track_dir: Path):
    """Download all data files from a competition's data page."""
    track_dir.mkdir(parents=True, exist_ok=True)

    data_url = f"{BASE_URL}/competitions/{comp_id}/{slug}/data/"
    page.goto(data_url)
    page.wait_for_load_state("networkidle")
    time.sleep(3)

    print(f"  Data page URL: {page.url}")
    print(f"  Data page title: {page.title()}")

    # Save data page screenshot
    page.screenshot(path=str(track_dir / "data_page.png"), full_page=True)

    if "Page not found" in page.title():
        print("  ERROR: Data page returned 404. Is the competition joined?")
        return False

    # Save HTML for analysis
    html = page.content()
    (track_dir / "data_page.html").write_text(html, encoding="utf-8")

    # Collect all links with their hrefs
    all_anchors = page.query_selector_all("a[href]")
    download_candidates = []
    for a in all_anchors:
        try:
            href = a.get_attribute("href") or ""
            text = (a.inner_text() or "").strip()[:80]
        except Exception:
            continue

        # Filter: only actual file downloads
        is_file = bool(DOWNLOAD_EXTENSIONS.search(href))
        is_download_endpoint = "download" in href.lower() and href != data_url
        # Exclude navigation links (same-site page links without file extensions)
        is_nav = bool(re.match(r"^/competitions/\d+/[^?]*/$", href))

        if (is_file or is_download_endpoint) and not is_nav:
            download_candidates.append({"href": href, "text": text, "element": a})

    # Deduplicate by href
    seen = set()
    unique = []
    for c in download_candidates:
        if c["href"] not in seen:
            seen.add(c["href"])
            unique.append(c)

    print(f"  Found {len(unique)} download links")
    for c in unique:
        print(f"    {c['text'][:50]} -> {c['href'][:100]}")

    if not unique:
        print("  WARNING: No download links found. Listing all links on page:")
        for a in all_anchors[:40]:
            try:
                href = a.get_attribute("href") or ""
                text = (a.inner_text() or "").strip()[:60]
                if text and not href.startswith("#"):
                    print(f"    {text} -> {href[:100]}")
            except Exception:
                pass
        return False

    success_count = 0
    for i, candidate in enumerate(unique):
        href = candidate["href"]
        text = candidate["text"]
        elem = candidate["element"]
        print(f"  [{i+1}/{len(unique)}] Downloading: {text or href[:60]}")

        try:
            with page.expect_download(timeout=300_000) as download_info:
                elem.click()
            download = download_info.value
            filename = download.suggested_filename
            save_path = track_dir / filename
            download.save_as(str(save_path))
            size_mb = save_path.stat().st_size / (1024 * 1024)
            print(f"    OK: {filename} ({size_mb:.1f} MB)")
            success_count += 1
        except Exception as e:
            err_msg = str(e).split("\n")[0][:100]
            print(f"    SKIP: {err_msg}")
            # If navigation happened, go back to data page
            if "/data/" not in page.url:
                page.goto(data_url)
                page.wait_for_load_state("networkidle")
                time.sleep(2)

    print(f"  Downloaded {success_count}/{len(unique)} files")
    return success_count > 0


def main():
    email = os.environ.get("DRIVENDATA_EMAIL")
    password = os.environ.get("DRIVENDATA_PASSWORD")
    if not email or not password:
        print("ERROR: Set DRIVENDATA_EMAIL and DRIVENDATA_PASSWORD env vars.")
        sys.exit(1)

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    any_success = False

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(accept_downloads=True)
        page = context.new_page()

        login(page, email, password)

        for track, info in COMPETITIONS.items():
            print(f"\n=== {track.upper()} TRACK (ID: {info['id']}) ===")
            track_dir = DOWNLOAD_DIR / track
            result = download_data(page, info["id"], info["slug"], track_dir)
            if result:
                any_success = True

        browser.close()

    # Summary
    print("\n=== DOWNLOAD SUMMARY ===")
    for track_dir in sorted(DOWNLOAD_DIR.iterdir()):
        if track_dir.is_dir():
            files = [
                f for f in track_dir.glob("*")
                if f.is_file() and f.suffix not in (".html", ".png")
            ]
            total = sum(f.stat().st_size for f in files)
            print(f"{track_dir.name}/: {len(files)} files, {total / (1024*1024):.1f} MB")
            for f in sorted(files):
                print(f"  {f.name} ({f.stat().st_size / (1024*1024):.1f} MB)")

    if not any_success:
        print("\nERROR: No files downloaded from any track.")
        sys.exit(1)


if __name__ == "__main__":
    main()
