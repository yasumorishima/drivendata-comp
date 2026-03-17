"""Google Drive API helper for Colab GPU training pipeline.

Uses a Service Account to upload/download files and poll for results.
Requires: GOOGLE_SERVICE_ACCOUNT_KEY (JSON string) and DRIVE_SHARED_FOLDER_ID.
"""

import io
import json
import os
import sys
import time
from pathlib import Path

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload

SCOPES = ["https://www.googleapis.com/auth/drive"]


def get_drive_service():
    """Create Drive API service from Service Account credentials.

    Accepts GOOGLE_SERVICE_ACCOUNT_KEY as either raw JSON or base64-encoded JSON.
    Also supports GOOGLE_SERVICE_ACCOUNT_KEY_B64 (always base64).
    """
    import base64

    key_b64 = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY_B64", "")
    key_raw = os.environ.get("GOOGLE_SERVICE_ACCOUNT_KEY", "")

    key_json = ""
    if key_b64:
        key_json = base64.b64decode(key_b64).decode("utf-8")
    elif key_raw:
        # Try raw JSON first, fallback to base64 decode
        if key_raw.strip().startswith("{"):
            key_json = key_raw
        else:
            try:
                key_json = base64.b64decode(key_raw).decode("utf-8")
            except Exception:
                key_json = key_raw

    if not key_json:
        print("ERROR: GOOGLE_SERVICE_ACCOUNT_KEY or GOOGLE_SERVICE_ACCOUNT_KEY_B64 not set")
        sys.exit(1)

    creds = Credentials.from_service_account_info(
        json.loads(key_json), scopes=SCOPES
    )
    return build("drive", "v3", credentials=creds)


def find_or_create_folder(service, name: str, parent_id: str) -> str:
    """Find existing folder or create new one under parent."""
    query = (
        f"name='{name}' and '{parent_id}' in parents "
        f"and mimeType='application/vnd.google-apps.folder' and trashed=false"
    )
    results = service.files().list(q=query, fields="files(id,name)").execute()
    files = results.get("files", [])

    if files:
        return files[0]["id"]

    metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder",
        "parents": [parent_id],
    }
    folder = service.files().create(body=metadata, fields="id").execute()
    print(f"Created folder: {name} ({folder['id']})")
    return folder["id"]


def upload_file(service, local_path: str, folder_id: str, filename: str = "") -> str:
    """Upload a file to Drive folder. Returns file ID."""
    name = filename or Path(local_path).name

    # Check if file already exists (overwrite)
    query = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    existing = service.files().list(q=query, fields="files(id)").execute().get("files", [])
    if existing:
        service.files().delete(fileId=existing[0]["id"]).execute()

    metadata = {"name": name, "parents": [folder_id]}
    media = MediaFileUpload(local_path, resumable=True)
    result = service.files().create(body=metadata, media_body=media, fields="id").execute()
    print(f"Uploaded: {name} -> {result['id']}")
    return result["id"]


def upload_content(service, content: str, folder_id: str, filename: str) -> str:
    """Upload string content as a file to Drive folder."""
    import tempfile

    query = f"name='{filename}' and '{folder_id}' in parents and trashed=false"
    existing = service.files().list(q=query, fields="files(id)").execute().get("files", [])
    if existing:
        service.files().delete(fileId=existing[0]["id"]).execute()

    metadata = {"name": filename, "parents": [folder_id]}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(content)
        tmp_path = f.name

    try:
        media = MediaFileUpload(tmp_path, mimetype="application/json")
        result = service.files().create(body=metadata, media_body=media, fields="id").execute()
    finally:
        os.unlink(tmp_path)

    print(f"Uploaded content: {filename} -> {result['id']}")
    return result["id"]


def find_file(service, name: str, folder_id: str) -> dict | None:
    """Find a file by name in a folder."""
    query = f"name='{name}' and '{folder_id}' in parents and trashed=false"
    results = service.files().list(
        q=query, fields="files(id,name,modifiedTime,size)"
    ).execute()
    files = results.get("files", [])
    return files[0] if files else None


def download_file(service, file_id: str, local_path: str):
    """Download a file from Drive to local path."""
    request = service.files().get_media(fileId=file_id)
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "wb") as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                print(f"  Download progress: {int(status.progress() * 100)}%")
    print(f"Downloaded: {local_path}")


def poll_for_result(
    service, folder_id: str, filename: str = "result.json",
    timeout_minutes: int = 360, poll_interval_sec: int = 60,
) -> dict | None:
    """Poll Drive folder for a result file. Returns parsed JSON or None on timeout."""
    deadline = time.time() + timeout_minutes * 60
    attempt = 0

    while time.time() < deadline:
        attempt += 1
        result_file = find_file(service, filename, folder_id)

        if result_file:
            print(f"Found {filename} after {attempt} polls")
            # Download and parse
            request = service.files().get_media(fileId=result_file["id"])
            content = io.BytesIO()
            downloader = MediaIoBaseDownload(content, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            content.seek(0)
            return json.loads(content.read().decode("utf-8"))

        if attempt % 10 == 0:
            elapsed = (time.time() - (deadline - timeout_minutes * 60)) / 60
            print(f"Poll #{attempt}: no result yet ({elapsed:.0f}m elapsed)")

        time.sleep(poll_interval_sec)

    print(f"Timeout: {filename} not found after {timeout_minutes} minutes")
    return None


def list_files(service, folder_id: str) -> list[dict]:
    """List all files in a folder."""
    results = service.files().list(
        q=f"'{folder_id}' in parents and trashed=false",
        fields="files(id,name,mimeType,size,modifiedTime)",
        orderBy="modifiedTime desc",
    ).execute()
    return results.get("files", [])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Drive API helper")
    sub = parser.add_subparsers(dest="command")

    # List files
    ls_cmd = sub.add_parser("ls", help="List files in folder")
    ls_cmd.add_argument("--folder-id", required=True)

    # Upload
    up_cmd = sub.add_parser("upload", help="Upload file")
    up_cmd.add_argument("--file", required=True)
    up_cmd.add_argument("--folder-id", required=True)
    up_cmd.add_argument("--name", default="")

    args = parser.parse_args()
    svc = get_drive_service()

    if args.command == "ls":
        for f in list_files(svc, args.folder_id):
            size = int(f.get("size", 0)) / 1024
            print(f"  {f['name']} ({size:.1f} KB) [{f['id']}]")
    elif args.command == "upload":
        upload_file(svc, args.file, args.folder_id, args.name)
