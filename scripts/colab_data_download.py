"""Download DrivenData competition data from GitHub Artifact in Google Colab.

Usage in Colab:
    !pip install requests
    import subprocess, os
    os.environ["GH_TOKEN"] = "ghp_xxx"  # or use Colab secrets
    subprocess.run(["python", "scripts/colab_data_download.py",
                    "--artifact", "drivendata-phonetic-data",
                    "--output", "data/phonetic"])
"""

import argparse
import io
import os
import sys
import zipfile

import requests

REPO = os.environ.get("GH_REPO", "yasumorishima/drivendata-comp")
GH_TOKEN = os.environ.get("GH_TOKEN", "")


def get_latest_artifact(repo: str, artifact_name: str, token: str) -> dict | None:
    """Find the latest artifact matching the given name."""
    url = f"https://api.github.com/repos/{repo}/actions/artifacts"
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}
    params = {"name": artifact_name, "per_page": 1}

    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()

    artifacts = data.get("artifacts", [])
    if not artifacts:
        return None
    return artifacts[0]


def download_artifact(artifact: dict, token: str, output_dir: str):
    """Download and extract an artifact ZIP."""
    url = artifact["archive_download_url"]
    headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github+json"}

    print(f"Downloading: {artifact['name']} ({artifact['size_in_bytes'] / 1024 / 1024:.1f} MB)")
    resp = requests.get(url, headers=headers, stream=True)
    resp.raise_for_status()

    content = io.BytesIO(resp.content)
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(content) as zf:
        zf.extractall(output_dir)

    files = list(os.listdir(output_dir))
    print(f"Extracted {len(files)} files to {output_dir}/")
    for f in sorted(files)[:20]:
        size = os.path.getsize(os.path.join(output_dir, f)) / 1024 / 1024
        print(f"  {f} ({size:.1f} MB)")
    if len(files) > 20:
        print(f"  ... and {len(files) - 20} more files")


def main():
    parser = argparse.ArgumentParser(description="Download DrivenData data from GitHub Artifact")
    parser.add_argument("--artifact", required=True, help="Artifact name (e.g. drivendata-phonetic-data)")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--repo", default=REPO, help="GitHub repo (owner/name)")
    args = parser.parse_args()

    token = GH_TOKEN
    if not token:
        print("ERROR: Set GH_TOKEN environment variable.")
        sys.exit(1)

    artifact = get_latest_artifact(args.repo, args.artifact, token)
    if not artifact:
        print(f"ERROR: Artifact '{args.artifact}' not found in {args.repo}.")
        print("Available artifact names: drivendata-word-data, drivendata-phonetic-data")
        sys.exit(1)

    if artifact.get("expired", False):
        print(f"ERROR: Artifact '{args.artifact}' has expired. Re-run Download Competition Data workflow.")
        sys.exit(1)

    download_artifact(artifact, token, args.output)
    print("Done.")


if __name__ == "__main__":
    main()
