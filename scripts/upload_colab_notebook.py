"""Upload Colab ASR runner notebook to Google Drive.

One-time setup: uploads the notebook to Drive so Colab can open it.
After upload, open the notebook in Colab and start the main loop.

Usage:
    export GOOGLE_SERVICE_ACCOUNT_KEY='...'
    export DRIVE_SHARED_FOLDER_ID='...'
    python scripts/upload_colab_notebook.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from drive_helper import get_drive_service, find_or_create_folder, upload_file


def main():
    service = get_drive_service()
    root_id = os.environ["DRIVE_SHARED_FOLDER_ID"]

    # Create drivendata-pasketti folder
    comp_folder_id = find_or_create_folder(service, "drivendata-pasketti", root_id)

    # Upload notebook
    notebook_path = os.path.join(
        os.path.dirname(__file__), "..", "colab", "drivendata_asr_runner.ipynb"
    )
    file_id = upload_file(service, notebook_path, comp_folder_id, "drivendata_asr_runner.ipynb")

    print(f"\nNotebook uploaded successfully!")
    print(f"File ID: {file_id}")
    print(f"\nOpen in Colab:")
    print(f"  https://colab.research.google.com/drive/{file_id}")
    print(f"\nOr find it in Drive: kaggle/drivendata-pasketti/drivendata_asr_runner.ipynb")


if __name__ == "__main__":
    main()
