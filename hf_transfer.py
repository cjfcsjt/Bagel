#!/usr/bin/env python3
"""
Hugging Face Hub File Upload/Download Script
Used to upload/download files to/from Hugging Face Hub
"""

import os
import argparse
from huggingface_hub import HfApi, create_repo, hf_hub_download, snapshot_download

def upload_to_hf(file_path, repo_id, repo_type="dataset", token=None):
    """
    Upload file to Hugging Face Hub
    
    Args:
        file_path (str): Local file path to upload
        repo_id (str): Hugging Face repo ID (format: username/repo-name)
        repo_type (str): Repo type, one of "model", "dataset", "space"
        token (str): Hugging Face token. If None, use HF_TOKEN env var.
    """
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} does not exist")
        return False
    
    # Initialize Hugging Face API
    api = HfApi(token=token)
    
    try:
        # Create repo (if it doesn't exist)
        create_repo(repo_id, repo_type=repo_type, exist_ok=True, token=token)
        print(f"Repo {repo_id} created or already exists")
        
        # Upload file
        print(f"Uploading file {file_path}...")
        api.upload_file(
            path_or_fileobj=file_path,
            path_in_repo=os.path.basename(file_path),
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ File uploaded successfully!")
        print(f"🔗 Access URL: https://huggingface.co/{repo_id}")
        return True
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        return False


def download_from_hf(repo_id, filename=None, local_dir=None, repo_type="dataset", token=None):
    """
    Download file(s) from Hugging Face Hub

    Args:
        repo_id (str): Hugging Face repo ID (format: username/repo-name)
        filename (str): Specific file to download. If None, download the entire repo.
        local_dir (str): Local directory to save downloaded files
        repo_type (str): Repo type, one of "model", "dataset", "space"
        token (str): Hugging Face token. If None, use HF_TOKEN env var.
    """

    # Create local directory if specified
    if local_dir is not None:
        os.makedirs(local_dir, exist_ok=True)

    api = HfApi(token=token)

    try:
        download_kwargs = dict(
            repo_id=repo_id,
            repo_type=repo_type,
            token=token,
        )
        if local_dir is not None:
            download_kwargs["local_dir"] = local_dir

        if filename:
            # Download a single file
            print(f"Downloading {filename} from {repo_id}...")
            download_kwargs["filename"] = filename
            path = hf_hub_download(**download_kwargs)
            print(f"✅ File downloaded to: {path}")
        else:
            # Download the entire repo
            print(f"Downloading entire repo {repo_id}...")
            path = snapshot_download(**download_kwargs)
            print(f"✅ Repo downloaded to: {path}")
        return True

    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Upload/Download files to/from Hugging Face Hub")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Upload subcommand
    upload_parser = subparsers.add_parser("upload", help="Upload a file to Hugging Face Hub")
    upload_parser.add_argument("--file", "-f", default="data/download/combined.zip",
                               help="File path to upload (default: data/download/combined.zip)")
    upload_parser.add_argument("--repo", "-r", required=True,
                               help="Hugging Face repo ID (format: username/repo-name)")
    upload_parser.add_argument("--type", "-t", default="dataset",
                               choices=["model", "dataset", "space"],
                               help="Repo type (default: dataset)")
    upload_parser.add_argument("--token", help="Hugging Face token (optional, defaults to HF_TOKEN env var)")

    # Download subcommand
    download_parser = subparsers.add_parser("download", help="Download file(s) from Hugging Face Hub")
    download_parser.add_argument("--repo", "-r", required=True,
                                  help="Hugging Face repo ID (format: username/repo-name)")
    download_parser.add_argument("--file", "-f", default=None,
                                  help="Specific filename to download (if omitted, download entire repo)")
    download_parser.add_argument("--local-dir", "-d", default=None,
                                  help="Local directory to save files (default: HF_HOME env var)")
    download_parser.add_argument("--type", "-t", default="dataset",
                                  choices=["model", "dataset", "space"],
                                  help="Repo type (default: dataset)")
    download_parser.add_argument("--token", help="Hugging Face token (optional, defaults to HF_TOKEN env var)")

    args = parser.parse_args()

    if args.command == "upload":
        upload_to_hf(args.file, args.repo, args.type, args.token)
    elif args.command == "download":
        download_from_hf(args.repo, args.file, args.local_dir, args.type, args.token)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()