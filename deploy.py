from huggingface_hub import HfApi, upload_folder
from pathlib import Path
# Set your repo info
username = "b3rian"
repo_name = "streamlit-ui"
local_dir =  Path(__file__).resolve().parent  # Automatically detect current folder
repo_type = "space"
space_sdk = "docker" 

# 1. Create the space (skip if already created)
api = HfApi()
api.create_repo(
    repo_id=f"{username}/{repo_name}",
    repo_type=repo_type,
    space_sdk=space_sdk,
    exist_ok=True  # Don't fail if it already exists
)

# 2. Upload the entire folder to the space
upload_folder(
    repo_id=f"{username}/{repo_name}",
    folder_path=local_dir,
    repo_type=repo_type
)

print(f"✅ Deployed to https://huggingface.co/spaces/{username}/{repo_name}")
