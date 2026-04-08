from huggingface_hub import HfApi

api = HfApi()

# Define your details
repo_id = "Wvidit/Qwen3-4B"
local_folder = "/home/vidit68/Synnapse/grpo_output/checkpoint-1000"

# Create the repo (it's okay if it already exists)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

# Upload the entire folder
api.upload_folder(
    folder_path=local_folder,
    repo_id=repo_id,
    repo_type="model",
)
