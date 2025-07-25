# pip install 'huggingface_hub[cli]' argparse gdown

import os
from huggingface_hub import hf_hub_download
import argparse
import gdown
import os

def download_model(repo_id, files, target_folder):
    os.makedirs(target_folder, exist_ok=True)
    for filename in files:
        file_path = os.path.join(target_folder, filename)
        if os.path.exists(file_path):
            print(f"Skipping {filename}: already exists.")
            continue

        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=target_folder,
            local_dir=target_folder,
            local_dir_use_symlinks=False
        )

def download_voice(output_dir="voices"):
    os.makedirs(output_dir, exist_ok=True)
    folder_id = "1OaRZ36YIgCWOXFZ1XWJjIOblr4-xsH6_"
    gdown.download_folder(id=folder_id, output=output_dir, quiet=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download tts models')
    parser.add_argument('--target_folder', type=str, default="f5tts", help='Output directory for tts models')
    parser.add_argument('--voice_folder', type=str, default="voices", help='Output directory for tts voices')

    args = parser.parse_args()

    download_model(
        "PapaRazi/Ijazah_Palsu_V2",
        [
            "model_last_v2.safetensors",
            "setting.json",
            "vocab.txt"
        ],
        args.target_folder
    )
    download_model(
        "charactr/vocos-mel-24khz",
        [
            "config.yaml",
            "pytorch_model.bin"
        ],
        args.target_folder
    )
    download_voice(args.voice_folder)
