import gdown
import os

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Google Drive file ID and output path
file_id = "???"  # Replace with actual Google Drive file ID
output_path = os.path.join(ROOT_PATH, "data", "model_outputs.pkl")

print(f"Downloading pickled model outputs to {output_path}...")

# Download file from Google Drive
gdown.download(
    f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False
)

print("Download complete!")
