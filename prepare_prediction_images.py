import os
import shutil
from pathlib import Path
from PIL import Image

# Source and destination folders
source_folder = r"C:\path\to\your\project\prediction_images"
temp_folder = r"C:\path\to\your\project\preprocessed_prediction_images"
# Create temporary folder
os.makedirs(temp_folder, exist_ok=True)

# Process and rename images to nnUNet format
for img_file in Path(source_folder).glob("*.png"):
    # Load image and convert to grayscale
    img = Image.open(img_file).convert("L")
    
    # Create new filename with _0000 suffix
    new_name = img_file.stem + "_0000.png"
    dest_path = Path(temp_folder) / new_name
    
    # Save as grayscale PNG
    img.save(dest_path)
    print(f"Processed: {img_file.name} -> {new_name}")

print(f"\nPrepared {len(list(Path(temp_folder).glob('*.png')))} images for inference")
print(f"Temporary folder: {temp_folder}")
