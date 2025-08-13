
import json
import os
from PIL import Image, ImageDraw

# 1. Load Label Studio JSON export
with open("\labeled_image.json") as f:
    data = json.load(f)

# 2. Create nnUNet folder structure  
os.makedirs("nnUNet_raw/Dataset214_Carotid/imagesTr", exist_ok=True)
os.makedirs("nnUNet_raw/Dataset214_Carotid/labelsTr", exist_ok=True)

# 3. Iterate over each annotation task
count = 0
for task in data:
    # Get original image path (update this if needed)
    img_path = task["data"]["image"].replace("/data/upload/7/", "training_images/")
    img = Image.open(img_path).convert("L")

    # Create blank mask (single channel)
    mask = Image.new("L", img.size, 0)  # background=0
    draw = ImageDraw.Draw(mask)

    # 4. Draw polygons to mask (Label Studio polygonlabels)
    has_annotation = False
    for annotation in task.get("annotations", []):
        for region in annotation.get("result", []):
            if region.get("type") == "polygonlabels":
                raw_points = region["value"]["points"]
                if raw_points:
                    # Convert from percentage (0-100) to pixel coordinates
                    points = [(int(x * img.width / 100), int(y * img.height / 100)) for x, y in raw_points]
                    draw.polygon(points, fill=1)  # Use 1 for carotid (nnUNet expects 0,1)
                    has_annotation = True

    # Skip images without annotations
    if not has_annotation:
        continue

    # 5. Save as PNG (nnUNet format)
    case_id = f"case_{str(count).zfill(4)}"
    img.save(f"nnUNet_raw/Dataset214_Carotid/imagesTr/{case_id}_0000.png")
    mask.save(f"nnUNet_raw/Dataset214_Carotid/labelsTr/{case_id}.png")
    count += 1

# 6. Create dataset.json
dataset_json = {
    "channel_names": {"0": "grayscale"},
    "labels": {"background": 0, "carotid": 1},
    "numTraining": count,
    "file_ending": ".png"
}

with open("nnUNet_raw/Dataset214_Carotid/dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=2)

print(f"Dataset ready: {count} samples")