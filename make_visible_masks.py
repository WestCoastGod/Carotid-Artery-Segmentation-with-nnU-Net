from PIL import Image
import numpy as np
import os

# Folders
mask_folder = r"C:\Users\cxoox\Desktop\BP_Prediction\BP_Prediction\masked_images"
original_folder = r"C:\Users\cxoox\Desktop\CCA"
output_folder = r"C:\Users\cxoox\Desktop\BP_Prediction\BP_Prediction\visible_masks"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

# Get mask files
mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png') and not f.startswith(('dataset', 'plans', 'predict'))]

for mask_file in mask_files:
    # Find corresponding original image
    original_name = mask_file.replace('_0000', '')  # Remove the _0000 suffix
    original_path = os.path.join(original_folder, original_name)
    
    if os.path.exists(original_path):
        # Load original image and mask
        original = Image.open(original_path).convert('RGB')
        mask = np.array(Image.open(os.path.join(mask_folder, mask_file)))
        
        # Resize original to match mask size
        original = original.resize((mask.shape[1], mask.shape[0]))
        original_array = np.array(original)
        
        # Create green overlay where mask is 1
        overlay = original_array.copy()
        overlay[mask == 1] = [0, 255, 0]  # Green
        
        # Blend original and overlay (70% original, 30% green)
        blended = (0.7 * original_array + 0.3 * overlay).astype(np.uint8)
        
        # Save result
        result_path = os.path.join(output_folder, mask_file)
        Image.fromarray(blended).save(result_path)
        print(f"Created overlay: {mask_file}")

print(f"Created {len(mask_files)} overlay images")
print(f"Overlays saved to: {output_folder}")
