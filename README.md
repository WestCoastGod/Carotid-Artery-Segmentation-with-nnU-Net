# Carotid Artery Segmentation with nnU-Net

This project uses nnU-Net to automatically segment carotid arteries from 2D ultrasound images.

## Overview

The pipeline converts Label Studio annotations into nnU-Net compatible format, trains a 2D U-Net model, and enables automatic segmentation of carotid arteries in new images.

## Prerequisites

- Python 3.11 (recommended for compatibility)
- Windows/Linux/macOS
- CUDA Version: 12.2 (GPU recommended for faster training)
- CPU or GPU

## Installation

### 1. Install Miniconda (Recommended for Linux/Cluster/No sudo)
Miniconda lets you create isolated Python environments without admin rights.

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# Follow prompts and restart your shell

# Add conda to PATH if conda command does not found
echo 'export PATH="/home/user453/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Create and Activate Environment
```bash
conda create -n nnunet python=3.11
conda activate nnunet
```

### 3. Install PyTorch
First install PyTorch with appropriate device support:

```bash
# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# For GPU (CUDA 12.2)
pip install torch torchvision
```

### 4. Install Compatible NumPy and nnU-Net
```bash
pip install "numpy<2"
pip install nnunetv2
```

### 5. Set Environment Variables
nnU-Net requires three environment variables. Set them for your session:

**Windows (PowerShell):**
```powershell
$env:nnUNet_raw="C:\path\to\your\project\nnUNet_raw"
$env:nnUNet_preprocessed="C:\path\to\your\project\nnUNet_preprocessed" 
$env:nnUNet_results="C:\path\to\your\project\nnUNet_results"
```

**Linux/macOS:**
```bash
export nnUNet_raw="/path/to/your/project/nnUNet_raw"
export nnUNet_preprocessed="/path/to/your/project/nnUNet_preprocessed"
export nnUNet_results="/path/to/your/project/nnUNet_results"
```

## Data Preprocessing

### Why Preprocessing is Required

1. **Format Conversion**: Label Studio exports annotations as JSON with percentage coordinates (0-100), but nnU-Net expects pixel coordinates and specific file naming
2. **Data Structure**: nnU-Net requires a strict folder structure with specific naming conventions
3. **Label Format**: Segmentation masks must use integer values (0=background, 1=carotid artery)
4. **Consistency**: All images must be in the same format (grayscale PNG) with consistent dimensions

### Input Data Format

- **Images**: 2D PNG files (662x464 pixels in this project)
- **Annotations**: Label Studio JSON export with polygon annotations
- **Coordinate System**: Label Studio uses percentage coordinates (0-100 scale)

### Preprocessing Steps

The `data_cleaning.py` script performs:

1. **Load Annotations**: Parse Label Studio JSON export
2. **Convert Coordinates**: Transform percentage coordinates to pixel coordinates
3. **Generate Masks**: Create binary segmentation masks from polygon annotations
4. **File Naming**: Rename files to nnU-Net convention (`case_XXXX_0000.png` for images, `case_XXXX.png` for masks)
5. **Create Metadata**: Generate `dataset.json` with channel and label information

## Project Structure

```
BP_Prediction/
├── README.md                          # This file
├── data_cleaning.py                   # Preprocessing script
├── labeled_image.json                 # Label Studio export
├── images/                            # Original ultrasound images
│   ├── image1.png
│   └── image2.png
├── nnUNet_raw/                        # Raw data for nnU-Net
│   └── Dataset214_Carotid/
│       ├── dataset.json               # Dataset metadata
│       ├── imagesTr/                  # Training images
│       │   ├── case_0000_0000.png
│       │   └── case_0001_0000.png
│       └── labelsTr/                  # Training masks
│           ├── case_0000.png
│           └── case_0001.png
├── nnUNet_preprocessed/               # Preprocessed data (auto-generated)
│   └── Dataset214_Carotid/
│       ├── nnUNetPlans.json
│       └── [preprocessed files]
└── nnUNet_results/                    # Training results (auto-generated)
    └── Dataset214_Carotid/
        └── [model checkpoints and logs]
```

## Usage Instructions

### Step 1: Prepare Your Data

1. Export annotations from Label Studio in JSON format
2. Place original images in `images/` folder
3. Update the image path in `data_cleaning.py` if needed:
   ```python
   img_path = task["data"]["image"].replace("/data/upload/7/", "images/")
   ```

### Step 2: Run Preprocessing

```bash
python data_cleaning.py
```

Expected output:
```
Dataset ready: [number of] samples
```

### Step 3: nnU-Net Preprocessing

```bash
nnUNetv2_plan_and_preprocess -d 214 --verify_dataset_integrity
```

This will:
- Extract dataset fingerprint
- Create training plans
- Preprocess all images
- Verify dataset integrity

### Step 4: Start Training

**For single GPU:**
```bash
nnUNetv2_train Dataset214_Carotid 2d 0 --npz
```

**For multiple GPUs (recommended - train different folds simultaneously):**
```bash
# Train different folds on separate GPUs (using available GPUs)
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset214_Carotid 2d 0 --npz &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train Dataset214_Carotid 2d 1 --npz &
wait

# If you can only use specific GPUs (e.g., GPU 2 and 3):
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train Dataset214_Carotid 2d 0 --npz &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train Dataset214_Carotid 2d 1 --npz &
wait
```

**For CPU:**
```bash
nnUNetv2_train Dataset214_Carotid 2d 0 --npz -device cpu
```

Training parameters:
- `Dataset214_Carotid`: Dataset name
- `2d`: Configuration (2D U-Net)
- `0`: Fold number (0-4 for 5-fold cross-validation)
- `--npz`: Save probability maps for ensemble

**Note:** When training multiple folds simultaneously:
- Each fold runs independently with its own epoch counter
- Training times may vary between folds due to different data complexity
- This approach is faster than DDP (Distributed Data Parallel) training
- Wait for both processes to complete before proceeding

### Step 5: Train Additional Folds (Optional)

For best results, train all 5 folds. You can run them sequentially or in parallel:

**Sequential (one at a time):**
```bash
nnUNetv2_train Dataset214_Carotid 2d 1 --npz
nnUNetv2_train Dataset214_Carotid 2d 2 --npz
nnUNetv2_train Dataset214_Carotid 2d 3 --npz
nnUNetv2_train Dataset214_Carotid 2d 4 --npz
```

**Parallel (if you have multiple GPUs):**
```bash
# Example with 4 GPUs
CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset214_Carotid 2d 0 --npz &
CUDA_VISIBLE_DEVICES=1 nnUNetv2_train Dataset214_Carotid 2d 1 --npz &
CUDA_VISIBLE_DEVICES=2 nnUNetv2_train Dataset214_Carotid 2d 2 --npz &
CUDA_VISIBLE_DEVICES=3 nnUNetv2_train Dataset214_Carotid 2d 3 --npz &
wait
# Then train fold 4
nnUNetv2_train Dataset214_Carotid 2d 4 --npz
```

### Step 6: Find Best Configuration

After training multiple folds:
```bash
nnUNetv2_find_best_configuration Dataset214_Carotid -c 2d
```

### Steps 7: Set epoch number (by default 1000)
Take reference to [nnUNetTrainer_Xepochs](https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/training/nnUNetTrainer/variants/training_length/nnUNetTrainer_Xepochs.py)
```bash
# 5 epochs
nnUNetv2_train Dataset214_Carotid 2d 1 --npz -tr nnUNetTrainer_5epochs

# 500 epochs
nnUNetv2_train Dataset214_Carotid 2d 1 --npz -tr nnUNetTrainer_500epochs
```

## Model Architecture

nnU-Net automatically configured a **2D U-Net with 7 stages**:

- **Input**: Grayscale images (464×232 after preprocessing)
- **Architecture**: U-shaped network with encoder-decoder structure
- **Features**: [32, 64, 128, 256, 512, 512, 512] features per stage
- **Patch Size**: 512×256 pixels
- **Batch Size**: 9 images
- **Normalization**: Z-score normalization using foreground mask

## Inference on New Images

After training, use the model for automatic segmentation:

### Step 1: Prepare Input Images
```bash
python prepare_prediction_images.py
```
This script:
- Converts images to grayscale
- Adds `_0000` suffix required by nnU-Net
- Saves formatted images to `temp_input/` folder

### Step 2: Run Prediction
```bash
# Set environment variables and run prediction
$env:nnUNet_raw="C:\path\to\your\project\nnUNet_raw"
$env:nnUNet_preprocessed="C:\path\to\your\project\nnUNet_preprocessed" 
$env:nnUNet_results="C:\path\to\your\project\nnUNet_results"
nnUNetv2_predict -i "preprocessed_prediction_images" -o "masked_images" -d Dataset214_Carotid -c 2d -f 0
```

### Step 3: Create Overlay Visualizations
```bash
python make_visible_masks.py
```
This creates images with transparent green overlays showing predicted carotid arteries in the `visible_masks/` folder.

## Complete Inference Workflow

For new ultrasound images:

1. **Place images** in `C:\path\to\your\project\prediction_images` (or update path in `prepare_prediction_images.py`)
2. **Run preparation**: `python prepare_inference.py`
3. **Run prediction**: Set environment variables and run nnUNetv2_predict command
4. **Create overlays**: `python make_visible_masks.py`
5. **View results** in `visible_masked_segmented_images/` folder

## Dataset Information

- **Total Samples**: 214 annotated ultrasound images
- **Image Size**: 662×464 pixels (resized to 464×232 for training)
- **Classes**: 2 (background=0, carotid artery=1)
- **Format**: Grayscale PNG
- **Split**: 80% training (171 cases), 20% validation (43 cases)

## File Descriptions

- **`data_cleaning.py`**: Main preprocessing script that converts Label Studio annotations to nnU-Net format
- **`prepare_prediction_images.py`**: Converts new images to nnU-Net format for prediction
- **`make_visible_masks.py`**: Creates overlay images with transparent green masks on original images
- **`labeled_image.json`**: Label Studio export containing polygon annotations
- **`dataset.json`**: nnU-Net metadata file with channel names, labels, and file format
- **`nnUNetPlans.json`**: Auto-generated training configuration

## Troubleshooting

### Multi-GPU Training Issues

1. **PyTorch/NumPy Compatibility Errors**
   ```
   Error: cannot import name 'GradScaler' from 'torch'
   A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
   ```
   Solution: Install compatible versions
   ```bash
   pip install "numpy<2"
   pip uninstall torchvision torch
   pip install torch torchvision
   ```

2. **Triton/torch.compile Errors in Multi-GPU Training**
   ```
   Error: AttributeError: 'constexpr_type' object has no attribute 'is_block'
   ```
   Solution: Use separate GPU training instead of DDP:
   ```bash
   # Instead of: nnUNetv2_train Dataset214_Carotid 2d 0 --npz -num_gpus 2
   # Use:
   CUDA_VISIBLE_DEVICES=0 nnUNetv2_train Dataset214_Carotid 2d 0 --npz &
   CUDA_VISIBLE_DEVICES=1 nnUNetv2_train Dataset214_Carotid 2d 1 --npz &
   wait
   ```

3. **GPU Selection Issues**
   ```
   Error: CUDA out of memory
   ```
   Solution: Use specific GPUs if some are occupied:
   ```bash
   nvidia-smi  # Check GPU usage
   CUDA_VISIBLE_DEVICES=2,3 nnUNetv2_train Dataset214_Carotid 2d 0 --npz
   ```

### Common Issues

1. **Environment Variables Not Set**
   ```
   Error: nnUNet_raw is not defined
   ```
   Solution: Set the three environment variables as shown above

2. **CUDA Not Available**
   ```
   Error: Torch not compiled with CUDA enabled
   ```
   Solution: Add `-device cpu` flag to training command

3. **Invalid Label Values**
   ```
   Error: Expected [0, 1] Found [0, 255]
   ```
   Solution: Ensure `fill=1` in polygon drawing (not 255)

4. **Missing Images**
   ```
   Error: Cannot open image file
   ```
   Solution: Update image path replacement in `data_cleaning.py`

### Performance Notes

- **CPU Training**: Very slow but functional (hours/days per fold)
- **Single GPU Training**: Much faster (minutes/hours per fold)
- **Multi-GPU Training**: 
  - **Separate folds on separate GPUs**: Fastest approach, trains multiple folds simultaneously
  - **DDP (Distributed Data Parallel)**: Not recommended for nnU-Net due to small model size and communication overhead
- **Memory**: Requires sufficient RAM for batch processing
- **Storage**: Preprocessed data takes additional disk space

### Why Separate GPU Training is Faster than DDP

For nnU-Net's relatively small models:
- **Communication overhead** in DDP outweighs computational benefits
- **Independent training** of different folds avoids GPU synchronization
- **Better resource utilization** as each GPU processes full batches
- **Fewer compatibility issues** with PyTorch versions and CUDA

## Citation

When using nnU-Net, please cite:

```
Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). 
nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. 
Nature methods, 18(2), 203-211.
```

## Contact

For questions about this specific implementation, refer to the nnU-Net documentation: https://github.com/MIC-DKFZ/nnUNet
