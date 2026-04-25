# preprocess_images_parallel.py

import argparse
import os
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm
import torchvision.transforms as T
import multiprocessing  # <-- Import multiprocessing library
import cv2
import numpy as np
# --- Global settings (same as before) ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


# --- 【New】 Function to auto-crop fundus ROI ---
def crop_fundus_roi(image_pil: Image.Image, threshold: int = 20) -> Image.Image:
    """
    Automatically detect and crop the minimum bounding rectangle region (ROI) of fundus image.
    """
    image_np = np.array(image_pil)
    if image_np.size == 0:
        # Handle empty image case
        raise ValueError("Input image is empty")

    gray_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray_img, threshold, 255, cv2.THRESH_BINARY)
    points = np.argwhere(mask > 0)

    if points.shape[0] < 100:
        # If foreground pixels are too few, may not be a valid fundus image
        # Return original image to avoid cropping the entire image
        tqdm.write(f"\nWarning: Too few foreground pixels detected in an image, skipping crop for this image.")
        return image_pil

    y_coords, x_coords = points[:, 0], points[:, 1]

    x_min, x_max = np.min(x_coords), np.max(x_coords)
    y_min, y_max = np.min(y_coords), np.max(y_coords)

    # Add a small boundary padding to avoid cropping too tight
    padding = 5
    x_min = max(0, x_min - padding)
    y_min = max(0, y_min - padding)
    x_max = min(image_np.shape[1], x_max + padding)
    y_max = min(image_np.shape[0], y_max + padding)

    cropped_np = image_np[y_min:y_max + 1, x_min:x_max + 1]
    cropped_pil = Image.fromarray(cropped_np)

    return cropped_pil


# --- 【Modified】 Update process_image function to integrate ROI cropping ---
def process_image(args_tuple):
    """
    Function to process a single image.
    Now includes complete workflow of ROI cropping -> resize -> center crop.
    """
    original_path, age, output_dir, image_size = args_tuple

    # Define image transforms
    # Note: The resize and center crop here are performed after ROI cropping
    preprocess_transform = T.Compose([
        T.Resize(image_size),  # Resize cropped ROI of varying size to uniform size
        T.CenterCrop(image_size)
    ])

    base_name = os.path.basename(original_path)
    new_path = os.path.join(output_dir, base_name)

    try:
        with Image.open(original_path).convert("RGB") as img:
            # --- Core modification point: Perform ROI cropping before all other transforms ---
            cropped_img = crop_fundus_roi(img)

            # Apply subsequent processing to cropped image
            processed_img = preprocess_transform(cropped_img)
            processed_img.save(new_path)

        return {'image_path': new_path, 'age': age}
    except Exception as e:
        tqdm.write(f"\nWarning: Cannot process image {original_path}. Error: {e}. This image will be skipped.")
        return None


def main():
    parser = argparse.ArgumentParser(description="[Multi-process accelerated] Batch preprocess images")
    parser.add_argument('--manifest_path', type=str, required=True, help='Path to original CSV manifest file.')
    parser.add_argument('--output_dir', type=str, required=True, help='New directory path to store preprocessed images.')
    parser.add_argument('--output_manifest', type=str, required=True, help='Filename for the new CSV manifest after preprocessing.')
    parser.add_argument('--image_size', type=int, default=256, help='Target size for preprocessing.')
    # New parameter: control number of processes
    parser.add_argument('--num_workers', type=int, default=os.cpu_count(),
                        help='Number of CPU processes to use. Defaults to all cores on the machine.')

    args = parser.parse_args()

    # --- 1. Create directory and read manifest (same as before) ---
    print(f"Creating output directory: {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Reading original manifest: {args.manifest_path}")
    df_original = pd.read_csv(args.manifest_path)

    # --- 2. Prepare all tasks to process ---
    # Pack each row of data into a tuple as task parameter
    jobs = [(row.image_path, row.age, args.output_dir, args.image_size)
            for row in df_original.itertuples()]

    # --- 3. Core modification: Create multiprocessing pool and execute tasks ---
    print(f"Starting to process {len(jobs)} images using {args.num_workers} processes...")

    new_records = []
    # Create a process pool
    with multiprocessing.Pool(processes=args.num_workers) as pool:
        # Use pool.imap_unordered to process tasks in parallel
        # imap_unordered returns results immediately as they complete, very efficient
        # We wrap it with tqdm to show overall progress bar
        pbar = tqdm(pool.imap_unordered(process_image, jobs), total=len(jobs))
        for result in pbar:
            # Collect successfully processed results
            if result is not None:
                new_records.append(result)

    # --- 4. Save new CSV manifest (same as before) ---
    print("\nAll images processed. Generating new CSV manifest...")
    if not new_records:
        print("Warning: No images were successfully processed, not generating new manifest file.")
        return

    df_new = pd.DataFrame(new_records)
    df_new.to_csv(args.output_manifest, index=False)

    print("=" * 50)
    print("Preprocessing complete!")
    print(f"Successfully processed and saved {len(df_new)} images.")
    print(f"New images stored in: {args.output_dir}")
    print(f"New manifest file saved to: {args.output_manifest}")
    print("=" * 50)


if __name__ == '__main__':
    # Set multiprocessing start method to 'fork', which is usually more stable and efficient on Linux/macOS
    # In some environments, it works fine without setting this
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass
    main()