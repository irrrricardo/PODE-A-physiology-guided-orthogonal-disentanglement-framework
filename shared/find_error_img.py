# find_corrupted_images.py

import argparse
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

# Same as in your data_utils.py, allow loading possibly truncated images
# This can maximize simulation of real loading during training
ImageFile.LOAD_TRUNCATED_IMAGES = True


def main():
    """
    A script to traverse dataset manifest and find all corrupted images that cannot be correctly read by PIL library.
    """
    parser = argparse.ArgumentParser(description="Check and count corrupted images in dataset")
    parser.add_argument('--manifest_path', type=str, required=True,
                        help='Path to CSV manifest file containing image_path column')
    parser.add_argument('--output_file', type=str, default='corrupted_files.txt',
                        help='Output filename to save list of corrupted image paths')
    args = parser.parse_args()

    print(f"--- Starting check on manifest file: {args.manifest_path} ---")

    try:
        df = pd.read_csv(args.manifest_path)
        if 'image_path' not in df.columns:
            print(f"Error: 'image_path' column not found in manifest file {args.manifest_path}.")
            return
    except FileNotFoundError:
        print(f"Error: Manifest file not found: {args.manifest_path}")
        return

    image_paths = df['image_path'].tolist()
    total_files = len(image_paths)
    corrupted_files = []

    # Use tqdm to create a progress bar for easy monitoring
    pbar = tqdm(image_paths, desc="Checking images", unit="file")

    for image_path in pbar:
        try:
            # Try to open image
            with Image.open(image_path) as img:
                # Force read image data to trigger potential errors
                # .verify() is an efficient check method that checks file header and basic structure
                img.verify()
        except Exception as e:
            # Any exceptions that cannot be opened or verified are considered corrupted
            corrupted_files.append(image_path)
            # Use tqdm.write to print info without disrupting progress bar
            tqdm.write(f"Corrupted image found: {image_path} | Error: {e}")

    # --- After loop ends, print summary report ---
    print("\n" + "=" * 50)
    print("Check complete!")
    print(f"Total images checked: {total_files}")
    print(f"Corrupted images found: {len(corrupted_files)}")

    if total_files > 0:
        corruption_rate = (len(corrupted_files) / total_files) * 100
        print(f"Data corruption rate: {corruption_rate:.2f}%")

    # --- Write corrupted file list to output file ---
    if corrupted_files:
        print(f"\nWriting paths of {len(corrupted_files)} corrupted files to: {args.output_file}")
        with open(args.output_file, 'w') as f:
            for path in corrupted_files:
                f.write(f"{path}\n")
        print("Write complete.")
    else:
        print("\n🎉 Congratulations! No corrupted images found in dataset.")

    print("=" * 50)


if __name__ == "__main__":
    # Example run command:
    # python find_corrupted_images.py --manifest_path ./your_data.csv
    main()