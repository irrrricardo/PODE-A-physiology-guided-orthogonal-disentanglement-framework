# clean_manifest.py

import argparse
import pandas as pd
import os


def main():
    """
    A script to remove records of specified corrupted files from the main CSV manifest.
    """
    parser = argparse.ArgumentParser(description="Clean corrupted file records from CSV manifest")
    parser.add_argument('--manifest_path', type=str, required=True,
                        help='Path to the original CSV manifest file')
    parser.add_argument('--corrupted_list_path', type=str, required=True,
                        help='Path to the txt file containing list of corrupted file paths (e.g., corrupted_files.txt)')
    parser.add_argument('--output_path', type=str,
                        help='Path to save the cleaned manifest. If not provided, defaults to original filename with "_clean.csv" appended')

    args = parser.parse_args()

    # Auto-generate output path if not specified
    if args.output_path is None:
        base, ext = os.path.splitext(args.manifest_path)
        args.output_path = f"{base}_clean{ext}"

    print("--- Starting manifest file cleanup ---")
    print(f"Original manifest: {args.manifest_path}")
    print(f"Files to remove: {args.corrupted_list_path}")
    print(f"Output file: {args.output_path}")

    try:
        # 1. Load original manifest DataFrame
        df_original = pd.read_csv(args.manifest_path)

        # 2. Load corrupted file path list
        with open(args.corrupted_list_path, 'r') as f:
            # Using set can deduplicate and provide faster lookup
            corrupted_paths = set(line.strip() for line in f if line.strip())

        # Check if 'image_path' column exists
        if 'image_path' not in df_original.columns:
            print(f"Error: 'image_path' column not found in original manifest file {args.manifest_path}.")
            return

        # 3. Core operation: filter out corrupted file records
        # .isin() checks if each value in 'image_path' column exists in corrupted_paths set
        # '~' operator is used for negation, i.e., keep rows that are NOT in the corrupted list
        df_clean = df_original[~df_original['image_path'].isin(corrupted_paths)]

        # 4. Save as new cleaned manifest file
        df_clean.to_csv(args.output_path, index=False)

        # 5. Print summary report
        print("\n" + "=" * 50)
        print("Cleanup complete!")
        print(f"Original record count: {len(df_original)}")
        print(f"Corrupted records removed: {len(df_original) - len(df_clean)}")
        print(f"New record count after cleanup: {len(df_clean)}")
        print(f"Generated cleaned manifest file: {args.output_path}")
        print("=" * 50)

    except FileNotFoundError as e:
        print(f"\nError: File not found {e.filename}. Please check if the path is correct.")
    except Exception as e:
        print(f"\nUnknown error occurred during processing: {e}")


if __name__ == "__main__":
    # Example run command:
    # python clean_manifest.py --manifest_path ./all_data.csv --corrupted_list_path ./corrupted_files.txt
    main()