
# prepare_dataset.py

import pandas as pd
import requests
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import csv
from tqdm import tqdm
from urllib.parse import urlparse

# --- Global configuration ---
# Use thread lock to ensure thread-safe CSV file writing
CSV_LOCK = Lock()


def read_input_file(file_path, no_header):
    """Read input file based on file extension and no_header flag"""
    try:
        if file_path.endswith('.csv'):
            header_arg = None if no_header else 0
            df = pd.read_csv(file_path, header=header_arg, low_memory=False)
        elif file_path.endswith(('.xlsx', '.xls')):
            header_arg = None if no_header else 0
            df = pd.read_excel(file_path, header=header_arg)
        else:
            raise ValueError("Unsupported file format, please provide .csv or .xlsx file.")

        if no_header:
            # If no header, auto-generate column names in the form A, B, C...
            df.columns = [chr(ord('A') + i) for i in range(len(df.columns))]
            print("Enabled --no_header mode, auto-generated column names A, B, C...")

        return df
    except FileNotFoundError:
        print(f"Error: Input file not found at '{file_path}'")
        return None
    except Exception as e:
        print(f"Error occurred while reading file: {e}")
        return None


def download_image_robust(session, url, save_path, timeout, retries, delay):
    """
    A robust image download function with retry logic.
    Uses session to reuse connections.
    """
    for attempt in range(retries):
        try:
            response = session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()  # Raise HTTPError if status code is not 200
            with open(save_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            return "success", None
        except requests.exceptions.RequestException as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            else:
                return "request_exception", str(e)
    return "max_retries_exceeded", None


def process_task(task_info):
    """
    Thread worker function: Process a single download task and log.
    """
    (
        session, image_id, url, save_path, age, original_id,
        log_filepath, timeout, retries, delay
    ) = task_info

    # 1. Check if file already exists
    if os.path.exists(save_path):
        status, error_msg = "already_exists", None

    # 2. Check if URL is valid
    elif not isinstance(url, str) or not url.startswith(('http://', 'https://')):
        status, error_msg = "invalid_url", None

    # 3. Execute download
    else:
        status, error_msg = download_image_robust(session, url, save_path, timeout, retries, delay)

    # 4. Prepare log entry
    log_entry = {
        'image_id': image_id,
        'original_id': original_id,
        'age': age,
        'url': url,
        'local_path': save_path if status in ['success', 'already_exists'] else '',
        'status': status,
        'error_message': error_msg or ''
    }

    # 5. Use lock to safely write log to CSV file
    with CSV_LOCK:
        with open(log_filepath, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_entry.keys())
            writer.writerow(log_entry)


def main(args):
    """Main function"""
    # --- 1. Initialize and set up environment ---
    os.makedirs(args.image_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    log_filepath = os.path.join(args.output_dir, "download_log.csv")
    manifest_filepath = os.path.join(args.output_dir, "manifest.csv")

    log_header = [
        'image_id', 'original_id', 'age', 'url', 'local_path',
        'status', 'error_message'
    ]

    # --- 2. Resume logic ---
    processed_ids = set()
    if os.path.exists(log_filepath):
        print(f"Found existing log file: {log_filepath}, will resume from checkpoint.")
        try:
            log_df_resume = pd.read_csv(log_filepath)
            # Only skip tasks that are confirmed successful
            processed_ids = set(log_df_resume[log_df_resume['status'].isin(['success', 'already_exists'])]['image_id'])
            print(f"Successfully processed {len(processed_ids)} image entries, will skip these records.")
        except (pd.errors.EmptyDataError, KeyError):
            print("Log file is empty or format incorrect, will restart.")
            # Recreate if log file has issues
            with open(log_filepath, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=log_header)
                writer.writeheader()
    else:
        # Create and write header if log file doesn't exist
        with open(log_filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=log_header)
            writer.writeheader()

    # --- 3. Read and prepare data ---
    df = read_input_file(args.input_file, args.no_header)
    if df is None:
        return

    print(f"Read {len(df)} records from '{args.input_file}'.")

    tasks_to_run = []
    required_cols = {args.id_col, args.target_col, args.url_col_1, args.url_col_2}
    if not required_cols.issubset(df.columns):
        print(f"Error: Input file is missing required columns. Required: {required_cols}, actually have: {list(df.columns)}")
        print("Hint: Use --no_header flag if your file has no header.")
        return

    # --- 4. Create task list ---
    for _, row in df.iterrows():
        original_id = row[args.id_col]
        age = row[args.target_col]

        for url_col_name in [args.url_col_1, args.url_col_2]:
            url = row[url_col_name]
            image_id = f"{original_id}_{url_col_name}"

            if image_id not in processed_ids:
                if pd.notna(url) and isinstance(url, str) and url.strip():
                    # Extract filename and extension from URL
                    try:
                        filename = os.path.basename(urlparse(url).path)
                        if not filename:  # If URL ends with /
                            filename = f"{image_id}.jpg"
                    except:
                        filename = f"{image_id}.jpg"

                    save_path = os.path.join(args.image_dir, filename)
                    # Put requests.Session in each task to ensure thread safety
                    session = requests.Session()
                    tasks_to_run.append((
                        session, image_id, url, save_path, age, original_id,
                        log_filepath, args.timeout, args.retries, args.delay
                    ))

    if not tasks_to_run:
        print("All images have been processed, no download needed.")
    else:
        print(f"Preparing to download {len(tasks_to_run)} new image entries...")
        # --- 5. Use multi-threading to execute downloads ---
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            list(tqdm(executor.map(process_task, tasks_to_run), total=len(tasks_to_run), desc="Download progress"))

    # --- 6. Generate manifest file and print summary ---
    print("\nAll download tasks completed. Generating manifest file and summarizing...")

    try:
        final_log_df = pd.read_csv(log_filepath)
        if final_log_df.empty:
            print("Warning: Log file is empty, cannot generate manifest or summary.")
            return

        successful_df = final_log_df[final_log_df['status'].isin(['success', 'already_exists'])].copy()

        # Ensure columns are correct type
        successful_df = successful_df.rename(columns={'local_path': 'image_path'})
        manifest_df = successful_df[['image_path', 'age']]
        manifest_df.to_csv(manifest_filepath, index=False)

        print("\n--- Download Summary ---")
        status_counts = final_log_df['status'].value_counts().to_dict()
        total_processed = len(final_log_df)

        print(f"Total URL entries processed: {total_processed}")
        print(f"  - Success (newly downloaded or already exists): {status_counts.get('success', 0) + status_counts.get('already_exists', 0)}")
        print(
            f"  - Download failed (request exception/timeout): {status_counts.get('request_exception', 0) + status_counts.get('max_retries_exceeded', 0)}")
        print(f"  - Invalid URL: {status_counts.get('invalid_url', 0)}")
        print("-" * 20)
        print(f"Successfully generated manifest file with {len(manifest_df)} records.")
        print(f"Manifest file saved to: {manifest_filepath}")
        print(f"Detailed download log saved to: {log_filepath}")

    except Exception as e:
        print(f"Error occurred while generating manifest or summary: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robustly download images from Excel/CSV file using multi-threading.")

    # --- Core parameters ---
    parser.add_argument('--input_file', type=str, required=True, help='Path to Excel or CSV file containing URLs and ages.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory to save downloaded images.')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save logs and manifest files.')

    # --- Column name parameters ---
    parser.add_argument('--url_col_1', type=str, required=True, help='Column name of the first image URL (e.g., J).')
    parser.add_argument('--url_col_2', type=str, required=True, help='Column name of the second image URL (e.g., K).')
    parser.add_argument('--target_col', type=str, required=True, help='Column name of the target column (e.g., D).')
    parser.add_argument('--id_col', type=str, required=True, help='Column name to use as unique identifier (e.g., A).')

    # --- Performance and robustness parameters ---
    parser.add_argument('--max_workers', type=int, default=20, help='Number of concurrent download threads.')
    parser.add_argument('--retries', type=int, default=3, help='Maximum number of retries on download failure.')
    parser.add_argument('--delay', type=int, default=5, help='Delay seconds between retries.')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout for network requests.')

    # --- Special case handling ---
    parser.add_argument('--no_header', action='store_true', help='Specify this flag if input file has no header.')

    args = parser.parse_args()
    main(args)