# pode_splitter/prediction_analysis/run_pipeline.py

import os
import argparse
import pandas as pd

# --- Import from the V2 module in the current directory ---
from .predict import predict_age_v2
from .plotting import (
    plot_scatter, 
    plot_bland_altman, 
    plot_error_distribution, 
    plot_bias_check
)

def plot_all_metrics(results_df: pd.DataFrame, output_dir: str, true_col: str, pred_col: str, prefix: str = ""):
    """
    Plot all performance analysis charts.
    
    Args:
        results_df: DataFrame containing the prediction results
        output_dir: Output directory
        true_col: Column name of the ground-truth age
        pred_col: Column name of the predicted age
        prefix: Filename prefix used to distinguish different prediction columns
    """
    if true_col not in results_df.columns:
        print(f"⚠️ Skipping {pred_col}: the DataFrame is missing the true-age column '{true_col}'")
        return
    if pred_col not in results_df.columns:
        print(f"⚠️ Skipping {pred_col}: the DataFrame is missing the predicted column '{pred_col}'")
        return
    
    print(f"\n--- Generating performance analysis charts for {pred_col} ---")
    
    # Call the individual plotting functions
    plot_scatter(results_df, output_dir, true_col=true_col, pred_col=pred_col, plot_name_prefix=prefix)
    plot_bland_altman(results_df, output_dir, true_col=true_col, pred_col=pred_col, plot_name_prefix=prefix)
    plot_error_distribution(results_df, output_dir, true_col=true_col, pred_col=pred_col, plot_name_prefix=prefix)
    plot_bias_check(results_df, output_dir, true_col=true_col, pred_col=pred_col, plot_name_prefix=prefix)

def main():
    """
    (V2 Version) Run the complete prediction and performance analysis pipeline.
    Supports analyzing both Final_Age and Base_Age at the same time.
    """
    parser = argparse.ArgumentParser(description="V2 - Complete age prediction and performance analysis pipeline")
    
    # --- Core arguments ---
    parser.add_argument('--data_path', type=str, required=True, help="Path to the .xlsx data file containing true ages and image paths")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained V2 model weights file (.pth)")
    parser.add_argument('--model_config_path', type=str, required=True, help="Path to the V2 model_config.pth file saved during training")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save all outputs (prediction Excel and charts)")
    
    # --- Data column-name arguments ---
    parser.add_argument('--image_col_left', type=str, required=True, help="Column name of the left-eye image path")
    parser.add_argument('--image_col_right', type=str, required=True, help="Column name of the right-eye image path")
    parser.add_argument('--age_col', type=str, required=True, help="Column name of the ground-truth age label")
    
    # --- Model and data-loader arguments ---
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224')
    parser.add_argument('--image_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size used for inference")
    parser.add_argument('--num_workers', type=int, default=4)

    # --- Switch to control whether plots are generated ---
    parser.add_argument('--no-plots', action='store_true', help="If set, only the prediction Excel file will be generated, no plots.")
    
    # --- New: control which prediction columns are analyzed ---
    parser.add_argument('--analyze-base-age', action='store_true', help="If set, additionally generate the Base_Age vs True_Age comparison charts.")
    
    # --- New: specify additional prediction columns to analyze ---
    parser.add_argument('--extra-pred-cols', type=str, default="", help="Additional prediction column names to analyze, comma separated. e.g.: 'Base_Age,Total_Age_Delta'")

    args = parser.parse_args()
    
    # --- Step 0: Create the output directory ---
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"All outputs will be saved to: {args.output_dir}")

    # --- Step 1: Load the original data ---
    print(f"Loading original data from {args.data_path}...")
    original_df = pd.read_excel(args.data_path)

    # --- Step 2: Run V2 prediction ---
    predictions_dict = predict_age_v2(original_df, args)

    # --- Step 3: Merge prediction results back into the original DataFrame ---
    print("Merging the prediction results into the original table...")
    predictions_df = pd.DataFrame(predictions_dict)
    results_df = pd.concat([original_df.reset_index(drop=True), predictions_df], axis=1)

    # --- Step 4: Save the final result containing all columns ---
    output_excel_path = os.path.join(args.output_dir, "predictions_v2.xlsx")
    results_df.to_excel(output_excel_path, index=False)
    print(f"✅ Complete table containing original data and all V2 predictions saved to: {output_excel_path}")

    # --- Step 5: Generate performance analysis charts depending on the switch ---
    if not args.no_plots:
        print("\n--- Starting to generate performance analysis charts ---")
        
        # Predicted-age column names in the V2 version
        final_age_col = 'Predicted_Age'
        base_age_col = 'Base_Age'
        
        # 1. Analyze Final_Age (default)
        if final_age_col in results_df.columns:
            plot_all_metrics(results_df, args.output_dir, args.age_col, final_age_col, prefix="final")
        else:
            print(f"⚠️ Column '{final_age_col}' not found, skipping Final Age analysis")
        
        # 2. Analyze Base_Age (if enabled)
        if args.analyze_base_age:
            if base_age_col in results_df.columns:
                plot_all_metrics(results_df, args.output_dir, args.age_col, base_age_col, prefix="base")
            else:
                print(f"⚠️ Column '{base_age_col}' not found, cannot run Base Age analysis")
        
        # 3. Analyze the other specified prediction columns
        if args.extra_pred_cols:
            extra_cols = [col.strip() for col in args.extra_pred_cols.split(',') if col.strip()]
            for pred_col in extra_cols:
                if pred_col in results_df.columns:
                    # Convert the column name into a valid prefix name
                    safe_prefix = pred_col.replace(' ', '_').replace('-', '_')
                    plot_all_metrics(results_df, args.output_dir, args.age_col, pred_col, prefix=safe_prefix)
                else:
                    print(f"⚠️ Column '{pred_col}' not found, skipping")
    else:
        print("\n--- Detected '--no-plots' flag, skipping the plotting step ---")

    print("\n--- All V2 pipeline steps completed successfully! ---")


if __name__ == '__main__':
    main()
