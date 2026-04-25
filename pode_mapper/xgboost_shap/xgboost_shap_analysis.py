# pode_mapper/xgboost_shap/xgboost_shap_analysis.py
# XGBoost + SHAP phenotypic stratification analysis (Supp. Fig. 2)
#
# Usage (from the repository root):
#   python PODE/pode_mapper/xgboost_shap/xgboost_shap_analysis.py \
#       --data data/full_age_02.xlsx \
#       --output_dir outputs/xgboost_shap

import argparse
import os

import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# ---------------------------------------------------------------------------
# Default feature list (matches the physiological grouping in PODE-Splitter)
# ---------------------------------------------------------------------------
DEFAULT_FEATURE_COLUMNS = [
    'BMI', 'SBP', 'DBP', 'Creatinine', 'WBC', 'PDW',
    'HDL-C', 'FBG', 'Lymphocyte_Count', 'Neutrophil_Count', 'LDL-C',
    'MCH', 'Eosinophil_Count', 'PCT', 'MPV', 'PLT',
    'HCT', 'Monocyte_Count', 'BUN', 'Basophil_Count', 'MCV', 'TG', 'TC',
    'Urine_pH'
]

DEFAULT_XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'n_estimators': 500,
    'max_depth': 4,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'seed': 42
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="XGBoost + SHAP phenotypic stratification analysis (Supp. Fig. 2)"
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help='Path to the master clinical table (.xlsx). Must contain delta_age column.'
    )
    parser.add_argument(
        '--output_dir', type=str, default='./outputs/xgboost_shap',
        help='Directory to save SHAP summary plots and feature importance charts.'
    )
    parser.add_argument(
        '--target_col', type=str, default='delta_age',
        help='Column used for octile stratification (default: delta_age).'
    )
    parser.add_argument(
        '--test_size', type=float, default=0.3,
        help='Fraction of data held out for evaluation (default: 0.3).'
    )
    parser.add_argument(
        '--n_estimators', type=int, default=500,
        help='Number of XGBoost boosting rounds (default: 500).'
    )
    parser.add_argument(
        '--max_display', type=int, default=20,
        help='Maximum number of features to display in SHAP plots (default: 20).'
    )
    parser.add_argument(
        '--shap_dpi', type=int, default=300,
        help='DPI for saved SHAP figures (default: 300).'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── 1. Load data ─────────────────────────────────────────────────────────
    print(f"\n[Step 1/4] Loading data from '{args.data}'...")
    try:
        df = pd.read_excel(args.data)
    except FileNotFoundError:
        print(f"Error: file not found → {args.data}")
        return

    df_cleaned = df.dropna(subset=[args.target_col]).copy()
    print(f"Loaded {len(df)} rows; {len(df_cleaned)} rows remain after dropping NaN in '{args.target_col}'.")

    # ── 2. Build octile groups ────────────────────────────────────────────────
    print(f"[Step 2/4] Defining analysis groups based on '{args.target_col}' (octile split)...")
    try:
        df_cleaned['Group_Octile'] = pd.qcut(
            df_cleaned[args.target_col], q=8,
            labels=[f'Q{i}' for i in range(1, 9)]
        )
    except Exception as e:
        print(f"Error: failed to bin '{args.target_col}': {e}")
        return

    group_mapping = {
        'Q1': 'Decelerated (Q1-Q2)', 'Q2': 'Decelerated (Q1-Q2)',
        'Q3': 'Other', 'Q6': 'Other',
        'Q4': 'Normative (Q4-Q5)',  'Q5': 'Normative (Q4-Q5)',
        'Q7': 'Accelerated (Q7-Q8)', 'Q8': 'Accelerated (Q7-Q8)',
    }
    df_cleaned['Analysis_Group'] = df_cleaned['Group_Octile'].map(group_mapping)
    print("Group distribution:")
    print(df_cleaned['Analysis_Group'].value_counts())

    # Keep only feature columns that actually exist in the dataframe
    feature_cols = [c for c in DEFAULT_FEATURE_COLUMNS if c in df_cleaned.columns]
    missing = set(DEFAULT_FEATURE_COLUMNS) - set(feature_cols)
    if missing:
        print(f"⚠️  Columns not found in data (will be skipped): {sorted(missing)}")

    X = df_cleaned[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(0)
    print(f"Feature matrix: {X.shape}  |  dtypes: {X.dtypes.value_counts().to_dict()}")

    # ── 3. Train XGBoost + SHAP for each group ────────────────────────────────
    groups_to_analyze = ['Decelerated (Q1-Q2)', 'Normative (Q4-Q5)', 'Accelerated (Q7-Q8)']
    results_summary = []

    print("\n[Step 3/4] Training XGBoost + SHAP for each group...")
    for group_name in groups_to_analyze:
        print(f"\n--- Analysing: {group_name} ---")
        y = (df_cleaned['Analysis_Group'] == group_name).astype(int)
        pos_rate = y.mean()
        print(f"  Positive rate: {pos_rate:.3f} ({y.sum()} / {len(y)})")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=42, stratify=y
        )

        xgb_params = {**DEFAULT_XGB_PARAMS, 'n_estimators': args.n_estimators}
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(X_train, y_train)

        y_pred  = model.predict(X_test)
        y_prob  = model.predict_proba(X_test)[:, 1]
        acc     = accuracy_score(y_test, y_pred)
        auc     = roc_auc_score(y_test, y_prob)
        print(f"  Accuracy: {acc:.2%}  |  AUC-ROC: {auc:.4f}")

        results_summary.append({
            'Group': group_name, 'N_positive': int(y.sum()),
            'Accuracy': round(acc, 4), 'AUC_ROC': round(auc, 4)
        })

        # SHAP
        print(f"  Computing SHAP values...")
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        plt.figure(figsize=(10, 8))
        shap.summary_plot(
            shap_values, X_test,
            show=False, max_display=args.max_display,
            plot_type='dot'
        )
        fig = plt.gcf()
        fig.suptitle(f"SHAP — {group_name}", fontsize=13, fontweight='bold')
        plt.tight_layout(rect=(0, 0.03, 1, 0.95))

        safe_name    = group_name.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '')
        shap_path    = os.path.join(args.output_dir, f'shap_summary_{safe_name}.png')
        plt.savefig(shap_path, dpi=args.shap_dpi, bbox_inches='tight')
        plt.close()
        print(f"  SHAP summary saved → {shap_path}")

        # Feature importance bar chart
        importance = (
            pd.DataFrame({'Feature': feature_cols, 'Importance': model.feature_importances_})
            .sort_values('Importance', ascending=False)
        )
        plt.figure(figsize=(10, 8))
        sns.barplot(
            data=importance.head(args.max_display),
            x='Importance', y='Feature',
            palette='viridis', hue='Feature', legend=False
        )
        plt.title(f"Feature Importance — {group_name}", fontsize=13, fontweight='bold')
        plt.xlabel("XGBoost Gain Importance", fontsize=11)
        plt.ylabel("Feature", fontsize=11)
        plt.tight_layout()
        imp_path = os.path.join(args.output_dir, f'feature_importance_{safe_name}.png')
        plt.savefig(imp_path, dpi=args.shap_dpi, bbox_inches='tight')
        plt.close()
        print(f"  Feature importance saved → {imp_path}")

    # ── 4. Save summary table ─────────────────────────────────────────────────
    print("\n[Step 4/4] Saving performance summary...")
    summary_df = pd.DataFrame(results_summary)
    summary_path = os.path.join(args.output_dir, 'xgboost_group_performance.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary table saved → {summary_path}")
    print(summary_df.to_string(index=False))
    print("\n✅ XGBoost + SHAP analysis complete!")


if __name__ == '__main__':
    main()
