"""
Build final dataset CSV directly from DICOM files (no PNG conversion needed)
This script merges the labeled reports with DICOM paths and creates train/val/test splits
"""

import os
import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
DATASET_DIR = r"E:\download\dataset\dataset"
OUTPUT_DIR = r"E:\download\dataset"

LABELED_REPORTS_CSV = os.path.join(OUTPUT_DIR, "labeled_reports.csv")
REPORTS_TO_LABEL_CSV = os.path.join(OUTPUT_DIR, "reports_to_label.csv")
FINAL_DATASET_CSV = os.path.join(OUTPUT_DIR, "final_dataset_with_labels.csv")
TRAIN_CSV = os.path.join(OUTPUT_DIR, "train.csv")
VALIDATE_CSV = os.path.join(OUTPUT_DIR, "validate.csv")
TEST_CSV = os.path.join(OUTPUT_DIR, "test.csv")

# CheXpert labels
CHEXPERT_LABELS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]

print("=" * 80)
print("Building CheXpert Dataset from DICOM files")
print("=" * 80)

# Step 1: Rebuild records with DICOM paths
print("\n[Step 1/3] Collecting DICOM file paths...")
print("-" * 80)

records = []
patient_dirs = [d for d in os.listdir(DATASET_DIR) if d.startswith("p")]

for patient_id in patient_dirs:
    patient_path = os.path.join(DATASET_DIR, patient_id)
    if not os.path.isdir(patient_path):
        continue

    for study_name in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study_name)

        if os.path.isdir(study_path) and study_name.startswith("s"):
            report_file = os.path.join(patient_path, f"{study_name}.txt")
            if not os.path.exists(report_file):
                continue

            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    report_text = f.read().replace("\n", " ").strip()
            except:
                continue

            dicom_files = [f for f in os.listdir(study_path) if f.endswith(".dcm")]

            for dcm_file in dicom_files:
                dcm_path = os.path.join(study_path, dcm_file)
                records.append({
                    "subject_id": patient_id,
                    "study_id": study_name,
                    "dicom_path": dcm_path,
                    "report_text": report_text,
                })

df_records = pd.DataFrame(records)
print(f"[OK] Collected {len(df_records)} DICOM file paths")

# Step 2: Merge with labels
print("\n[Step 2/3] Merging with labeled reports...")
print("-" * 80)

df_labels = pd.read_csv(LABELED_REPORTS_CSV)
print(f"[OK] Loaded {len(df_labels)} labeled reports")

# Merge
df_combined = pd.concat(
    [df_records.reset_index(drop=True), df_labels.reset_index(drop=True)],
    axis=1,
)

# Convert uncertain labels (-1) to positive (1) as per CheXpert convention
for col in CHEXPERT_LABELS:
    df_combined[col] = df_combined[col].replace(-1.0, 1.0).fillna(0.0)

# Select final columns
final_cols = ["dicom_path", "subject_id", "study_id"] + CHEXPERT_LABELS
df_final = df_combined[final_cols]

# Save complete dataset
df_final.to_csv(FINAL_DATASET_CSV, index=False)
print(f"[OK] Final dataset saved: {FINAL_DATASET_CSV}")
print(f"    Total images: {len(df_final)}")

# Step 3: Split by patient
print("\n[Step 3/3] Splitting dataset by patient...")
print("-" * 80)

subjects = df_final["subject_id"].unique()
print(f"[OK] Unique patients: {len(subjects)}")

# 80% train, 10% val, 10% test
train_subj, temp_subj = train_test_split(subjects, test_size=0.2, random_state=42)
val_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=42)

df_train = df_final[df_final["subject_id"].isin(train_subj)]
df_validate = df_final[df_final["subject_id"].isin(val_subj)]
df_test = df_final[df_final["subject_id"].isin(test_subj)]

df_train.to_csv(TRAIN_CSV, index=False)
df_validate.to_csv(VALIDATE_CSV, index=False)
df_test.to_csv(TEST_CSV, index=False)

print("\n" + "=" * 80)
print("[SUCCESS] Dataset preparation complete!")
print("=" * 80)
print(f"\nDataset split:")
print(f"  Training:   {len(df_train):5d} images ({len(train_subj):4d} patients)")
print(f"  Validation: {len(df_validate):5d} images ({len(val_subj):4d} patients)")
print(f"  Test:       {len(df_test):5d} images ({len(test_subj):4d} patients)")
print(f"  Total:      {len(df_final):5d} images ({len(subjects):4d} patients)")

print(f"\nOutput files:")
print(f"  • {FINAL_DATASET_CSV}")
print(f"  • {TRAIN_CSV}")
print(f"  • {VALIDATE_CSV}")
print(f"  • {TEST_CSV}")

print("\n[OK] Ready for model training with DICOM files!")
print("=" * 80)
