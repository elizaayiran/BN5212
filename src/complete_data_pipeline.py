"""
Complete Data Processing Pipeline for CheXpert Dataset
======================================================
This script processes the full CheXpert dataset:
1. Extract DICOM paths and reports
2. Label reports using improved CheXpert labeler
3. Convert DICOM to PNG
4. Build final dataset CSV
5. Split into train/val/test sets

Author: BN5212 Group 15
"""

import os
import sys
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import pydicom
from PIL import Image
import numpy as np

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the improved CheXpert labeler
from improved_chexpert_labeler import ImprovedCheXpertLabeler

# -----------------------------
# Configuration
# -----------------------------
DATASET_DIR = r"E:\download\dataset\dataset"  # Original dataset directory
OUTPUT_IMAGE_DIR = r"E:\download\dataset\images"  # PNG output directory
OUTPUT_DIR = r"E:\download\dataset"  # Output CSV directory

# CSV file paths
REPORTS_TO_LABEL_CSV = os.path.join(OUTPUT_DIR, "reports_to_label.csv")
LABELED_REPORTS_CSV = os.path.join(OUTPUT_DIR, "labeled_reports.csv")
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

# Create output directories
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 80)
print("CheXpert Complete Data Processing Pipeline")
print("=" * 80)

# -----------------------------
# Step 1: Collect DICOM paths and reports
# -----------------------------
print("\n[Step 1/5] Collecting DICOM files and reports...")
print("-" * 80)

records = []
patient_dirs = [d for d in os.listdir(DATASET_DIR) if d.startswith("p")]

for patient_id in tqdm(patient_dirs, desc="Processing patients"):
    patient_path = os.path.join(DATASET_DIR, patient_id)
    if not os.path.isdir(patient_path):
        continue

    # Iterate through each study for this patient
    for study_name in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study_name)

        # If it's a directory and starts with 's', it's a study
        if os.path.isdir(study_path) and study_name.startswith("s"):
            # Find corresponding report file
            report_file = os.path.join(patient_path, f"{study_name}.txt")
            if not os.path.exists(report_file):
                print(f"[WARN] Report not found: {report_file}")
                continue

            # Read report text
            try:
                with open(report_file, "r", encoding="utf-8") as f:
                    report_text = f.read().replace("\n", " ").strip()
            except Exception as e:
                print(f"[WARN] Error reading {report_file}: {e}")
                continue

            # Find all DICOM files in this study
            dicom_files = [f for f in os.listdir(study_path) if f.endswith(".dcm")]
            if len(dicom_files) == 0:
                print(f"[WARN] No DICOM files in {study_path}")
                continue

            # Add each DICOM file as a record
            for dcm_file in dicom_files:
                dcm_path = os.path.join(study_path, dcm_file)
                records.append({
                    "subject_id": patient_id,
                    "study_id": study_name,
                    "dicom_path": dcm_path,
                    "report_text": report_text,
                })

df_records = pd.DataFrame(records)

if len(df_records) == 0:
    raise ValueError("[ERROR] No records found. Please check your dataset folder structure.")

print(f"\n[OK] Collected {len(df_records)} DICOM files from {len(patient_dirs)} patients")
print(f"[OK] Unique studies: {df_records['study_id'].nunique()}")
print(f"\nSample records:")
print(df_records.head())

# Save reports for labeling
df_records[["report_text"]].to_csv(REPORTS_TO_LABEL_CSV, index=False)
print(f"\n[OK] Saved reports to: {REPORTS_TO_LABEL_CSV}")

# -----------------------------
# Step 2: Label reports using improved CheXpert labeler
# -----------------------------
print("\n[Step 2/5] Labeling reports with CheXpert labeler...")
print("-" * 80)

try:
    # Initialize the labeler
    labeler = ImprovedCheXpertLabeler()

    # Read reports
    df_to_label = pd.read_csv(REPORTS_TO_LABEL_CSV)

    print(f"Processing {len(df_to_label)} reports...")

    # Label all reports
    results = []
    for idx, row in tqdm(df_to_label.iterrows(), total=len(df_to_label), desc="Labeling reports"):
        report_text = row["report_text"]
        labels_dict = labeler.label_report(report_text)
        # Convert dict to list in correct order
        labels_list = [float(labels_dict.get(label, 0) or 0) for label in CHEXPERT_LABELS]
        results.append(labels_list)

    # Create DataFrame with labels
    df_labels = pd.DataFrame(results, columns=CHEXPERT_LABELS)

    # Save labeled reports
    df_labels.to_csv(LABELED_REPORTS_CSV, index=False)
    print(f"\n[OK] Labeled reports saved to: {LABELED_REPORTS_CSV}")

    # Show label statistics
    print("\nLabel distribution:")
    for col in CHEXPERT_LABELS:
        positive = (df_labels[col] == 1.0).sum()
        negative = (df_labels[col] == 0.0).sum()
        uncertain = (df_labels[col] == -1.0).sum()
        print(f"  {col:30s}: Positive={positive:4d}, Negative={negative:4d}, Uncertain={uncertain:4d}")

except Exception as e:
    print(f"[ERROR] Error during labeling: {e}")
    raise

# -----------------------------
# Step 3: Process labels and merge with records
# -----------------------------
print("\n[Step 3/5] Merging labels with DICOM records...")
print("-" * 80)

# Read labeled reports
df_labels_raw = pd.read_csv(LABELED_REPORTS_CSV)

# Merge with original records
df_combined = pd.concat(
    [df_records.reset_index(drop=True), df_labels_raw.reset_index(drop=True)],
    axis=1,
)

# Convert uncertain labels (-1) to positive (1) as per CheXpert convention
# Convert NaN to 0 (negative/not mentioned)
for col in CHEXPERT_LABELS:
    df_combined[col] = df_combined[col].replace(-1.0, 1.0).fillna(0.0)

print(f"[OK] Merged {len(df_combined)} records with labels")

# -----------------------------
# Step 4: Convert DICOM to PNG
# -----------------------------
print("\n[Step 4/5] Converting DICOM files to PNG...")
print("-" * 80)

def dicom_to_png(dcm_path, subject_id, study_id):
    """Convert DICOM file to PNG and save in organized directory structure."""
    try:
        # Read DICOM file
        ds = pydicom.dcmread(dcm_path)
        arr = ds.pixel_array.astype(float)

        # Normalize to 0-255
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255.0
        arr = arr.astype(np.uint8)

        # Create save directory: images/patient_id/study_id/
        save_dir = os.path.join(OUTPUT_IMAGE_DIR, subject_id, study_id)
        os.makedirs(save_dir, exist_ok=True)

        # Generate PNG filename
        base_name = os.path.basename(dcm_path).replace(".dcm", ".png")
        save_path = os.path.join(save_dir, base_name)

        # Save as PNG
        Image.fromarray(arr).save(save_path)

        return save_path
    except Exception as e:
        print(f"[WARN] Error converting {dcm_path}: {e}")
        return None

# Convert all DICOM files
image_paths = []
for dcm_path, subject_id, study_id in tqdm(
    zip(df_combined["dicom_path"], df_combined["subject_id"], df_combined["study_id"]),
    total=len(df_combined),
    desc="Converting DICOM→PNG"
):
    png_path = dicom_to_png(dcm_path, subject_id, study_id)
    image_paths.append(png_path)

df_combined["image_path"] = image_paths

# Remove records where conversion failed
df_combined = df_combined[df_combined["image_path"].notna()]

print(f"\n[OK] Successfully converted {len(df_combined)} DICOM files to PNG")

# -----------------------------
# Step 5: Build final dataset and split
# -----------------------------
print("\n[Step 5/5] Building final dataset and splitting...")
print("-" * 80)

# Select final columns
final_cols = ["image_path", "subject_id", "study_id"] + CHEXPERT_LABELS
df_final = df_combined[final_cols]

# Save complete dataset
df_final.to_csv(FINAL_DATASET_CSV, index=False)
print(f"[OK] Final dataset saved: {FINAL_DATASET_CSV}")
print(f"  Total images: {len(df_final)}")

# Split by patient (not by image) to avoid data leakage
subjects = df_final["subject_id"].unique()
print(f"  Unique patients: {len(subjects)}")

# 80% train, 10% val, 10% test
train_subj, temp_subj = train_test_split(subjects, test_size=0.2, random_state=42)
val_subj, test_subj = train_test_split(temp_subj, test_size=0.5, random_state=42)

# Split dataframes
df_train = df_final[df_final["subject_id"].isin(train_subj)]
df_validate = df_final[df_final["subject_id"].isin(val_subj)]
df_test = df_final[df_final["subject_id"].isin(test_subj)]

# Save splits
df_train.to_csv(TRAIN_CSV, index=False)
df_validate.to_csv(VALIDATE_CSV, index=False)
df_test.to_csv(TEST_CSV, index=False)

print("\n" + "=" * 80)
print("[SUCCESS] Data processing complete!")
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
print(f"  • PNG images in: {OUTPUT_IMAGE_DIR}")

print("\n[OK] Ready for model training!")
print("=" * 80)
