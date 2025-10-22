import pandas as pd
import os
import pydicom
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import numpy as np

# --- 1. 配置路径 (Configuration) ---
# 存放 files/p10/p10... 等影像(.dcm)和报告(.txt)的根目录
FILES_DIR = 'E:/data_subset1/'

# CheXpert Labeler 项目的根目录 (需要您从GitHub克隆)
CHEXPERT_LABELER_DIR = './chexpert-labeler/' # <--- 请确认此路径

# --- 中间和最终输出文件路径 (Intermediate and final output paths) ---
PRELIMINARY_METADATA_CSV = './_temp_preliminary_metadata.csv'
REPORTS_TO_LABEL_CSV = './reports_to_label.csv'
REPORTS_WITH_IDS_TEMP_CSV = './_temp_reports_with_ids.csv' 
LABELED_REPORTS_CSV = './labeled_reports.csv' 
FINAL_DATASET_CSV = './final_dataset_with_labels.csv'
TRAIN_CSV = './train.csv'
VALIDATE_CSV = './validate.csv'
TEST_CSV = './test.csv'


# --- 2. 自建元数据：扫描文件并读取DCM头信息 ---
print("--- Step 2: Scanning files to build metadata from scratch ---")
if not os.path.exists(PRELIMINARY_METADATA_CSV):
    image_data = []
    # 使用 os.walk 遍历所有文件
    for root, dirs, files in tqdm(os.walk(FILES_DIR), desc="Scanning DICOM files"):
        for file in files:
            if file.endswith('.dcm'):
                dcm_path = os.path.join(root, file)
                try:
                    # 解析路径获取ID - 修正路径分隔符问题
                    normalized_path = os.path.normpath(dcm_path)
                    parts = normalized_path.split(os.sep)
                    # 从路径中提取包含p和s的部分
                    subject_part = parts[-3]  # 应该是 pXXXXXXX
                    study_part = parts[-2]    # 应该是 sXXXXXXX
                    
                    if not subject_part.startswith('p') or not study_part.startswith('s'):
                        raise ValueError(f"路径格式不正确: {dcm_path}")
                        
                    subject_id = int(subject_part.replace('p', ''))
                    study_id = int(study_part.replace('s', ''))
                    dicom_id = file.replace('.dcm', '')

                    # 读取DCM文件头获取视角
                    dcm_file = pydicom.dcmread(dcm_path, stop_before_pixels=True)
                    view_position = dcm_file.ViewPosition

                    image_data.append({
                        'subject_id': subject_id,
                        'study_id': study_id,
                        'dicom_id': dicom_id,
                        'ViewPosition': view_position,
                        'image_path': dcm_path # 暂时保存完整路径
                    })
                except Exception as e:
                    print(f"Could not process {dcm_path}: {e}") # 输出详细异常信息，便于调试

    df_meta = pd.DataFrame(image_data)
    df_meta.to_csv(PRELIMINARY_METADATA_CSV, index=False)
    print(f"Scanned {len(df_meta)} images and created preliminary metadata.")
else:
    print(f"Found existing preliminary metadata: {PRELIMINARY_METADATA_CSV}, loading it.")
    df_meta = pd.read_csv(PRELIMINARY_METADATA_CSV)

# 筛选正面视角
df_frontal = df_meta[df_meta['ViewPosition'].isin(['AP', 'PA'])].copy()
print(f"Found {len(df_frontal)} frontal view images.")


# --- 3. 准备CheXpert Labeler的输入文件 ---
if not os.path.exists(LABELED_REPORTS_CSV):
    print(f"\n--- Step 3: Preparing reports for labeling... ---")
    studies_to_process = df_frontal[['subject_id', 'study_id']].drop_duplicates()
    report_data = []

    for index, row in tqdm(studies_to_process.iterrows(), total=len(studies_to_process), desc="Reading .txt reports"):
        subject_id = str(int(row['subject_id']))
        study_id = str(int(row['study_id']))
        
        # 正确的路径构建：E:/data_subset1/p10000032/s50414267.txt
        report_path = os.path.join(FILES_DIR, f'p{subject_id}', f's{study_id}.txt')
        
        try:
            with open(report_path, 'r', encoding='utf-8') as f:
                report_text = f.read()
                report_data.append({'subject_id': row['subject_id'], 'study_id': row['study_id'], 'report_text': report_text})
        except FileNotFoundError:
            pass

    df_reports = pd.DataFrame(report_data)
    print(f"成功读取 {len(df_reports)} 个报告文件")
    print(f"df_reports 的列: {df_reports.columns.tolist()}")
    
    if len(df_reports) == 0:
        print("错误：没有找到任何报告文件！请检查路径和文件结构。")
        exit()
        
    df_reports.to_csv(REPORTS_WITH_IDS_TEMP_CSV, index=False)
    df_reports[['report_text']].to_csv(REPORTS_TO_LABEL_CSV, index=False, header=False)

    print(f"Successfully created {REPORTS_TO_LABEL_CSV} with {len(df_reports)} reports.")
    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!! ACTION REQUIRED !!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("This script has paused. Please perform the following manual step:")
    print(f"1. Make sure you have cloned the CheXpert labeler into: {CHEXPERT_LABELER_DIR}")
    print(f"2. Navigate to your CheXpert labeler directory: cd {CHEXPERT_LABELER_DIR}")
    print(f"3. Run the labeler on the generated file. The command should be:")
    print(f"   python label.py --reports_path ../{REPORTS_TO_LABEL_CSV} --output_path ../{LABELED_REPORTS_CSV}")
    print("4. Once labeling is complete and labeled_reports.csv is created, re-run this script.")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n")
    exit()

# --- 4. 处理标签 ---
print(f"\n--- Step 4: Processing labels from {LABELED_REPORTS_CSV}... ---")
chexpert_cols = ['Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum','Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion','Pleural Other','Pneumonia','Pneumothorax','Support Devices']
df_reports_with_ids = pd.read_csv(REPORTS_WITH_IDS_TEMP_CSV)
df_labeled_reports_raw = pd.read_csv(LABELED_REPORTS_CSV, header=None, names=chexpert_cols)
df_labeled_reports = pd.concat([df_reports_with_ids, df_labeled_reports_raw], axis=1)

print("Applying 'U-Ones' strategy for uncertain labels.")
df_processed_labels = df_labeled_reports.copy()
for col in chexpert_cols:
    df_processed_labels[col] = df_processed_labels[col].replace(-1.0, 1.0)
    df_processed_labels[col] = df_processed_labels[col].fillna(0.0)
print("Label processing complete.")

# --- 5. 合并并创建自定义数据集划分 ---
print("\n--- Step 5: Merging data and creating custom train/validate/test split ---")
df_final = pd.merge(df_frontal, df_processed_labels, on=['subject_id', 'study_id'], how='inner')

# 以患者ID为单位进行划分
all_patient_ids = df_final['subject_id'].unique()
np.random.seed(0) # for reproducibility
np.random.shuffle(all_patient_ids)

# 80% train, 10% validation, 10% test
train_ids, test_val_ids = train_test_split(all_patient_ids, test_size=0.2, random_state=0)
validate_ids, test_ids = train_test_split(test_val_ids, test_size=0.5, random_state=0)

def assign_split(subject_id):
    if subject_id in train_ids:
        return 'train'
    elif subject_id in validate_ids:
        return 'validate'
    else:
        return 'test'

df_final['split'] = df_final['subject_id'].apply(assign_split)
print("Patient-level split created:")
print(df_final['split'].value_counts())

# --- 6. 生成最终的CSV文件 ---
print("\n--- Step 6: Generating final train, validate, and test CSV files ---")
output_cols = ['image_path'] + chexpert_cols

df_train = df_final[df_final['split'] == 'train'][output_cols]
df_validate = df_final[df_final['split'] == 'validate'][output_cols]
df_test = df_final[df_final['split'] == 'test'][output_cols]

df_train.to_csv(TRAIN_CSV, index=False)
df_validate.to_csv(VALIDATE_CSV, index=False)
df_test.to_csv(TEST_CSV, index=False)

print(f"Train set: {len(df_train)} images. Saved to {TRAIN_CSV}")
print(f"Validation set: {len(df_validate)} images. Saved to {VALIDATE_CSV}")
print(f"Test set: {len(df_test)} images. Saved to {TEST_CSV}")
print("\nPreprocessing finished successfully!")

