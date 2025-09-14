import pandas as pd
import os
from tqdm import tqdm

# --- 1. 配置路径 (Configuration) ---
# 请根据您的文件存放位置修改以下路径
# Please modify the following paths according to your file locations

# MIMIC-CXR-JPG 数据集的根目录
MIMIC_JPG_DIR = './mimic-cxr-jpg/2.0.0/'
# MIMIC-CXR 报告文本数据集的根目录
MIMIC_REPORTS_DIR = './mimic-cxr/2.0.0/'
# CheXpert Labeler 项目的根目录 (需要先从GitHub克隆)
# We assume you have cloned the chexpert-labeler repository
# https://github.com/stanfordmlgroup/chexpert-labeler
CHEXPERT_LABELER_DIR = './chexpert-labeler/'

# 定义元数据和分割文件的路径
METADATA_FILE = os.path.join(MIMIC_JPG_DIR, 'mimic-cxr-2.0.0-metadata.csv')
SPLIT_FILE = os.path.join(MIMIC_JPG_DIR, 'mimic-cxr-2.0.0-split.csv')

# 定义中间和最终输出文件的路径
REPORTS_TO_LABEL_CSV = './reports_to_label.csv'
LABELED_REPORTS_CSV = './labeled_reports.csv' # 这是CheXpert-labeler的输出文件
FINAL_DATASET_CSV = './final_dataset_with_labels.csv'
TRAIN_CSV = './train.csv'
VALIDATE_CSV = './validate.csv'
TEST_CSV = './test.csv'

# --- 2. 数据获取与筛选 (Data Acquisition & Filtering) ---
print("--- Step 2: Loading metadata and filtering for frontal views ---")

# 加载元数据
df_meta = pd.read_csv(METADATA_FILE)

# 筛选正面视图 (AP/PA)
df_frontal = df_meta[df_meta['ViewPosition'].isin(['AP', 'PA'])].copy()

print(f"Found {len(df_frontal)} frontal view images.")


# --- 3. 准备CheXpert Labeler的输入文件 (Prepare Input for CheXpert Labeler) ---
# 这一步会为每个 study_id 提取其对应的放射学报告文本
print(f"\n--- Step 3: Preparing reports for labeling, saving to {REPORTS_TO_LABEL_CSV} ---")

# 获取需要提取报告的唯一 study_id 和 subject_id
studies_to_process = df_frontal[['subject_id', 'study_id']].drop_duplicates()

report_data = []
for index, row in tqdm(studies_to_process.iterrows(), total=len(studies_to_process), desc="Reading reports"):
    subject_id = str(row['subject_id'])
    study_id = str(row['study_id'])
    
    # 构建报告文件的路径
    # e.g., files/p10/p10000032/s50414267.txt
    p_group = 'p' + subject_id[:2]
    report_path = os.path.join(MIMIC_REPORTS_DIR, 'files', p_group, f'p{subject_id}', f's{study_id}.txt')
    
    try:
        with open(report_path, 'r') as f:
            report_text = f.read()
            report_data.append({'subject_id': row['subject_id'], 'study_id': row['study_id'], 'report_text': report_text})
    except FileNotFoundError:
        # print(f"Warning: Report file not found for study_id {study_id}, skipping.")
        pass

df_reports = pd.DataFrame(report_data)

# CheXpert Labeler 需要一个只包含报告文本的CSV
df_reports[['report_text']].to_csv(REPORTS_TO_LABEL_CSV, index=False)

print(f"Successfully created {REPORTS_TO_LABEL_CSV} with {len(df_reports)} reports.")
print("!!! ACTION REQUIRED !!!")
print(f"1. Navigate to your CheXpert labeler directory: cd {CHEXPERT_LABELER_DIR}")
print(f"2. Run the labeler on the generated file. The command will be something like:")
print(f"   python label.py --reports_path ../{REPORTS_TO_LABEL_CSV} --output_path ../{LABELED_REPORTS_CSV}")
print("3. Once the labeling is complete, re-run this script, or continue from the next step if running interactively.")


# --- 4. 处理不确定性标签 (Handle Uncertainty) ---
# 只有当CheXpert Labeler的输出文件存在时，才执行后续步骤
if not os.path.exists(LABELED_REPORTS_CSV):
    print(f"\n--- CheXpert labeler output '{LABELED_REPORTS_CSV}' not found. Exiting. ---")
    print("Please run the CheXpert labeler as instructed above and then run this script again.")
else:
    print(f"\n--- Step 4: Found {LABELED_REPORTS_CSV}. Processing uncertainty in labels. ---")
    # CheXpert labeler的输出没有列名，我们需要手动添加
    # The columns are based on the standard CheXpert 14 labels
    chexpert_cols = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum',
        'Fracture', 'Lung Lesion', 'Lung Opacity', 'No Finding', 'Pleural Effusion',
        'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices'
    ]
    
    df_labeled_reports_raw = pd.read_csv(LABELED_REPORTS_CSV)
    df_labeled_reports_raw.columns = chexpert_cols
    
    # 将原始的报告信息和标签信息合并
    # The labeler maintains the original order, so we can concatenate horizontally
    df_labeled_reports = pd.concat([df_reports.reset_index(drop=True), df_labeled_reports_raw.reset_index(drop=True)], axis=1)

    # 推荐策略：U-Ones (将-1.0映射为1.0)
    # Recommended Strategy: U-Ones (map uncertain -1.0 to positive 1.0)
    print("Applying 'U-Ones' strategy for uncertain labels.")
    df_processed_labels = df_labeled_reports.copy()
    for col in chexpert_cols:
        df_processed_labels[col] = df_processed_labels[col].replace(-1.0, 1.0)
        # 将空值 (NaN) 视为 Negative (0.0)
        df_processed_labels[col] = df_processed_labels[col].fillna(0.0)

    print("Label processing complete.")

    # --- 5. 构建最终数据集 (Construct Final Dataset) ---
    print("\n--- Step 5: Merging labels with image paths to create final dataset ---")

    # 将处理好的标签与筛选后的正面影像元数据合并
    df_final = pd.merge(df_frontal, df_processed_labels, on=['subject_id', 'study_id'], how='inner')
    
    # 构建每张图片的完整路径
    def build_image_path(row):
        subject_id = str(row['subject_id'])
        study_id = str(row['study_id'])
        dicom_id = row['dicom_id']
        p_group = 'p' + subject_id[:2]
        return os.path.join(MIMIC_JPG_DIR, 'files', p_group, f'p{subject_id}', f's{study_id}', f'{dicom_id}.jpg')

    df_final['image_path'] = df_final.apply(build_image_path, axis=1)
    
    # 选择最终需要的列
    final_cols = ['image_path', 'subject_id', 'study_id'] + chexpert_cols
    df_final = df_final[final_cols]
    
    df_final.to_csv(FINAL_DATASET_CSV, index=False)
    print(f"Final combined dataset saved to {FINAL_DATASET_CSV}. Contains {len(df_final)} image records.")

    # --- 6. 按官方分割文件划分数据集 (Split Dataset) ---
    print("\n--- Step 6: Splitting dataset into train, validate, and test sets ---")
    df_split = pd.read_csv(SPLIT_FILE)
    
    # 合并分割信息
    df_final_split = pd.merge(df_final, df_split, on=['subject_id', 'study_id'], how='inner')
    
    # 创建 train, validate, test 数据集
    df_train = df_final_split[df_final_split['split'] == 'train'].copy()
    df_validate = df_final_split[df_final_split['split'] == 'validate'].copy()
    df_test = df_final_split[df_final_split['split'] == 'test'].copy()
    
    # 保存文件
    df_train.to_csv(TRAIN_CSV, index=False)
    df_validate.to_csv(VALIDATE_CSV, index=False)
    df_test.to_csv(TEST_CSV, index=False)
    
    print("Dataset splitting complete.")
    print(f"Train set: {len(df_train)} images. Saved to {TRAIN_CSV}")
    print(f"Validation set: {len(df_validate)} images. Saved to {VALIDATE_CSV}")
    print(f"Test set: {len(df_test)} images. Saved to {TEST_CSV}")
    print("\nPreprocessing finished successfully!")
