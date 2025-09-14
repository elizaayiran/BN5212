"""
快速查看CheXpert标注统计
"""

import pandas as pd

# 读取数据
df = pd.read_csv('labeled_reports_with_ids.csv')

# 标签列
label_columns = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
    'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
    'Pleural Other', 'Fracture', 'Support Devices'
]

print("CheXpert标注结果统计")
print("="*50)
print(f"总报告数: {len(df)}")
print(f"总患者数: {df['subject_id'].nunique()}")
print("\n各标签统计:")

# 统计结果
results = []
for label in label_columns:
    # 转换为字符串并处理NaN
    label_data = df[label].astype(str)
    
    positive = (label_data == '1.0').sum()
    negative = (label_data == '0.0').sum() 
    uncertain = (label_data == '-1.0').sum()
    unmentioned = (label_data == 'nan').sum()
    
    total_mentioned = positive + negative + uncertain
    positive_rate = positive / len(df) * 100 if len(df) > 0 else 0
    
    results.append({
        'Label': label,
        'Positive': positive,
        'Negative': negative,
        'Uncertain': uncertain,
        'Unmentioned': unmentioned,
        'Positive_Rate': positive_rate,
        'Total_Mentioned': total_mentioned
    })
    
    print(f"{label:<25}: 阳性={positive:3d} ({positive_rate:5.1f}%), 阴性={negative:3d}, 不确定={uncertain:3d}, 未提及={unmentioned:3d}")

# 按阳性数量排序
print(f"\n按阳性病例数排序 (前10):")
sorted_results = sorted(results, key=lambda x: x['Positive'], reverse=True)[:10]
for i, result in enumerate(sorted_results, 1):
    print(f"{i:2d}. {result['Label']:<25}: {result['Positive']:3d} 例 ({result['Positive_Rate']:5.1f}%)")

# 按提及次数排序  
print(f"\n按提及次数排序 (前10):")
sorted_mentioned = sorted(results, key=lambda x: x['Total_Mentioned'], reverse=True)[:10]
for i, result in enumerate(sorted_mentioned, 1):
    print(f"{i:2d}. {result['Label']:<25}: {result['Total_Mentioned']:3d} 次")

# 保存到CSV
results_df = pd.DataFrame(results)
results_df.to_csv('quick_statistics.csv', index=False)
print(f"\n详细统计已保存到: quick_statistics.csv")