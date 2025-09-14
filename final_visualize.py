"""
简单的CheXpert结果可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# 读取统计数据
df_stats = pd.read_csv('quick_statistics.csv')

# 创建可视化图表
fig = plt.figure(figsize=(20, 16))
ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)
# 1. 阳性病例数量 (前10个)
top10_positive = df_stats.nlargest(10, 'Positive')
bars1 = ax1.barh(range(len(top10_positive)), top10_positive['Positive'], color='#FF6B6B', alpha=0.7)
ax1.set_yticks(range(len(top10_positive)))
ax1.set_yticklabels([label.replace(' ', '\n') for label in top10_positive['Label']])
ax1.set_xlabel('Positive Cases')
ax1.set_title('Top 10 - Positive Cases Count', fontweight='bold')

# 添加数值标签
for i, bar in enumerate(bars1):
    width = bar.get_width()
    ax1.text(width + 5, bar.get_y() + bar.get_height()/2, 
             f'{int(width)}', ha='left', va='center')

# 2. 阳性率百分比 (前10个)
bars2 = ax2.barh(range(len(top10_positive)), top10_positive['Positive_Rate'], color='#4ECDC4', alpha=0.7)
ax2.set_yticks(range(len(top10_positive)))
ax2.set_yticklabels([label.replace(' ', '\n') for label in top10_positive['Label']])
ax2.set_xlabel('Positive Rate (%)')
ax2.set_title('Top 10 - Positive Rate (%)', fontweight='bold')

# 添加百分比标签
for i, bar in enumerate(bars2):
    width = bar.get_width()
    ax2.text(width + 0.5, bar.get_y() + bar.get_height()/2, 
             f'{width:.1f}%', ha='left', va='center')

# 3. 最常被提及的标签 (前8个)
top8_mentioned = df_stats.nlargest(8, 'Total_Mentioned')
bars3 = ax3.bar(range(len(top8_mentioned)), top8_mentioned['Total_Mentioned'], color='#95E1D3', alpha=0.7)
ax3.set_xticks(range(len(top8_mentioned)))
ax3.set_xticklabels([label.replace(' ', '\n') for label in top8_mentioned['Label']], rotation=45, ha='right')
ax3.set_ylabel('Total Mentions')
ax3.set_title('Top 8 - Most Mentioned Labels', fontweight='bold')

# 添加数值标签
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{int(height)}', ha='center', va='bottom')

# 4. 前6个最常见发现的详细分布
top6_labels = df_stats.nlargest(6, 'Positive')['Label'].tolist()

x = np.arange(len(top6_labels))
width = 0.25

positive_vals = [df_stats[df_stats['Label'] == label]['Positive'].iloc[0] for label in top6_labels]
negative_vals = [df_stats[df_stats['Label'] == label]['Negative'].iloc[0] for label in top6_labels]
uncertain_vals = [df_stats[df_stats['Label'] == label]['Uncertain'].iloc[0] for label in top6_labels]

ax4.bar(x - width, positive_vals, width, label='Positive', color='#FF6B6B', alpha=0.8)
ax4.bar(x, negative_vals, width, label='Negative', color='#95E1D3', alpha=0.8)
ax4.bar(x + width, uncertain_vals, width, label='Uncertain', color='#F3D250', alpha=0.8)

ax4.set_xlabel('Labels')
ax4.set_ylabel('Count')
ax4.set_title('Top 6 - Detailed Distribution', fontweight='bold')
ax4.set_xticks(x)
ax4.set_xticklabels([label.replace(' ', '\n') for label in top6_labels], rotation=45, ha='right')
ax4.legend()

plt.tight_layout()
plt.savefig('chexpert_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("✅ 综合分析图表已保存: chexpert_comprehensive_analysis.png")

# 创建数据质量总结
print("\n" + "="*60)
print("                数据质量总结")
print("="*60)

total_reports = 934
total_patients = 242

print(f"📊 数据集概览:")
print(f"   • 总报告数: {total_reports:,}")
print(f"   • 总患者数: {total_patients:,}")
print(f"   • 平均每患者报告数: {total_reports/total_patients:.1f}")

print(f"\n🔍 标注质量:")
total_positive = df_stats['Positive'].sum()
total_negative = df_stats['Negative'].sum()
total_uncertain = df_stats['Uncertain'].sum()

print(f"   • 总阳性标注: {total_positive:,}")
print(f"   • 总阴性标注: {total_negative:,}")
print(f"   • 总不确定标注: {total_uncertain:,}")
print(f"   • 平均每报告发现数: {total_positive/total_reports:.1f}")

print(f"\n🏆 关键发现:")
top3 = df_stats.nlargest(3, 'Positive')
for i, (_, row) in enumerate(top3.iterrows(), 1):
    print(f"   {i}. {row['Label']}: {row['Positive']} 例 ({row['Positive_Rate']:.1f}%)")

print(f"\n📈 数据适用性:")
print(f"   • 多标签分类: ✅ (14个独立标签)")
print(f"   • 不确定性处理: ✅ (包含不确定标注)")
print(f"   • 患者追踪: ✅ (subject_id + study_id)")
print(f"   • 临床相关性: ✅ (基于真实胸部X光报告)")

print("="*60)