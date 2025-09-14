"""
简化的CheXpert标注结果可视化
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def analyze_chexpert_results():
    """分析CheXpert标注结果"""
    
    # 读取数据
    print("加载数据...")
    df = pd.read_csv('labeled_reports_with_ids.csv')
    
    # 标签列
    label_columns = [
        'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
        'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 
        'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
        'Pleural Other', 'Fracture', 'Support Devices'
    ]
    
    print(f"总计 {len(df)} 份报告")
    print(f"总计 {df['subject_id'].nunique()} 个独特患者")
    
    # 统计每个标签
    results = {}
    for label in label_columns:
        # 填充空值
        label_data = df[label].fillna('')
        
        positive = (label_data == '1.0').sum()
        negative = (label_data == '0.0').sum()
        uncertain = (label_data == '-1.0').sum()
        unmentioned = (label_data == '').sum()
        
        results[label] = {
            'positive': positive,
            'negative': negative, 
            'uncertain': uncertain,
            'unmentioned': unmentioned
        }
    
    # 创建图表
    create_visualizations(results, label_columns, len(df))
    
    # 打印统计
    print_statistics(results, label_columns, len(df))

def create_visualizations(results, label_columns, total_reports):
    """创建可视化图表"""
    
    # 1. 阳性病例数量
    plt.figure(figsize=(15, 8))
    
    labels = []
    positive_counts = []
    
    for label in label_columns:
        labels.append(label.replace(' ', '\n'))  # 换行显示
        positive_counts.append(results[label]['positive'])
    
    # 按数量排序
    sorted_data = sorted(zip(labels, positive_counts), key=lambda x: x[1], reverse=True)
    sorted_labels, sorted_counts = zip(*sorted_data)
    
    bars = plt.bar(range(len(sorted_labels)), sorted_counts, color='#FF6B6B', alpha=0.7)
    plt.title('CheXpert标签阳性病例数量分布', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('标签', fontsize=12)
    plt.ylabel('阳性病例数', fontsize=12)
    plt.xticks(range(len(sorted_labels)), sorted_labels, rotation=45, ha='right')
    
    # 添加数值标签
    for bar, count in zip(bars, sorted_counts):
        if count > 0:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 5,
                    f'{count}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('chexpert_positive_distribution.png', dpi=300, bbox_inches='tight')
    print("✅ 保存图表: chexpert_positive_distribution.png")
    
    # 2. 阳性率百分比
    plt.figure(figsize=(15, 8))
    
    positive_rates = [(results[label]['positive'] / total_reports * 100) for label in label_columns]
    
    # 按百分比排序
    sorted_data2 = sorted(zip(labels, positive_rates), key=lambda x: x[1], reverse=True)
    sorted_labels2, sorted_rates = zip(*sorted_data2)
    
    bars2 = plt.bar(range(len(sorted_labels2)), sorted_rates, color='#4ECDC4', alpha=0.7)
    plt.title('CheXpert标签阳性率分布', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('标签', fontsize=12)
    plt.ylabel('阳性率 (%)', fontsize=12)
    plt.xticks(range(len(sorted_labels2)), sorted_labels2, rotation=45, ha='right')
    
    # 添加百分比标签
    for bar, rate in zip(bars2, sorted_rates):
        if rate > 0:
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('chexpert_positive_rates.png', dpi=300, bbox_inches='tight')
    print("✅ 保存图表: chexpert_positive_rates.png")
    
    # 3. 前10个最常见发现的详细分布
    plt.figure(figsize=(16, 10))
    
    # 选择阳性病例最多的前10个标签
    top_labels = sorted(label_columns, key=lambda x: results[x]['positive'], reverse=True)[:10]
    
    x_pos = np.arange(len(top_labels))
    width = 0.2
    
    positive_vals = [results[label]['positive'] for label in top_labels]
    negative_vals = [results[label]['negative'] for label in top_labels]
    uncertain_vals = [results[label]['uncertain'] for label in top_labels]
    
    plt.bar(x_pos - width, positive_vals, width, label='阳性', color='#FF6B6B', alpha=0.8)
    plt.bar(x_pos, negative_vals, width, label='阴性', color='#95E1D3', alpha=0.8)
    plt.bar(x_pos + width, uncertain_vals, width, label='不确定', color='#F3D250', alpha=0.8)
    
    plt.title('前10个最常见发现的详细分布', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('标签', fontsize=12)
    plt.ylabel('病例数', fontsize=12)
    plt.xticks(x_pos, [label.replace(' ', '\n') for label in top_labels], rotation=45, ha='right')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('chexpert_top10_detailed.png', dpi=300, bbox_inches='tight')
    print("✅ 保存图表: chexpert_top10_detailed.png")
    
    plt.show()

def print_statistics(results, label_columns, total_reports):
    """打印统计信息"""
    
    print("\n" + "="*60)
    print("           CheXpert标注结果统计报告")
    print("="*60)
    
    print(f"\n📊 总体统计:")
    print(f"   总报告数: {total_reports:,}")
    
    total_positive = sum(results[label]['positive'] for label in label_columns)
    total_negative = sum(results[label]['negative'] for label in label_columns)
    total_uncertain = sum(results[label]['uncertain'] for label in label_columns)
    
    print(f"   总阳性标注: {total_positive:,}")
    print(f"   总阴性标注: {total_negative:,}")
    print(f"   总不确定标注: {total_uncertain:,}")
    
    print(f"\n🔝 前10个最常见的阳性发现:")
    sorted_by_positive = sorted(label_columns, key=lambda x: results[x]['positive'], reverse=True)
    
    for i, label in enumerate(sorted_by_positive[:10], 1):
        count = results[label]['positive']
        percentage = count / total_reports * 100
        print(f"   {i:2d}. {label:<25}: {count:3d} 例 ({percentage:5.1f}%)")
    
    print(f"\n📝 标注提及率最高的前10个标签:")
    sorted_by_mentioned = sorted(label_columns, 
                                key=lambda x: results[x]['positive'] + results[x]['negative'] + results[x]['uncertain'], 
                                reverse=True)
    
    for i, label in enumerate(sorted_by_mentioned[:10], 1):
        mentioned = results[label]['positive'] + results[label]['negative'] + results[label]['uncertain']
        percentage = mentioned / total_reports * 100
        print(f"   {i:2d}. {label:<25}: {mentioned:3d} 次 ({percentage:5.1f}%)")
    
    print("="*60)
    
    # 保存详细统计到CSV
    stats_data = []
    for label in label_columns:
        stats_data.append({
            'Label': label,
            'Positive': results[label]['positive'],
            'Negative': results[label]['negative'],
            'Uncertain': results[label]['uncertain'],
            'Unmentioned': results[label]['unmentioned'],
            'Positive_Rate_%': results[label]['positive'] / total_reports * 100,
            'Mentioned_Total': results[label]['positive'] + results[label]['negative'] + results[label]['uncertain']
        })
    
    stats_df = pd.DataFrame(stats_data)
    stats_df.to_csv('chexpert_detailed_statistics.csv', index=False)
    print("\n✅ 详细统计已保存: chexpert_detailed_statistics.csv")

if __name__ == '__main__':
    try:
        analyze_chexpert_results()
        print("\n🎉 可视化分析完成！")
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()