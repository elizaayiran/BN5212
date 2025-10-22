"""
改进版本的CheXpert Labeler
保留subject_id和study_id信息
"""

import pandas as pd
import re
import argparse
from pathlib import Path

class ImprovedCheXpertLabeler:
    def __init__(self):
        # 定义14个观察标签
        self.labels = [
            'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 
            'Lung Lesion', 'Lung Opacity', 'Edema', 'Consolidation', 
            'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 
            'Pleural Other', 'Fracture', 'Support Devices'
        ]
        
        # 定义关键词模式（基于CheXpert论文的词汇）
        self.patterns = {
            'No Finding': {
                'positive': [r'\bno\s+finding\b', r'\bnormal\b', r'\bclear\s+lungs?\b', 
                           r'\bno\s+acute\s+cardiopulmonary\s+process\b'],
                'negative': [],
                'uncertain': []
            },
            'Enlarged Cardiomediastinum': {
                'positive': [r'\benlarged\s+cardiomediastinum\b', r'\bwidened\s+mediastinum\b'],
                'negative': [r'\bno\s+enlarged\s+cardiomediastinum\b', r'\bnormal\s+mediastinum\b'],
                'uncertain': [r'\bmay\s+be\s+enlarged\s+cardiomediastinum\b']
            },
            'Cardiomegaly': {
                'positive': [r'\bcardiomegaly\b', r'\benlarged\s+heart\b', r'\bheart\s+size\s+enlarged\b'],
                'negative': [r'\bno\s+cardiomegaly\b', r'\bnormal\s+heart\s+size\b', r'\bheart\s+size\s+normal\b'],
                'uncertain': [r'\bmild\s+cardiomegaly\b', r'\bpossible\s+cardiomegaly\b']
            },
            'Lung Lesion': {
                'positive': [r'\blung\s+lesion\b', r'\bpulmonary\s+nodule\b', r'\bmass\b'],
                'negative': [r'\bno\s+lung\s+lesion\b', r'\bno\s+nodule\b'],
                'uncertain': [r'\bpossible\s+lesion\b', r'\bmay\s+represent\s+nodule\b']
            },
            'Lung Opacity': {
                'positive': [r'\bopacity\b', r'\bopacities\b', r'\bairspace\s+disease\b', 
                           r'\bpatchy\s+opacity\b', r'\bdiffuse\s+opacity\b'],
                'negative': [r'\bno\s+opacity\b', r'\bclear\s+lungs?\b'],
                'uncertain': [r'\bminimal\s+opacity\b', r'\bmild\s+opacity\b']
            },
            'Edema': {
                'positive': [r'\bedema\b', r'\bpulmonary\s+edema\b'],
                'negative': [r'\bno\s+edema\b', r'\bwithout\s+edema\b'],
                'uncertain': [r'\bmild\s+edema\b', r'\bpossible\s+edema\b']
            },
            'Consolidation': {
                'positive': [r'\bconsolidation\b', r'\bconsolidative\b'],
                'negative': [r'\bno\s+consolidation\b'],
                'uncertain': [r'\bmay\s+reflect\s+consolidation\b', r'\bpossible\s+consolidation\b']
            },
            'Pneumonia': {
                'positive': [r'\bpneumonia\b', r'\bpneumonic\b'],
                'negative': [r'\bno\s+pneumonia\b'],
                'uncertain': [r'\bpossible\s+pneumonia\b', r'\bmay\s+represent\s+pneumonia\b']
            },
            'Atelectasis': {
                'positive': [r'\batelectasis\b', r'\batelectatic\b'],
                'negative': [r'\bno\s+atelectasis\b'],
                'uncertain': [r'\bmay\s+reflect\s+atelectasis\b', r'\bpossible\s+atelectasis\b']
            },
            'Pneumothorax': {
                'positive': [r'\bpneumothorax\b'],
                'negative': [r'\bno\s+pneumothorax\b'],
                'uncertain': [r'\bpossible\s+pneumothorax\b']
            },
            'Pleural Effusion': {
                'positive': [r'\bpleural\s+effusion\b', r'\beffusion\b'],
                'negative': [r'\bno\s+effusion\b', r'\bno\s+pleural\s+effusion\b'],
                'uncertain': [r'\bpossible\s+effusion\b', r'\bmild\s+effusion\b']
            },
            'Pleural Other': {
                'positive': [r'\bpleural\s+thickening\b', r'\bpleural\s+scarring\b'],
                'negative': [r'\bno\s+pleural\s+abnormality\b'],
                'uncertain': [r'\bpossible\s+pleural\s+abnormality\b']
            },
            'Fracture': {
                'positive': [r'\bfracture\b', r'\bfractured\b'],
                'negative': [r'\bno\s+fracture\b'],
                'uncertain': [r'\bpossible\s+fracture\b']
            },
            'Support Devices': {
                'positive': [r'\bendotracheal\s+tube\b', r'\bng\s+tube\b', r'\bcatheter\b', 
                           r'\bpacemaker\b', r'\bwires?\b'],
                'negative': [r'\bno\s+support\s+devices\b'],
                'uncertain': []
            }
        }
    
    def preprocess_text(self, text):
        """预处理文本"""
        if pd.isna(text):
            return ""
        # 转换为小写
        text = str(text).lower()
        # 移除多余的空格
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def extract_label(self, text, label_name):
        """为单个标签提取标注"""
        if label_name not in self.patterns:
            return ''
        
        text = self.preprocess_text(text)
        patterns = self.patterns[label_name]
        
        # 检查否定模式
        for pattern in patterns['negative']:
            if re.search(pattern, text, re.IGNORECASE):
                return '0.0'
        
        # 检查不确定模式
        for pattern in patterns['uncertain']:
            if re.search(pattern, text, re.IGNORECASE):
                return '-1.0'
        
        # 检查阳性模式
        for pattern in patterns['positive']:
            if re.search(pattern, text, re.IGNORECASE):
                return '1.0'
        
        # 默认返回空值（未提及）
        return ''
    
    def process_no_finding(self, labels_dict):
        """处理"No Finding"标签的特殊逻辑"""
        # 如果其他任何标签为阳性，则"No Finding"为阴性
        other_labels = [k for k in labels_dict.keys() if k != 'No Finding']
        has_positive_finding = any(labels_dict[label] == '1.0' for label in other_labels)
        
        if has_positive_finding:
            return '0.0'
        elif labels_dict['No Finding'] == '1.0':
            return '1.0'
        else:
            return labels_dict['No Finding']
    
    def label_report(self, report_text):
        """标注单个报告"""
        labels_dict = {}
        
        # 为每个标签提取标注
        for label in self.labels:
            labels_dict[label] = self.extract_label(report_text, label)
        
        # 处理"No Finding"的特殊逻辑
        labels_dict['No Finding'] = self.process_no_finding(labels_dict)
        
        return labels_dict
    
    def label_csv_with_ids(self, input_path, output_path):
        """处理带有ID的CSV文件"""
        print(f"读取报告文件: {input_path}")
        
        # 读取CSV文件
        df = pd.read_csv(input_path)
        
        print(f"总共 {len(df)} 份报告需要处理")
        print(f"文件列: {list(df.columns)}")
        
        # 检查列名
        if 'report_text' not in df.columns:
            print("错误: 找不到'report_text'列")
            return
        
        # 创建结果DataFrame
        result_data = []
        
        for i, (idx, row) in enumerate(df.iterrows()):
            # 保留原有的ID信息
            result_row = {}
            if 'subject_id' in df.columns:
                result_row['subject_id'] = row['subject_id']
            if 'study_id' in df.columns:
                result_row['study_id'] = row['study_id']
            
            # 添加报告文本
            report_text = row['report_text']
            result_row['Reports'] = report_text
            
            # 进行标注
            labels = self.label_report(report_text)
            result_row.update(labels)
            
            result_data.append(result_row)
            
            if (i + 1) % 100 == 0:
                print(f"已处理 {i + 1} 份报告...")
        
        # 创建结果DataFrame
        result_df = pd.DataFrame(result_data)
        
        # 确保列的顺序：ID列 + Reports + 标签列
        id_columns = [col for col in ['subject_id', 'study_id'] if col in result_df.columns]
        columns = id_columns + ['Reports'] + self.labels
        result_df = result_df[columns]
        
        # 保存结果
        result_df.to_csv(output_path, index=False)
        print(f"标注结果已保存到: {output_path}")
        
        # 显示统计信息
        print("\n标注统计:")
        for label in self.labels:
            positive = (result_df[label] == '1.0').sum()
            negative = (result_df[label] == '0.0').sum()
            uncertain = (result_df[label] == '-1.0').sum()
            unmentioned = (result_df[label] == '').sum()
            print(f"{label}: 阳性={positive}, 阴性={negative}, 不确定={uncertain}, 未提及={unmentioned}")


def main():
    parser = argparse.ArgumentParser(description='改进版CheXpert Labeler（保留ID信息）')
    parser.add_argument('--reports_path', required=True, help='输入报告CSV文件路径（包含ID列）')
    parser.add_argument('--output_path', required=True, help='输出标注CSV文件路径')
    
    args = parser.parse_args()
    
    # 检查输入文件是否存在
    if not Path(args.reports_path).exists():
        print(f"错误: 输入文件不存在: {args.reports_path}")
        return
    
    # 创建labeler并处理
    labeler = ImprovedCheXpertLabeler()
    labeler.label_csv_with_ids(args.reports_path, args.output_path)


if __name__ == '__main__':
    main()