"""
整理后的CheXpert API服务
使用新的项目结构，修复导入路径
"""

from flask import Flask, request, jsonify
import pandas as pd
import sys
import os

# 添加src目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(current_dir), 'src')
sys.path.insert(0, src_dir)

from improved_chexpert_labeler import ImprovedCheXpertLabeler

app = Flask(__name__)
nlp_labeler = None

def initialize_models():
    """初始化NLP模型"""
    global nlp_labeler
    print("🔧 初始化NLP标注器...")
    nlp_labeler = ImprovedCheXpertLabeler()
    print("✅ NLP模型初始化完成")

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'nlp_loaded': nlp_labeler is not None,
        'version': '1.0_organized'
    })

@app.route('/predict/text', methods=['POST'])
def predict_text():
    """文本标注接口"""
    try:
        data = request.get_json()
        if 'report_text' not in data:
            return jsonify({'error': '缺少report_text字段'}), 400
        
        if nlp_labeler is None:
            return jsonify({'error': 'NLP模型未初始化'}), 500
            
        predictions = nlp_labeler.label_report(data['report_text'])
        return jsonify({
            'success': True,
            'predictions': predictions,
            'model': 'Improved CheXpert NLP Labeler'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if nlp_labeler is None:
        return jsonify({'error': '模型未初始化'}), 500
    
    return jsonify({
        'model': {
            'name': 'Improved CheXpert Labeler',
            'type': 'Rule-based NLP',
            'labels': nlp_labeler.labels,
            'total_labels': len(nlp_labeler.labels)
        },
        'structure': 'Organized project structure',
        'api_version': '1.0_clean'
    })

if __name__ == '__main__':
    print("🚀 启动整理后的CheXpert API服务...")
    try:
        initialize_models()
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"❌ 启动失败: {e}")