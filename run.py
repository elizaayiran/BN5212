"""
CheXpert项目启动器
提供便捷的项目运行方式
"""

import os
import sys
import subprocess

def show_menu():
    """显示菜单"""
    print("\n" + "="*50)
    print("🏥 CheXpert 胸部X光分析系统")
    print("="*50)
    print("1. 🔬 运行NLP文本标注")
    print("2. 🚀 启动API服务")
    print("3. 🧪 运行完整测试系统")
    print("4. 🎯 训练深度学习模型")
    print("5. 📊 数据可视化")
    print("6. 📁 查看项目结构")
    print("7. ❌ 退出")
    print("="*50)

def run_nlp_labeler():
    """运行NLP标注器"""
    print("🔬 启动NLP标注器...")
    os.chdir("src")
    subprocess.run([sys.executable, "improved_chexpert_labeler.py"])
    os.chdir("..")

def start_api_service():
    """启动API服务"""
    print("🚀 启动API服务...")
    os.chdir("api")
    subprocess.run([sys.executable, "chexpert_api.py"])
    os.chdir("..")

def run_test_system():
    """运行测试系统"""
    print("🧪 启动完整测试系统...")
    os.chdir("src")
    subprocess.run([sys.executable, "final_test_system.py"])
    os.chdir("..")

def train_model():
    """训练模型"""
    print("🎯 启动模型训练...")
    os.chdir("src")
    subprocess.run([sys.executable, "enhanced_densenet_training.py"])
    os.chdir("..")

def run_visualization():
    """运行可视化"""
    print("📊 启动数据可视化...")
    os.chdir("src")
    subprocess.run([sys.executable, "final_visualize.py"])
    os.chdir("..")

def show_structure():
    """显示项目结构"""
    print("\n📁 项目结构:")
    print("BN5212/")
    print("├── src/          # 源代码")
    print("├── api/          # API服务")
    print("├── data/         # 数据文件")
    print("├── results/      # 结果输出")
    print("├── docs/         # 文档")
    print("└── config/       # 配置文件")
    print("\n详细说明请查看: docs/ORGANIZED_STRUCTURE.md")

def main():
    """主函数"""
    while True:
        show_menu()
        
        try:
            choice = input("\n请选择操作 (1-7): ").strip()
            
            if choice == '1':
                run_nlp_labeler()
            elif choice == '2':
                start_api_service()
            elif choice == '3':
                run_test_system()
            elif choice == '4':
                train_model()
            elif choice == '5':
                run_visualization()
            elif choice == '6':
                show_structure()
            elif choice == '7':
                print("👋 再见！")
                break
            else:
                print("❌ 无效选择，请输入1-7")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，再见！")
            break
        except Exception as e:
            print(f"❌ 运行出错: {e}")
            
        input("\n按Enter键继续...")

if __name__ == "__main__":
    print("🏥 欢迎使用CheXpert胸部X光分析系统!")
    main()