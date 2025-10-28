# 模型架构对比图生成 Prompt

## 目标
创建一个专业的、并排对比的DenseNet-121与ResNet-50架构图，用于学术报告。

---

## 完整 Prompt（推荐使用 DALL-E 3 或 Midjourney）

```
Create a professional, side-by-side architectural comparison diagram for academic publication, showing DenseNet-121 and ResNet-50 neural networks for medical image classification.

Layout: Two columns, equal width, clean white background

LEFT COLUMN - DenseNet-121:
- Title: "DenseNet-121 Architecture"
- Subtitle: "7M Parameters | Test AUC: 0.610 | 84.5MB Model Size"
- Network flow (top to bottom):
  * Input Layer: "X-ray Image 224×224×3" (light blue box)
  * Conv + Pool: "7×7 Conv, MaxPool" (blue box)
  * Dense Block 1: "6 layers, 64→256 channels" (orange/yellow box with internal dense connections)
  * Transition Layer 1: "1×1 Conv + 2×2 AvgPool" (green box)
  * Dense Block 2: "12 layers, 256→512" (orange/yellow box)
  * Transition Layer 2 (green box)
  * Dense Block 3: "24 layers, 512→1024" (orange/yellow box)
  * Transition Layer 3 (green box)
  * Dense Block 4: "16 layers, 1024→1024" (orange/yellow box)
  * Global Average Pooling: "7×7→1×1" (green box)
  * Fully Connected: "1024→14" (red box)
  * Output: "14 Disease Probabilities (Multi-label)" (dark green box)
- KEY FEATURE: Show dense connectivity with curved GREEN arrows connecting all previous layers to current layer within dense blocks
- Add annotation: "Dense Connectivity = Feature Reuse + Gradient Flow"

RIGHT COLUMN - ResNet-50:
- Use the existing ResNet-50 diagram structure you already have
- Title: "ResNet-50 Architecture"
- Subtitle: "23.5M Parameters | Test AUC: 0.627 | 269.8MB Model Size"
- KEY FEATURE: Emphasize skip connections with RED arrows

Visual Style:
- Clean, technical diagram suitable for IEEE/NeurIPS publication
- Color scheme:
  * Light blue: Input layer
  * Blue: Convolution layers
  * Green: Pooling layers
  * Orange/Yellow: Main computation blocks (Dense Blocks / Residual Blocks)
  * Red: Fully connected layers
  * Dark green: Output layer
- Font: Arial or Helvetica, clean sans-serif
- Include spatial dimension labels (e.g., "56×56×256", "28×28×512")
- Arrow thickness: 2-3pt for main flow, 1-2pt for skip/dense connections

Bottom Section:
- Centered comparison table:
  | Metric | DenseNet-121 | ResNet-50 |
  |--------|--------------|-----------|
  | Parameters | 7M | 23.5M |
  | Model Size | 84.5MB | 269.8MB |
  | Test AUC | 0.610 | 0.627 |
  | Parameter Efficiency | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

- Add annotation box: "Key Insight: DenseNet-121 achieves 97% of ResNet-50's performance with only 30% of the parameters, demonstrating superior efficiency for resource-constrained clinical deployment."

Aspect ratio: 16:9 or 4:3 landscape
Resolution: Minimum 300 DPI for publication quality
Format: PNG with transparent background preferred
```

---

## 替代 Prompt（更简洁，适合代码生成工具如 Python matplotlib）

```python
# Architecture Comparison Diagram Generation Prompt

Create a side-by-side comparison diagram using matplotlib/seaborn with the following specifications:

## Figure Layout
- Figure size: (16, 10) inches at 300 DPI
- Two subplots: Left = DenseNet-121, Right = ResNet-50
- White background, professional academic style

## DenseNet-121 (Left Panel)
Architecture blocks (top to bottom):
1. Input: "224×224×3" - light blue rectangle
2. Conv+Pool: "7×7 Conv, 3×3 MaxPool" - blue
3. Dense Block 1: "6 layers, 256 channels" - orange
4. Transition 1: "1×1 Conv, 2×2 Pool" - green
5. Dense Block 2: "12 layers, 512 channels" - orange
6. Transition 2 - green
7. Dense Block 3: "24 layers, 1024 channels" - orange
8. Transition 3 - green
9. Dense Block 4: "16 layers, 1024 channels" - orange
10. Global Average Pooling: "7×7→1×1" - green
11. FC Layer: "1024→14" - red
12. Output: "14 Classes" - dark green

Dense Connections: Draw curved arrows within each Dense Block showing all-to-all connectivity
Annotation: "7M Parameters | AUC: 0.610"

## ResNet-50 (Right Panel)
Architecture blocks:
1. Input: "224×224×3" - light blue
2. Conv1: "7×7×64, MaxPool" - blue
3. ResLayer1: "3 blocks, 64→256" - yellow
4. ResLayer2: "4 blocks, 128→512" - yellow
5. ResLayer3: "6 blocks, 256→1024" - yellow
6. ResLayer4: "3 blocks, 512→2048" - yellow
7. Global Average Pooling: "7×7→1×1" - green
8. FC Layer: "2048→14" - red
9. Output: "14 Classes" - dark green

Skip Connections: Draw red curved arrows bypassing each residual block
Annotation: "23.5M Parameters | AUC: 0.627"

## Bottom Comparison Table
Add text box at bottom:
"Parameter Efficiency: DenseNet-121 uses 70% fewer parameters (7M vs 23.5M)
Performance Gap: ΔAUC = +0.017 (2.8% relative improvement)
Conclusion: Dense connectivity achieves comparable accuracy with superior efficiency"

## Color Scheme
- Input: #E3F2FD (light blue)
- Convolution: #2196F3 (blue)
- Pooling: #66BB6A (green)
- Dense/Residual Blocks: #FFA726 (orange)
- Fully Connected: #EF5350 (red)
- Output: #2E7D32 (dark green)
- Skip/Dense connections: Use alpha=0.6 for transparency

## Font Specifications
- Title: 16pt, bold
- Subtitles: 12pt, semibold
- Layer labels: 10pt, regular
- Dimension labels: 8pt, italic
```

---

## 关键参数总结

### DenseNet-121 技术参数
- **Total Parameters**: 7,037,504 (约7M)
- **Model Size**: 84.5 MB
- **Test AUC**: 0.6100 (mean across 14 classes)
- **Best Validation AUC**: 0.6813 (Epoch 4)
- **Architecture**:
  - Dense Block 1: 6 layers
  - Dense Block 2: 12 layers
  - Dense Block 3: 24 layers
  - Dense Block 4: 16 layers
  - Growth rate: k=32
  - Compression factor: θ=0.5
- **Final feature map**: 7×7×1024
- **Dense connectivity**: Each layer receives feature maps from all preceding layers

### ResNet-50 技术参数
- **Total Parameters**: 23,512,078 (约23.5M)
- **Model Size**: 269.8 MB (best_model_resnet50.pth: 282 MB)
- **Test AUC**: 0.6268 (mean across 14 classes)
- **Best Validation AUC**: 0.6792 (Epoch 5)
- **Architecture**:
  - ResLayer1: 3 blocks, 64→256 channels
  - ResLayer2: 4 blocks, 128→512 channels
  - ResLayer3: 6 blocks, 256→1024 channels
  - ResLayer4: 3 blocks, 512→2048 channels
- **Final feature map**: 7×7×2048
- **Skip connections**: Identity shortcuts every 2-3 conv layers

### 对比关键指标
- **Parameter Ratio**: ResNet-50 / DenseNet-121 = 3.35× more parameters
- **AUC Improvement**: +0.0168 (2.76% relative improvement)
- **Efficiency Metric**: DenseNet-121 achieves 97.3% of ResNet-50's AUC with 29.9% of parameters
- **Training Time**: DenseNet-121 ~2.5 hours, ResNet-50 ~3 hours (RTX 4060)

---

## 使用工具建议

### Option 1: AI绘图工具
- **DALL-E 3**: 适合快速生成高质量示意图
- **Midjourney**: 适合艺术风格的技术图
- **Stable Diffusion**: 可本地运行，自定义程度高

### Option 2: 专业绘图工具
- **draw.io / diagrams.net**: 免费在线工具，导出高清PNG
- **Lucidchart**: 专业流程图工具
- **Microsoft Visio**: 企业级绘图

### Option 3: 代码生成（Python）
```python
# 使用 matplotlib + networkx 生成架构图
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# [完整代码见下方Python脚本]
```

---

## Python代码生成脚本（完整版）

如果你想用代码生成，我可以提供一个完整的Python脚本：

```python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

def create_architecture_comparison():
    """
    生成DenseNet-121 vs ResNet-50架构对比图
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), dpi=300)

    # 颜色方案
    colors = {
        'input': '#E3F2FD',
        'conv': '#2196F3',
        'pool': '#66BB6A',
        'block': '#FFA726',
        'fc': '#EF5350',
        'output': '#2E7D32'
    }

    # === DenseNet-121 (左侧) ===
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 20)
    ax1.axis('off')
    ax1.set_title('DenseNet-121 Architecture\n7M Parameters | AUC: 0.610',
                  fontsize=14, fontweight='bold', pad=20)

    # 绘制各层
    y_pos = 18
    layer_height = 0.8

    # Input
    ax1.add_patch(FancyBboxPatch((1, y_pos), 8, layer_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['input'],
                                  edgecolor='black', linewidth=2))
    ax1.text(5, y_pos+0.4, 'Input: 224×224×3', ha='center', va='center', fontsize=10)
    y_pos -= 1.2

    # Conv + Pool
    ax1.add_patch(FancyBboxPatch((1, y_pos), 8, layer_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['conv'],
                                  edgecolor='black', linewidth=2))
    ax1.text(5, y_pos+0.4, 'Conv 7×7 + MaxPool', ha='center', va='center', fontsize=10)
    y_pos -= 1.5

    # Dense Blocks with transitions
    dense_blocks = [
        ('Dense Block 1\n6 layers, 256ch', 1.2),
        ('Transition 1', 0.6),
        ('Dense Block 2\n12 layers, 512ch', 1.5),
        ('Transition 2', 0.6),
        ('Dense Block 3\n24 layers, 1024ch', 2.0),
        ('Transition 3', 0.6),
        ('Dense Block 4\n16 layers, 1024ch', 1.5),
    ]

    for label, height in dense_blocks:
        color = colors['block'] if 'Dense' in label else colors['pool']
        ax1.add_patch(FancyBboxPatch((1, y_pos-height), 8, height,
                                      boxstyle="round,pad=0.1",
                                      facecolor=color,
                                      edgecolor='black', linewidth=2))
        ax1.text(5, y_pos-height/2, label, ha='center', va='center',
                fontsize=9, multialignment='center')
        y_pos -= height + 0.3

    # GAP + FC + Output
    final_layers = [
        ('Global Avg Pool\n7×7→1×1', colors['pool'], 0.8),
        ('FC: 1024→14', colors['fc'], 0.8),
        ('14 Disease Classes', colors['output'], 0.8),
    ]

    for label, color, height in final_layers:
        ax1.add_patch(FancyBboxPatch((1, y_pos), 8, height,
                                      boxstyle="round,pad=0.1",
                                      facecolor=color,
                                      edgecolor='black', linewidth=2))
        ax1.text(5, y_pos+height/2, label, ha='center', va='center', fontsize=10)
        y_pos -= height + 0.3

    # Dense connectivity arrows (示意性)
    ax1.annotate('', xy=(9, 12), xytext=(9, 14),
                arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.6))
    ax1.text(9.5, 13, 'Dense\nConnections', fontsize=8, style='italic')

    # === ResNet-50 (右侧) ===
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 20)
    ax2.axis('off')
    ax2.set_title('ResNet-50 Architecture\n23.5M Parameters | AUC: 0.627',
                  fontsize=14, fontweight='bold', pad=20)

    y_pos = 18

    # Input
    ax2.add_patch(FancyBboxPatch((1, y_pos), 8, layer_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['input'],
                                  edgecolor='black', linewidth=2))
    ax2.text(5, y_pos+0.4, 'Input: 224×224×3', ha='center', va='center', fontsize=10)
    y_pos -= 1.2

    # Conv1
    ax2.add_patch(FancyBboxPatch((1, y_pos), 8, layer_height,
                                  boxstyle="round,pad=0.1",
                                  facecolor=colors['conv'],
                                  edgecolor='black', linewidth=2))
    ax2.text(5, y_pos+0.4, 'Conv 7×7×64 + MaxPool', ha='center', va='center', fontsize=10)
    y_pos -= 1.5

    # Residual Layers
    res_layers = [
        ('ResLayer1\n3 blocks, 64→256', 1.2),
        ('ResLayer2\n4 blocks, 128→512', 1.4),
        ('ResLayer3\n6 blocks, 256→1024', 1.8),
        ('ResLayer4\n3 blocks, 512→2048', 1.2),
    ]

    for label, height in res_layers:
        # Main block
        ax2.add_patch(FancyBboxPatch((1, y_pos-height), 8, height,
                                      boxstyle="round,pad=0.1",
                                      facecolor=colors['block'],
                                      edgecolor='black', linewidth=2))
        ax2.text(5, y_pos-height/2, label, ha='center', va='center',
                fontsize=9, multialignment='center')

        # Skip connection arrow
        arrow = FancyArrowPatch((9.5, y_pos), (9.5, y_pos-height),
                               arrowstyle='->', mutation_scale=20,
                               color='red', lw=2.5, alpha=0.8,
                               connectionstyle="arc3,rad=0.3")
        ax2.add_patch(arrow)

        y_pos -= height + 0.3

    # GAP + FC + Output
    for label, color, height in final_layers:
        ax2.add_patch(FancyBboxPatch((1, y_pos), 8, height,
                                      boxstyle="round,pad=0.1",
                                      facecolor=color,
                                      edgecolor='black', linewidth=2))
        ax2.text(5, y_pos+height/2, label, ha='center', va='center', fontsize=10)
        y_pos -= height + 0.3

    # Skip connection label
    ax2.text(9.8, 12, 'Skip\nConnections', fontsize=8, style='italic', color='red')

    # === Bottom comparison text ===
    fig.text(0.5, 0.02,
             'Key Insight: DenseNet-121 achieves 97% of ResNet-50\'s performance with only 30% of the parameters\n'
             'Parameter Efficiency: 7M vs 23.5M | Performance: AUC 0.610 vs 0.627 (Δ=+0.017)',
             ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig('model_architecture_comparison.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("✅ Architecture comparison saved as 'model_architecture_comparison.png'")

    return fig

# 运行生成
if __name__ == '__main__':
    create_architecture_comparison()
    plt.show()
```

---

## 输出文件建议

**文件名**: `DenseNet_vs_ResNet_Architecture_Comparison.png`

**建议尺寸**:
- 报告单栏图: 3000×2000 px (10×6.7 inches @ 300 DPI)
- 报告双栏图: 4800×3000 px (16×10 inches @ 300 DPI)
- 演示文稿: 1920×1080 px (16:9)

**格式要求**:
- PNG格式（保留清晰度）
- 300 DPI（打印质量）
- RGB色彩空间
- 透明背景或白色背景

---

## 使用步骤

### 如果使用AI工具（DALL-E/Midjourney）:
1. 复制上方"完整 Prompt"
2. 粘贴到AI工具对话框
3. 生成后检查技术参数是否准确
4. 如需调整，添加具体修改指令

### 如果使用Python脚本:
1. 复制上方Python代码
2. 安装依赖: `pip install matplotlib numpy`
3. 运行脚本: `python create_architecture_diagram.py`
4. 检查生成的PNG文件

### 如果使用draw.io:
1. 打开 https://app.diagrams.net/
2. 按照"关键参数总结"中的架构手动绘制
3. 导出为高分辨率PNG (File → Export as → PNG, 300% zoom)

---

## ✅ 验证清单

生成后请检查:
- [ ] DenseNet-121参数量显示为7M
- [ ] ResNet-50参数量显示为23.5M
- [ ] AUC数值正确 (0.610 vs 0.627)
- [ ] Dense connectivity用绿色箭头表示
- [ ] Skip connections用红色箭头表示
- [ ] 所有层的尺寸标注清晰
- [ ] 图片分辨率≥300 DPI
- [ ] 颜色对比度适合打印

---

**生成完成后，这张对比图可以作为报告的 Figure 2，完美支持你的"DenseNet参数效率优势"论述！**
