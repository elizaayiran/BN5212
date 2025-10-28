import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
import numpy as np

# Create figure
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
fig.suptitle('DenseNet-121 vs ResNet-50 Architecture Comparison', fontsize=20, fontweight='bold', y=0.98)

# Color scheme
densenet_color = '#2ecc71'  # Green
resnet_color = '#e74c3c'     # Red
block_color = '#3498db'      # Blue
text_color = '#2c3e50'       # Dark gray

# ==================== DenseNet-121 (Left) ====================
ax1.set_xlim(0, 10)
ax1.set_ylim(0, 12)
ax1.axis('off')
ax1.set_title('DenseNet-121\n7M params | AUC: 0.610 | 84.5MB',
              fontsize=16, fontweight='bold', pad=20)

# Input layer
input_box = FancyBboxPatch((3.5, 10.5), 3, 0.8, boxstyle="round,pad=0.1",
                           edgecolor=text_color, facecolor='lightgray', linewidth=2)
ax1.add_patch(input_box)
ax1.text(5, 10.9, 'Input\n224×224×3', ha='center', va='center', fontsize=11, fontweight='bold')

# Conv + Pool
conv_box = FancyBboxPatch((3.5, 9.2), 3, 0.8, boxstyle="round,pad=0.1",
                          edgecolor=text_color, facecolor='lightblue', linewidth=2)
ax1.add_patch(conv_box)
ax1.text(5, 9.6, 'Conv + Pool', ha='center', va='center', fontsize=10, fontweight='bold')

# Dense Blocks
dense_blocks = [
    (8.0, 'Dense Block 1\n(6 layers)'),
    (6.5, 'Dense Block 2\n(12 layers)'),
    (5.0, 'Dense Block 3\n(24 layers)'),
    (3.5, 'Dense Block 4\n(16 layers)')
]

prev_y = 9.2
for i, (y_pos, label) in enumerate(dense_blocks):
    # Dense block
    block = FancyBboxPatch((2.5, y_pos), 5, 1.2, boxstyle="round,pad=0.1",
                           edgecolor=densenet_color, facecolor='#d5f4e6', linewidth=3)
    ax1.add_patch(block)
    ax1.text(5, y_pos + 0.6, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Dense connections (green arrows showing all-to-all connectivity)
    if i > 0:
        # Multiple curved arrows to show dense connectivity
        for j in range(3):
            x_offset = -1 + j * 1
            arrow = FancyArrowPatch((5 + x_offset, prev_y), (5 + x_offset, y_pos + 1.2),
                                   arrowstyle='->', mutation_scale=15, linewidth=2,
                                   color=densenet_color, alpha=0.7,
                                   connectionstyle="arc3,rad=.3")
            ax1.add_patch(arrow)

    # Transition layer between blocks (except last)
    if i < 3:
        trans_y = y_pos - 0.5
        trans_box = FancyBboxPatch((3.5, trans_y), 3, 0.4, boxstyle="round,pad=0.05",
                                   edgecolor='gray', facecolor='lightyellow', linewidth=1.5)
        ax1.add_patch(trans_box)
        ax1.text(5, trans_y + 0.2, 'Transition', ha='center', va='center', fontsize=8)
        prev_y = trans_y
    else:
        prev_y = y_pos

# Global Average Pooling + FC
gap_box = FancyBboxPatch((3.5, 2.0), 3, 0.8, boxstyle="round,pad=0.1",
                         edgecolor=text_color, facecolor='lightblue', linewidth=2)
ax1.add_patch(gap_box)
ax1.text(5, 2.4, 'GAP + FC', ha='center', va='center', fontsize=10, fontweight='bold')

# Output
output_box = FancyBboxPatch((3.5, 0.5), 3, 0.8, boxstyle="round,pad=0.1",
                           edgecolor=text_color, facecolor='lightcoral', linewidth=2)
ax1.add_patch(output_box)
ax1.text(5, 0.9, 'Output\n14 classes', ha='center', va='center', fontsize=11, fontweight='bold')

# Connect last block to GAP
arrow = FancyArrowPatch((5, 3.5), (5, 2.8),
                       arrowstyle='->', mutation_scale=20, linewidth=2.5, color=text_color)
ax1.add_patch(arrow)

# Connect GAP to Output
arrow = FancyArrowPatch((5, 2.0), (5, 1.3),
                       arrowstyle='->', mutation_scale=20, linewidth=2.5, color=text_color)
ax1.add_patch(arrow)

# Key feature annotation
ax1.text(0.5, 6, 'Key Feature:\nDense Connectivity', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=densenet_color, alpha=0.3), color=text_color)
ax1.text(0.5, 5, 'Each layer connects\nto all previous layers', fontsize=9, style='italic')

# ==================== ResNet-50 (Right) ====================
ax2.set_xlim(0, 10)
ax2.set_ylim(0, 12)
ax2.axis('off')
ax2.set_title('ResNet-50\n23.5M params | AUC: 0.627 | 269.8MB',
              fontsize=16, fontweight='bold', pad=20)

# Input layer
input_box = FancyBboxPatch((3.5, 10.5), 3, 0.8, boxstyle="round,pad=0.1",
                           edgecolor=text_color, facecolor='lightgray', linewidth=2)
ax2.add_patch(input_box)
ax2.text(5, 10.9, 'Input\n224×224×3', ha='center', va='center', fontsize=11, fontweight='bold')

# Conv + Pool
conv_box = FancyBboxPatch((3.5, 9.2), 3, 0.8, boxstyle="round,pad=0.1",
                          edgecolor=text_color, facecolor='lightblue', linewidth=2)
ax2.add_patch(conv_box)
ax2.text(5, 9.6, 'Conv + Pool', ha='center', va='center', fontsize=10, fontweight='bold')

# Residual Blocks
residual_blocks = [
    (8.0, 'Conv2_x\n(3 blocks)'),
    (6.5, 'Conv3_x\n(4 blocks)'),
    (5.0, 'Conv4_x\n(6 blocks)'),
    (3.5, 'Conv5_x\n(3 blocks)')
]

prev_y = 9.2
for i, (y_pos, label) in enumerate(residual_blocks):
    # Residual block
    block = FancyBboxPatch((2.5, y_pos), 5, 1.2, boxstyle="round,pad=0.1",
                           edgecolor=resnet_color, facecolor='#ffe6e6', linewidth=3)
    ax2.add_patch(block)
    ax2.text(5, y_pos + 0.6, label, ha='center', va='center', fontsize=10, fontweight='bold')

    # Skip connection (red curved arrow)
    if i >= 0:
        # Main path
        arrow = FancyArrowPatch((5, prev_y), (5, y_pos + 1.2),
                               arrowstyle='->', mutation_scale=15, linewidth=2.5,
                               color=text_color)
        ax2.add_patch(arrow)

        # Skip connection (identity shortcut)
        skip_arrow = FancyArrowPatch((7.5, prev_y), (7.5, y_pos + 0.2),
                                    arrowstyle='->', mutation_scale=20, linewidth=3,
                                    color=resnet_color, alpha=0.8,
                                    connectionstyle="arc3,rad=.5")
        ax2.add_patch(skip_arrow)

        # Addition symbol
        ax2.text(7.8, y_pos + 0.6, '+', ha='center', va='center', fontsize=20,
                fontweight='bold', color=resnet_color)

    prev_y = y_pos

# Global Average Pooling + FC
gap_box = FancyBboxPatch((3.5, 2.0), 3, 0.8, boxstyle="round,pad=0.1",
                         edgecolor=text_color, facecolor='lightblue', linewidth=2)
ax2.add_patch(gap_box)
ax2.text(5, 2.4, 'GAP + FC', ha='center', va='center', fontsize=10, fontweight='bold')

# Output
output_box = FancyBboxPatch((3.5, 0.5), 3, 0.8, boxstyle="round,pad=0.1",
                           edgecolor=text_color, facecolor='lightcoral', linewidth=2)
ax2.add_patch(output_box)
ax2.text(5, 0.9, 'Output\n14 classes', ha='center', va='center', fontsize=11, fontweight='bold')

# Connect last block to GAP
arrow = FancyArrowPatch((5, 3.5), (5, 2.8),
                       arrowstyle='->', mutation_scale=20, linewidth=2.5, color=text_color)
ax2.add_patch(arrow)

# Connect GAP to Output
arrow = FancyArrowPatch((5, 2.0), (5, 1.3),
                       arrowstyle='->', mutation_scale=20, linewidth=2.5, color=text_color)
ax2.add_patch(arrow)

# Key feature annotation
ax2.text(0.5, 6, 'Key Feature:\nSkip Connections', fontsize=11, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor=resnet_color, alpha=0.3), color=text_color)
ax2.text(0.5, 5, 'Identity shortcuts\nease gradient flow', fontsize=9, style='italic')

# ==================== Bottom Comparison Table ====================
fig.text(0.5, 0.12, 'Performance & Efficiency Comparison', ha='center', fontsize=14, fontweight='bold')

table_data = [
    ['Metric', 'DenseNet-121', 'ResNet-50', 'Advantage'],
    ['Parameters', '7.0M', '23.5M', 'DenseNet (3.35×)'],
    ['Model Size', '84.5 MB', '269.8 MB', 'DenseNet (3.19×)'],
    ['Test AUC', '0.610', '0.627', 'ResNet (+2.76%)'],
    ['Efficiency', 'High', 'Lower', 'DenseNet'],
    ['Feature Reuse', 'Excellent', 'Good', 'DenseNet']
]

table = plt.table(cellText=table_data, cellLoc='center', loc='bottom',
                  bbox=[0.15, -0.35, 0.7, 0.22],
                  colWidths=[0.2, 0.2, 0.2, 0.2])

table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2)

# Style table
for i in range(len(table_data)):
    for j in range(len(table_data[0])):
        cell = table[(i, j)]
        if i == 0:  # Header
            cell.set_facecolor('#34495e')
            cell.set_text_props(weight='bold', color='white')
        else:
            if j % 2 == 0:
                cell.set_facecolor('#ecf0f1')
            else:
                cell.set_facecolor('white')
        cell.set_edgecolor('#bdc3c7')
        cell.set_linewidth(1.5)

plt.tight_layout()
plt.subplots_adjust(top=0.93, bottom=0.25)

# Save figure
output_path = 'results/model_comparison/architecture_comparison_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Architecture comparison diagram saved to: {output_path}")
plt.close()
