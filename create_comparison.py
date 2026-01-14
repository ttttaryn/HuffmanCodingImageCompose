import os
import matplotlib.pyplot as plt
from PIL import Image

# ================= 配置区域 =================
# 1. 图片名称 (请修改为你用来做实验的那张 BMP 图片名)
TARGET_IMAGE_NAME = "landscope.bmp"  # 例如 "lena.bmp" 或 "1.bmp"

# 2. 想要展示的 Q 值列表
Q_LIST = [1, 5, 10, 20, 50]

# 3. 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")


# ===========================================

def get_file_size_str(path):
    """获取文件大小并格式化为 KB"""
    if os.path.exists(path):
        size_kb = os.path.getsize(path) / 1024
        return f"{size_kb:.2f} KB"
    return "N/A"


def create_grid_comparison():
    # 1. 准备图片列表
    # 格式: (图片路径, 标题, 是否是压缩文件用于计算大小)
    images_to_show = []

    # --- 添加原始图片 ---
    origin_path = os.path.join(INPUT_DIR, TARGET_IMAGE_NAME)
    if not os.path.exists(origin_path):
        print(f"错误: 找不到原始图片 {origin_path}")
        return

    origin_size = get_file_size_str(origin_path)
    images_to_show.append({
        'path': origin_path,
        'title': f"Original\n(Raw: {origin_size})",
        'is_origin': True
    })

    # --- 添加压缩还原后的图片 ---
    name_no_ext = os.path.splitext(TARGET_IMAGE_NAME)[0]

    for q in Q_LIST:
        # 还原图路径 (用于显示)
        restored_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}_q{q}_restored.jpg")
        # 压缩包路径 (用于读取压缩后大小)
        bin_path = os.path.join(OUTPUT_DIR, f"{name_no_ext}_q{q}.bin")

        size_str = get_file_size_str(bin_path)

        images_to_show.append({
            'path': restored_path,
            'title': f"Q={q}\n(Size: {size_str})",
            'is_origin': False
        })

    # 2. 创建画布 (2行3列)
    # 动态计算: 如果有6张图，就是2x3
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(f'Visual Comparison: {TARGET_IMAGE_NAME} (Huffman + Quantization)', fontsize=20)

    # 展平 axes 方便遍历
    axes = axes.flatten()

    # 3. 循环绘图
    for i, ax in enumerate(axes):
        if i < len(images_to_show):
            item = images_to_show[i]
            path = item['path']

            if os.path.exists(path):
                img = Image.open(path)
                ax.imshow(img)
                ax.set_title(item['title'], fontsize=14, fontweight='bold')

                # 如果是压缩图，标题用红色标记，原始图用蓝色
                title_color = 'blue' if item['is_origin'] else '#333333'
                ax.title.set_color(title_color)
            else:
                ax.text(0.5, 0.5, "File Not Found", ha='center', va='center')
                print(f"警告: 找不到文件 {path}，请先运行 main.py 或 experiment_runner.py 生成它")

        # 隐藏坐标轴刻度
        ax.axis('off')

    # 4. 调整布局并保存
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # 留出顶部标题空间

    save_path = os.path.join(BASE_DIR, f"comparison_{name_no_ext}.png")
    plt.savefig(save_path, dpi=150)
    print(f"✅ 对比图已生成: {save_path}")
    plt.show()


if __name__ == "__main__":
    create_grid_comparison()