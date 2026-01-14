import os
import csv
import time
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from huffman import HuffmanCompressor
import utils

# ================= 实验配置区域 =================
# 1. 定义你要测试的 Q 值列表
Q_LIST = [1, 5, 10, 20, 30, 50]

# 2. 路径设置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
RESULT_CSV = "experiment_results.csv"  # 结果保存的文件名

# 3. 支持的格式
SUPPORTED_EXTENSIONS = ('.bmp',)


# ===============================================

def run_experiment():
    # 准备工作
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录 {INPUT_DIR} 不存在")
        return
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    images = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]
    if not images:
        print("未找到图片文件")
        return

    # 初始化 CSV 文件头
    headers = ['Filename', 'Q_Factor', 'Raw_Size(KB)', 'Compressed_Size(KB)', 'Compression_Rate(%)', 'PSNR(dB)',
               'Time(s)']

    # 创建/覆盖 CSV 文件
    with open(RESULT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    print(f"=== 开始自动化实验 ===")
    print(f"测试 Q 值: {Q_LIST}")
    print(f"图片数量: {len(images)}")
    print("-" * 50)

    compressor = HuffmanCompressor()

    # --- 双重循环：遍历图片 -> 遍历 Q 值 ---
    for img_name in images:
        input_path = os.path.join(INPUT_DIR, img_name)
        # 读取原图 (Planar模式)
        pixels, shape = utils.read_image(input_path)
        raw_size_bytes = len(pixels)
        raw_size_kb = raw_size_bytes / 1024
        channels = 3 if len(shape) == 3 else 1

        print(f"正在处理: {img_name} (Raw: {raw_size_kb:.2f} KB)")

        for q in Q_LIST:
            try:
                start_time = time.time()

                # 1. 压缩
                compressed_bytes, frequency = compressor.compress_data(pixels, quantization_factor=q)

                # 计算压缩后大小 (模拟写入文件后的大小: Header + Freq + Data)
                # Header: H(4)+W(4)+C(1)+Q(4) = 13 bytes
                # Freq: pickle dump size + 4 bytes length
                freq_bytes = pickle.dumps(frequency)
                total_compressed_size = 13 + 4 + len(freq_bytes) + len(compressed_bytes)
                comp_size_kb = total_compressed_size / 1024

                # 计算压缩率 (相对于 Raw Data)
                rate = (1 - total_compressed_size / raw_size_bytes) * 100

                # 2. 解压还原 (为了计算 PSNR)
                restored_pixels = compressor.decompress_data(compressed_bytes, frequency, quantization_factor=q)

                # 3. 计算 PSNR
                if len(pixels) == len(restored_pixels):
                    psnr = utils.calculate_psnr(pixels, restored_pixels)
                else:
                    psnr = 0

                elapsed = time.time() - start_time

                # 4. 写入 CSV
                with open(RESULT_CSV, 'a', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(
                        [img_name, q, round(raw_size_kb, 2), round(comp_size_kb, 2), round(rate, 2), round(psnr, 2),
                         round(elapsed, 3)])

                print(f"  -> Q={q:<3} | Rate: {rate:>5.2f}% | PSNR: {psnr:>5.2f} dB | Time: {elapsed:.2f}s")

                # (可选) 保存一张样图，避免磁盘爆炸，可以只保存每张图的 Q=10 或 Q=50
                # if q in [10, 50]:
                #     save_path = os.path.join(OUTPUT_DIR, f"{os.path.splitext(img_name)[0]}_q{q}.jpg")
                #     shape_to_save = (shape[0], shape[1], 3) if channels==3 else (shape[0], shape[1])
                #     utils.save_image_from_array(restored_pixels, shape_to_save, save_path)

            except Exception as e:
                print(f"  -> Q={q} 失败: {e}")

        print("-" * 30)

    print(f"\n实验结束！数据已保存至 {RESULT_CSV}")
    return RESULT_CSV


def plot_results(csv_path):
    """读取CSV并自动画图"""
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        print("读取CSV失败，可能需要安装pandas: pip install pandas")
        return

    # 按图片分组画图
    unique_images = df['Filename'].unique()

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    for img_name in unique_images:
        subset = df[df['Filename'] == img_name]

        # 排序
        subset = subset.sort_values(by='Q_Factor')

        q_vals = subset['Q_Factor']
        rates = subset['Compression_Rate(%)']
        psnrs = subset['PSNR(dB)']

        # === 绘图 ===
        fig, ax1 = plt.subplots(figsize=(10, 6))

        # 左轴：压缩率
        color1 = 'tab:blue'
        ax1.set_xlabel('量化因子 (Q)')
        ax1.set_ylabel('压缩率 (%)', color=color1, fontsize=12, fontweight='bold')
        l1 = ax1.plot(q_vals, rates, color=color1, marker='o', label='压缩率')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim(0, 100)
        ax1.grid(True, linestyle='--', alpha=0.5)

        # 右轴：PSNR
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.set_ylabel('PSNR (dB)', color=color2, fontsize=12, fontweight='bold')
        l2 = ax2.plot(q_vals, psnrs, color=color2, marker='s', linestyle='--', label='PSNR')
        ax2.tick_params(axis='y', labelcolor=color2)

        # 合并图例
        lines = l1 + l2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='center right')

        plt.title(f'图像压缩性能分析: {img_name}')
        plt.tight_layout()

        save_name = f"plot_{os.path.splitext(img_name)[0]}.png"
        plt.savefig(save_name)
        print(f"生成图表: {save_name}")
        # plt.show() # 如果不需要弹出窗口可注释掉
        plt.close()


if __name__ == "__main__":
    # 1. 运行实验生成数据
    csv_file = run_experiment()

    # 2. 读取数据自动画图
    if csv_file and os.path.exists(csv_file):
        plot_results(csv_file)