# main.py
import os
import sys
import pickle
import time
from huffman import HuffmanCompressor
import utils

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")
# 只处理 BMP 格式，这样对比最直观
SUPPORTED_EXTENSIONS = ('.bmp',)


# ===========================================

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_single_image(filename):
    input_path = os.path.join(INPUT_DIR, filename)
    name_without_ext = os.path.splitext(filename)[0]
    compressed_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}_compressed.bin")
    restored_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}_restored.jpg")

    print(f"\n[正在处理] {filename} ...")

    # --- 1. 压缩阶段 ---
    try:
        start_time = time.time()
        # 改用新的读取函数
        pixels, shape = utils.read_image(input_path)

        # 判断通道数 (彩色=3, 灰度=1)
        channels = 3 if len(shape) == 3 else 1

        compressor = HuffmanCompressor()
        compressed_bytes, frequency = compressor.compress_data(pixels)

        with open(compressed_path, 'wb') as f:
            # 写入图片尺寸信息
            f.write(int(shape[0]).to_bytes(4, byteorder='big'))  # Height
            f.write(int(shape[1]).to_bytes(4, byteorder='big'))  # Width
            f.write(int(channels).to_bytes(1, byteorder='big'))  # Channels (新增! 用1个字节存储)

            # 写入频率表
            freq_bytes = pickle.dumps(frequency)
            f.write(len(freq_bytes).to_bytes(4, byteorder='big'))
            f.write(freq_bytes)

            # 写入图像数据
            f.write(compressed_bytes)

        comp_time = time.time() - start_time
        print(f"  -> 压缩完成 ({comp_time:.4f}s)")

    except Exception as e:
        print(f"  -> 压缩失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. 统计阶段 ---
    size_origin = utils.get_file_size(input_path)
    size_comp = utils.get_file_size(compressed_path)
    if size_origin > 0:
        ratio = (1 - size_comp / size_origin) * 100
        print(f"  -> 原始: {size_origin / 1024:.2f}KB | 压缩后: {size_comp / 1024:.2f}KB | 节省: {ratio:.2f}%")

    # --- 3. 解压还原阶段 ---
    try:
        start_time = time.time()
        with open(compressed_path, 'rb') as f:
            height = int.from_bytes(f.read(4), byteorder='big')
            width = int.from_bytes(f.read(4), byteorder='big')
            channels = int.from_bytes(f.read(1), byteorder='big')  # 读取通道数

            freq_len = int.from_bytes(f.read(4), byteorder='big')
            frequency = pickle.loads(f.read(freq_len))
            byte_data = f.read()

        pixels = compressor.decompress_data(byte_data, frequency)

        # 重建形状元组
        if channels == 3:
            shape = (height, width, 3)
        else:
            shape = (height, width)

        utils.save_image_from_array(pixels, shape, restored_path)

        print(f"  -> 还原完成，已保存至 output 目录")

    except Exception as e:
        print(f"  -> 解压失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 请创建 {INPUT_DIR} 并放入图片")
        return
    ensure_dir(OUTPUT_DIR)
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not files:
        print("未找到图片文件")
        return

    print(f"=== 开始处理 {len(files)} 张图片 (支持彩色) ===")
    for f in files:
        process_single_image(f)


if __name__ == "__main__":
    main()