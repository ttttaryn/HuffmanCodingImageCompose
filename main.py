# main.py
import os
import pickle
import time
from huffman import HuffmanCompressor
import utils

# ================= 配置区域 =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "output")

# ★★★ 量化因子 (Q-Factor) 设置 ★★★
# Q = 1  : 无损压缩 (Lossless)，画质完美，压缩率较低
# Q = 5、10 : 轻微有损，画质肉眼难辨，压缩率显著提升 (推荐)
# Q = 20 : 明显有损，有色块/马赛克，压缩率极高
Q_FACTOR = 10

# 支持的图片格式 (建议使用 bmp 测试以获得最准确的压缩率对比)
SUPPORTED_EXTENSIONS = ('.bmp', )


# ===========================================

def ensure_dir(directory):
    """确保输出目录存在"""
    if not os.path.exists(directory):
        os.makedirs(directory)


def process_single_image(filename):
    input_path = os.path.join(INPUT_DIR, filename)

    # 构造输出文件名，文件名包含 Q 值方便区分
    name_without_ext = os.path.splitext(filename)[0]
    compressed_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}_q{Q_FACTOR}.bin")
    restored_path = os.path.join(OUTPUT_DIR, f"{name_without_ext}_q{Q_FACTOR}_restored.jpg")

    print(f"\n[正在处理] {filename} (Q={Q_FACTOR}) ...")

    compressor = HuffmanCompressor()

    # --- 1. 压缩阶段 ---
    try:
        start_time = time.time()

        # 读取图片 (自动处理通道分离)
        pixels, shape = utils.read_image(input_path)

        # 判断通道数 (彩色=3, 灰度=1)
        channels = 3 if len(shape) == 3 else 1

        # ★ 核心：执行压缩 (传入量化因子)
        compressed_bytes, frequency = compressor.compress_data(pixels, quantization_factor=Q_FACTOR)

        # 写入二进制文件
        with open(compressed_path, 'wb') as f:
            # [文件头] 13字节元数据
            f.write(int(shape[0]).to_bytes(4, byteorder='big'))  # Height (4 bytes)
            f.write(int(shape[1]).to_bytes(4, byteorder='big'))  # Width  (4 bytes)
            f.write(int(channels).to_bytes(1, byteorder='big'))  # Channels (1 byte)
            f.write(int(Q_FACTOR).to_bytes(4, byteorder='big'))  # Q_Factor (4 bytes) - 存下来以便解压时知道乘多少

            # [频率表]
            freq_bytes = pickle.dumps(frequency)
            f.write(len(freq_bytes).to_bytes(4, byteorder='big'))  # 频率表长度
            f.write(freq_bytes)  # 频率表内容

            # [压缩数据]
            f.write(compressed_bytes)

        comp_time = time.time() - start_time
        print(f"  -> 压缩完成 ({comp_time:.4f}s)")

    except Exception as e:
        print(f"  -> 压缩失败: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- 2. 统计分析阶段 ---
    file_size_on_disk = utils.get_file_size(input_path)  # 原图文件大小 (如果是PNG可能很小)
    huffman_size = utils.get_file_size(compressed_path)  # 我们压缩后的 bin 大小
    raw_pixel_size = len(pixels)  # 内存中原始像素大小 (最真实的基准)

    print(f"  [数据统计]")
    print(f"  -> 内存原始数据 (Raw): {raw_pixel_size / 1024:.2f} KB")
    print(f"  -> 压缩后文件 (Bin):   {huffman_size / 1024:.2f} KB")

    # 计算真实压缩率 (相对于 Raw Data)
    if raw_pixel_size > 0:
        ratio = (1 - huffman_size / raw_pixel_size) * 100
        print(f"  -> ★ 真实压缩率 (vs Raw): {ratio:.2f}%")

    # 计算相对于磁盘文件的比例 (如果原图是 PNG，这个可能是负数，正常现象)
    disk_ratio = (1 - huffman_size / file_size_on_disk) * 100
    print(f"  -> (参考) 文件压缩率:     {disk_ratio:.2f}%")

    # --- 3. 解压还原阶段 (验证 + 画质计算) ---
    try:
        start_time = time.time()
        with open(compressed_path, 'rb') as f:
            # 读取文件头
            height = int.from_bytes(f.read(4), byteorder='big')
            width = int.from_bytes(f.read(4), byteorder='big')
            channels = int.from_bytes(f.read(1), byteorder='big')
            q_factor_read = int.from_bytes(f.read(4), byteorder='big')  # 读取存入的 Q 值

            # 读取频率表
            freq_len = int.from_bytes(f.read(4), byteorder='big')
            frequency = pickle.loads(f.read(freq_len))

            # 读取数据主体
            byte_data = f.read()

        # ★ 核心：执行解压 (使用读取到的 Q 值进行反量化)
        restored_pixels = compressor.decompress_data(byte_data, frequency, quantization_factor=q_factor_read)

        # 还原形状
        if channels == 3:
            shape_to_restore = (height, width, 3)
        else:
            shape_to_restore = (height, width)

        # 保存还原图片
        utils.save_image_from_array(restored_pixels, shape_to_restore, restored_path)

        # ★ 计算 PSNR (峰值信噪比)
        # 只有当像素数量一致时才计算
        if len(pixels) == len(restored_pixels):
            psnr_value = utils.calculate_psnr(pixels, restored_pixels)
            print(f"  -> 还原完成 (使用 Q={q_factor_read})")
            print(f"  -> ★ PSNR (画质指标): {psnr_value:.2f} dB")
        else:
            print("  -> 警告: 像素数量不匹配，跳过 PSNR 计算")

    except Exception as e:
        print(f"  -> 解压/验证失败: {e}")
        import traceback
        traceback.print_exc()


def main():
    # 检查目录
    if not os.path.exists(INPUT_DIR):
        print(f"错误: 输入目录不存在 -> {INPUT_DIR}")
        print("请手动创建文件夹，并放入 .bmp 或 .jpg 图片")
        return

    ensure_dir(OUTPUT_DIR)

    # 获取图片列表
    files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(SUPPORTED_EXTENSIONS)]

    if not files:
        print(f"警告: {INPUT_DIR} 为空。请放入图片。")
        return

    print(f"=== 开始批量处理 {len(files)} 张图片 | 当前量化因子 Q={Q_FACTOR} ===")

    total_start = time.time()
    for f in files:
        process_single_image(f)

    print(f"\n=== 全部完成! 总耗时: {time.time() - total_start:.2f}s ===")
    print(f"结果已保存至: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()