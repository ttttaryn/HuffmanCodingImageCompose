# utils.py
import os
import numpy as np
from PIL import Image


def read_image(path):
    """
    读取图片 -> 分离通道 (Planar) -> 展平
    目的是为了让差分编码更高效 (计算 R-R, G-G, B-B 而不是 R-G)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"文件未找到: {path}")

    # 1. 读取图片
    img = Image.open(path)
    # 转换为 numpy 数组 (H, W, C)
    img_array = np.array(img)
    shape = img_array.shape

    # 2. 通道分离处理
    if len(shape) == 3:  # 彩色图片 (H, W, 3)
        # 转换为 (3, H, W)，即把 RGB 分开存
        # transpose(2, 0, 1) 表示把第2维(通道)移到最前，然后是第0维(高)，第1维(宽)
        planar_array = img_array.transpose(2, 0, 1)
        return planar_array.flatten(), shape
    else:
        # 灰度图片不需要分离
        return img_array.flatten(), shape


def save_image_from_array(pixel_array, shape, output_path):
    """
    像素数组 -> 还原通道结构 -> 保存图片
    """
    # 确保数据类型
    pixel_array = np.array(pixel_array, dtype=np.uint8)

    if len(shape) == 3:  # 彩色图片还原
        # 1. 此时数组是 (3, H, W) 结构的扁平流，先reshape回去
        channels, height, width = shape[2], shape[0], shape[1]
        try:
            # 注意：这里要 reshape 成 (3, H, W)
            img_planar = pixel_array.reshape((channels, height, width))
            # 2. 转置回 (H, W, 3) 以便 PIL 处理
            img_data = img_planar.transpose(1, 2, 0)
            mode = 'RGB'
        except ValueError:
            print(f"错误: 数据长度不匹配，无法还原为 {shape}")
            return
    else:  # 灰度图片
        img_data = pixel_array.reshape(shape)
        mode = 'L'

    # 保存
    img = Image.fromarray(img_data, mode=mode)
    img.save(output_path)


def get_file_size(path):
    """获取文件大小"""
    if os.path.exists(path):
        return os.path.getsize(path)
    return 0