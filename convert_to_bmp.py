# convert_to_bmp.py
import os
from PIL import Image

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.path.join(BASE_DIR, "data", "input")


def convert_all_to_bmp():
    print(f"正在将 {INPUT_DIR} 下的图片转换为 BMP 格式...")

    files = os.listdir(INPUT_DIR)
    count = 0

    for filename in files:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            name, ext = os.path.splitext(filename)
            src_path = os.path.join(INPUT_DIR, filename)
            dst_path = os.path.join(INPUT_DIR, f"{name}.bmp")

            try:
                # 打开图片并保存为 BMP
                img = Image.open(src_path).convert('RGB')
                img.save(dst_path, "BMP")
                print(f"  [转换成功] {filename} -> {name}.bmp")

                os.remove(src_path)
                count += 1
            except Exception as e:
                print(f"  [失败] {filename}: {e}")

    print(f"转换完成，共生成 {count} 张 BMP 图片。")


if __name__ == "__main__":
    convert_all_to_bmp()