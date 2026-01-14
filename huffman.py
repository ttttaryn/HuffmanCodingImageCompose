# huffman.py
import heapq
import numpy as np
from collections import Counter


class HuffmanNode:

    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    def __lt__(self, other):
        return self.freq < other.freq


class HuffmanCompressor:
    def __init__(self):
        self.codes = {}
        self.reverse_mapping = {}

    def _make_frequency_dict(self, pixels):
        """统计频率"""
        # pixels 此时已经是经过差分处理的数据
        return dict(Counter(pixels))

    def _make_heap(self, frequency):
        """构建最小堆"""
        heap = []
        for key in frequency:
            node = HuffmanNode(key, frequency[key])
            heapq.heappush(heap, node)
        return heap

    def _merge_nodes(self, heap):
        """合并节点构建树"""
        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)

            merged = HuffmanNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(heap, merged)

    def _make_codes(self, root, current_code=""):
        """生成 0/1 编码表"""
        if root is None:
            return
        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return
        self._make_codes(root.left, current_code + "0")
        self._make_codes(root.right, current_code + "1")

    def _get_encoded_bytes(self, encoded_text):
        """将 01 字符串打包成字节流"""
        extra_padding = 8 - len(encoded_text) % 8
        encoded_text = encoded_text + "0" * extra_padding

        # 记录填充了多少位，放在最开头
        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text

        # 转换为字节数组
        b = bytearray()
        for i in range(0, len(encoded_text), 8):
            byte = encoded_text[i:i + 8]
            b.append(int(byte, 2))
        return b

    def compress_data(self, pixels):
        """
        核心压缩逻辑：
        1. 差分变换 (Delta Encoding) -> 优化数据分布
        2. 哈夫曼编码
        """
        # 1. 转为 numpy 数组，确保类型为 uint8
        pixels = np.array(pixels, dtype=np.uint8)

        # 2. 计算差分: Current - Previous
        # prepend=pixels[0] 使得第一个元素保持不变，作为还原的基准
        # astype(np.uint8) 自动处理负数溢出 (例如 -1 会变成 255)，这是合法的图像处理操作
        delta = np.diff(pixels, prepend=pixels[0])
        pixels_to_encode = delta.astype(np.uint8)

        # 3. 正常的哈夫曼流程
        frequency = self._make_frequency_dict(pixels_to_encode)
        heap = self._make_heap(frequency)
        self._merge_nodes(heap)

        root = heap[0]
        self._make_codes(root)

        # 生成编码
        encoded_text = "".join([self.codes[pixel] for pixel in pixels_to_encode])
        byte_data = self._get_encoded_bytes(encoded_text)

        return byte_data, frequency

    def decompress_data(self, byte_data, frequency):
        """
        核心解压逻辑：
        1. 哈夫曼解码 -> 得到差分值
        2. 反差分变换 (Inverse Delta) -> 还原原始像素
        """
        # 1. 重建哈夫曼树
        heap = self._make_heap(frequency)
        self._merge_nodes(heap)
        root = heap[0]
        self.codes = {}
        self.reverse_mapping = {}
        self._make_codes(root)

        # 2. 解析二进制流
        bit_string = []
        # 优化：大文件时 join 可能慢，但在 Python 中这是标准做法
        # 如果追求极致性能，这里可用 C++ 扩展，但作业级 Python 足够
        for byte in byte_data:
            bit_string.append(f"{byte:08b}")
        bit_string = "".join(bit_string)

        # 移除填充位
        extra_padding = int(bit_string[:8], 2)
        bit_string = bit_string[8:]
        encoded_text = bit_string[:-extra_padding]

        # 3. 解码得到差分数组
        decoded_deltas = []
        current_code = ""
        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                decoded_deltas.append(self.reverse_mapping[current_code])
                current_code = ""

        # 4. 反差分还原 (累加)
        # 逻辑：Current = (Previous + Delta) % 256
        # np.cumsum(dtype=np.uint8) 会自动处理溢出回绕，完美还原原始数据
        delta_array = np.array(decoded_deltas, dtype=np.uint8)
        restored_pixels = np.cumsum(delta_array, dtype=np.uint8)

        return restored_pixels