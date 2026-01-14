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
        return dict(Counter(pixels))

    def _make_heap(self, frequency):
        heap = []
        for key in frequency:
            node = HuffmanNode(key, frequency[key])
            heapq.heappush(heap, node)
        return heap

    def _merge_nodes(self, heap):
        while len(heap) > 1:
            node1 = heapq.heappop(heap)
            node2 = heapq.heappop(heap)
            merged = HuffmanNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2
            heapq.heappush(heap, merged)

    def _make_codes(self, root, current_code=""):
        if root is None:
            return
        if root.char is not None:
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return
        self._make_codes(root.left, current_code + "0")
        self._make_codes(root.right, current_code + "1")

    def _get_encoded_bytes(self, encoded_text):
        extra_padding = 8 - len(encoded_text) % 8
        encoded_text = encoded_text + "0" * extra_padding
        padded_info = "{0:08b}".format(extra_padding)
        encoded_text = padded_info + encoded_text

        b = bytearray()
        for i in range(0, len(encoded_text), 8):
            byte = encoded_text[i:i + 8]
            b.append(int(byte, 2))
        return b

    def compress_data(self, pixels, quantization_factor=1):
        """
        核心压缩逻辑 (已修复基准值丢失 Bug)
        """
        pixels = np.array(pixels, dtype=np.uint8)

        # --- 1. 标量量化 ---
        if quantization_factor > 1:
            pixels = (pixels // quantization_factor).astype(np.uint8)

        # --- 2. 差分编码 (修复点) ---
        # 错误写法: prepend=pixels[0] (会导致第一个像素归零，整图偏移)
        # 正确写法: prepend=0 (保存第一个像素的绝对值)
        # 注意: 0 需要转为与 pixels 相同的类型以防报错，但通常 int 0 也可以

        # 计算差分: Current - Previous
        # 使用 int16 中间态防止 uint8 减法时的意外溢出困扰（虽然 numpy diff 通常处理得好）
        # 但为了绝对安全，我们手动处理一下首位

        # 方案：使用 prepend=0，这样 delta[0] = pixels[0] - 0 = pixels[0]
        delta = np.diff(pixels, prepend=0)

        # 再次转回 uint8 (利用模运算特性存储负数差值)
        pixels_to_encode = delta.astype(np.uint8)

        # --- 3. 哈夫曼编码 ---
        frequency = self._make_frequency_dict(pixels_to_encode)
        heap = self._make_heap(frequency)
        self._merge_nodes(heap)

        root = heap[0]
        self._make_codes(root)

        encoded_text = "".join([self.codes[pixel] for pixel in pixels_to_encode])
        byte_data = self._get_encoded_bytes(encoded_text)

        return byte_data, frequency

    def decompress_data(self, byte_data, frequency, quantization_factor=1):
        """
        核心解压逻辑：
        1. 哈夫曼解码
        2. 反差分 (累加)
        3. 反量化 (恢复精度)
        """
        # 1. 重建哈夫曼树
        heap = self._make_heap(frequency)
        self._merge_nodes(heap)
        root = heap[0]
        self.codes = {}
        self.reverse_mapping = {}
        self._make_codes(root)

        # 2. 解析位流
        bit_string = []
        for byte in byte_data:
            bit_string.append(f"{byte:08b}")
        bit_string = "".join(bit_string)

        extra_padding = int(bit_string[:8], 2)
        bit_string = bit_string[8:]
        encoded_text = bit_string[:-extra_padding]

        decoded_deltas = []
        current_code = ""
        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                decoded_deltas.append(self.reverse_mapping[current_code])
                current_code = ""

        # 3. 反差分还原 (累加)
        delta_array = np.array(decoded_deltas, dtype=np.uint8)
        restored_pixels = np.cumsum(delta_array, dtype=np.uint8)

        # 4. [有损] 反量化 (Restoration)
        if quantization_factor > 1:
            # 核心：乘回因子。例如 15 * 10 = 150 (原始是155, 误差5)
            # 注意防止溢出，但在 uint8 图像中通常还好，或者先转 int16 乘完再转 uint8
            restored_pixels = restored_pixels.astype(np.float32) * quantization_factor
            # 截断到 0-255
            restored_pixels = np.clip(restored_pixels, 0, 255).astype(np.uint8)

        return restored_pixels