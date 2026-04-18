import numpy as np
import struct
from PIL import Image
import os


# ====================== ЗАПИСЬ ФАЙЛА ======================
def save_compressed_image(filename, width, height, quality,
                          quant_table_y, quant_table_c,
                          dc_huffman_codes, ac_huffman_codes_list,
                          color_space=1):
    with open(filename, 'wb') as f:
        f.write(b'XIMG_V1')  # сигнатура
        f.write(struct.pack('<B', color_space))
        f.write(struct.pack('<HH', width, height))
        f.write(struct.pack('<B', quality))

        block_h = (height + 7) // 8
        block_w = (width + 7) // 8
        f.write(struct.pack('<HH', block_h, block_w))

        # Таблицы квантования
        f.write(quant_table_y.astype(np.uint16).flatten().tobytes())
        f.write(quant_table_c.astype(np.uint16).flatten().tobytes())

        # DC коэффициенты
        f.write(struct.pack('<I', len(dc_huffman_codes)))
        for dc in dc_huffman_codes:
            bits = dc['full_bitstring']
            if bits:  # Проверяем, что строка не пустая
                f.write(struct.pack('<H', len(bits)))  # вместо '<B'
                f.write(int(bits, 2).to_bytes((len(bits) + 7) // 8, 'big'))
            else:
                # Если пустая строка, записываем 0 бит
                f.write(struct.pack('<B', 0))

        # AC коэффициенты
        f.write(struct.pack('<I', len(ac_huffman_codes_list)))
        for block in ac_huffman_codes_list:
            f.write(struct.pack('<H', len(block)))
            for ac in block:
                bits = ac['full_code']
                if bits:  # Проверяем, что строка не пустая
                    f.write(struct.pack('<H', len(bits)))
                    f.write(int(bits, 2).to_bytes((len(bits) + 7) // 8, 'big'))
                else:
                    # Если пустая строка, записываем 0 бит
                    f.write(struct.pack('<B', 0))

    print(f"Файл сохранён: {filename} ({os.path.getsize(filename)} байт)")

# ====================== ЧТЕНИЕ ФАЙЛА ======================
def load_compressed_image(filename):
    """Читает сжатый файл и возвращает основную информацию"""
    with open(filename, 'rb') as f:
        signature = f.read(7)
        if signature != b'XIMG_V1':
            raise ValueError("Неверная сигнатура файла!")

        color_space = struct.unpack('<B', f.read(1))[0]
        width = struct.unpack('<H', f.read(2))[0]
        height = struct.unpack('<H', f.read(2))[0]
        quality = struct.unpack('<B', f.read(1))[0]

        block_h = struct.unpack('<H', f.read(2))[0]
        block_w = struct.unpack('<H', f.read(2))[0]

        # Читаем таблицы квантования
        quant_y = np.frombuffer(f.read(128), dtype=np.uint16).reshape(8, 8)
        quant_c = np.frombuffer(f.read(128), dtype=np.uint16).reshape(8, 8)

        # Читаем DC
        num_dc = struct.unpack('<I', f.read(4))[0]
        dc_codes = []
        for _ in range(num_dc):
            bit_len = struct.unpack('<H', f.read(2))[0]  # вместо '<B', читаем 2 байта
            data = f.read((bit_len + 7) // 8)
            bitstring = bin(int.from_bytes(data, 'big'))[2:].zfill(bit_len)
            dc_codes.append(bitstring)

        # Читаем AC
        num_blocks = struct.unpack('<I', f.read(4))[0]
        ac_codes_list = []
        for _ in range(num_blocks):
            num_pairs = struct.unpack('<H', f.read(2))[0]
            block = []
            for __ in range(num_pairs):
                bit_len = struct.unpack('<H', f.read(2))[0]
                data = f.read((bit_len + 7) // 8)
                bitstring = bin(int.from_bytes(data, 'big'))[2:].zfill(bit_len)
                block.append(bitstring)
            ac_codes_list.append(block)

    print(f"Файл успешно прочитан: {width}x{height}, Quality={quality}")
    return {
        'width': width,
        'height': height,
        'quality': quality,
        'color_space': color_space,
        'quant_y': quant_y,
        'quant_c': quant_c,
        'dc_codes': dc_codes,
        'ac_codes_list': ac_codes_list
    }

def rgb_to_ycbcr(rgb_array):
    """Преобразует RGB изображение в YCbCr"""
    # rgb_array должен быть shape (height, width, 3) и тип uint8
    R = rgb_array[:, :, 0].astype(np.float32)
    G = rgb_array[:, :, 1].astype(np.float32)
    B = rgb_array[:, :, 2].astype(np.float32)

    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B
    Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B

    # Ограничиваем диапазон 0-255 и переводим обратно в uint8
    Y = np.clip(Y, 0, 255).astype(np.uint8)
    Cb = np.clip(Cb, 0, 255).astype(np.uint8)
    Cr = np.clip(Cr, 0, 255).astype(np.uint8)

    return np.stack([Y, Cb, Cr], axis=2)

def split_into_8x8_blocks(image_array):
    """
    Разбивает изображение (2D массив) на блоки 8x8

    Вход:  numpy массив размером (H, W) — обычно яркостный канал Y или grayscale
    Выход: numpy массив размером (num_blocks_h, num_blocks_w, 8, 8)
    """

    h, w = image_array.shape[:2]

    # Проверяем, кратны ли размеры 8
    if h % 8 != 0 or w % 8 != 0:
        print(f"Размеры {w}x{h} не кратны 8! Будет добавлен padding.")

        # Добавляем padding (дополняем до ближайшего кратного 8)
        new_h = ((h + 7) // 8) * 8
        new_w = ((w + 7) // 8) * 8

        padded = np.zeros((new_h, new_w), dtype=image_array.dtype)
        padded[:h, :w] = image_array
        image_array = padded
        print(f"   → Добавлен padding до размера {new_w}x{new_h}")

    h, w = image_array.shape

    # Разбиваем на блоки 8x8
    blocks = image_array.reshape(h // 8, 8, w // 8, 8)
    blocks = blocks.transpose(0, 2, 1, 3)  # меняем порядок осей

    print(f"Изображение разбито на {blocks.shape[0]}x{blocks.shape[1]} блоков по 8x8")
    print(f"Итоговый размер массива блоков: {blocks.shape}")

    return blocks

def zigzag_scan(square_matrix):
    """Зигзаг-обход квадратной матрицы NxN"""
    N = square_matrix.shape[0]
    result = []

    for sum_diag in range(2 * N - 1):  # сумма индексов по диагоналям
        diag = []
        for i in range(max(0, sum_diag - N + 1), min(sum_diag + 1, N)):
            j = sum_diag - i
            diag.append(square_matrix[i, j])

        # Чередуем направление: чётные диагонали — в одну сторону, нечётные — в другую
        if sum_diag % 2 == 0:
            result.extend(diag[::-1])  # разворачиваем
        else:
            result.extend(diag)

    return np.array(result)

def adjust_quantization_table(quant_table, quality):
    """
    Адаптирует таблицу квантования под заданный уровень качества

    quality: от 1 до 100
    """
    if quality < 1 or quality > 100:
        raise ValueError("Quality должен быть от 1 до 100")

    # Вычисляем масштабный коэффициент S
    if quality < 50:
        S = 5000 / quality
    else:
        S = 200 - 2 * quality

    # Создаём новую таблицу
    new_table = np.zeros_like(quant_table, dtype=np.int32)

    for i in range(8):
        for j in range(8):
            # Основная формула
            temp = (quant_table[i, j] * S) / 100
            new_table[i, j] = np.ceil(temp)  # округление вверх

            # Защита от нуля (нельзя делить на 0 при квантовании)
            if new_table[i, j] < 1:
                new_table[i, j] = 1

    return new_table


def print_quant_table(table, title=""):
    print(f"\n{title}")
    print("   ", end="")
    for j in range(8):
        print(f"{j:3}", end="")
    print()
    for i in range(8):
        print(f"{i:2} ", end="")
        for j in range(8):
            print(f"{table[i, j]:3}", end="")
        print()

# Матрица квантования для канала яркости Y ( luminance )
quant_table_y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

# Матрица квантования для цветовых каналов Cb и Cr (chrominance)
quant_table_c = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]
], dtype=np.float32)

def apply_dct_to_image(Y_channel):
    blocks = split_into_8x8_blocks(Y_channel)  # твоя функция из прошлого пункта
    h_blocks, w_blocks, _, _ = blocks.shape

    dct_blocks = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.float32)

    for i in range(h_blocks):
        for j in range(w_blocks):
            # Можно использовать любую из двух функций
            dct_blocks[i, j] = dct_8x8_primitive(blocks[i, j])
            # dct_blocks[i, j] = dct_2d_general(blocks[i, j])  # если хочешь общую

    print(f"DCT применён к {h_blocks}×{w_blocks} блокам 8x8")
    return dct_blocks

def dct_8x8_primitive(block):
    """Примитивное ДКП по формуле JPEG для блока 8x8"""
    block = block.astype(np.float32)
    N = 8
    dct = np.zeros((N, N), dtype=np.float32)

    for u in range(N):
        for v in range(N):
            sum_val = 0.0
            for x in range(N):
                for y in range(N):
                    cos1 = np.cos((2 * x + 1) * u * np.pi / (2 * N))
                    cos2 = np.cos((2 * y + 1) * v * np.pi / (2 * N))
                    sum_val += block[x, y] * cos1 * cos2

            cu = 1 / np.sqrt(2) if u == 0 else 1
            cv = 1 / np.sqrt(2) if v == 0 else 1

            dct[u, v] = 0.25 * cu * cv * sum_val

    return dct

def create_dct_matrix(N=8):
    """Создаёт матрицу DCT преобразования C размером NxN"""
    C = np.zeros((N, N), dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == 0:
                C[i, j] = np.sqrt(1.0 / N)
            else:
                C[i, j] = np.sqrt(2.0 / N) * np.cos((2 * j + 1) * i * np.pi / (2 * N))

    return C


def apply_dct_matrix_to_image(Y_channel):
    C = create_dct_matrix(8)
    blocks = split_into_8x8_blocks(Y_channel)

    h_blocks, w_blocks, _, _ = blocks.shape
    dct_blocks = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.float32)

    for i in range(h_blocks):
        for j in range(w_blocks):
            # Приводим к диапазону -128..127
            normalized_block = blocks[i, j].astype(np.float32) - 128.0
            dct_blocks[i, j] = dct_8x8_matrix(normalized_block, C)

    return dct_blocks, C
def dct_8x8_matrix(block, C):
    """Прямое DCT через матрицы: F = C * f * C^T"""
    block = block.astype(np.float32)
    # НЕ ДЕЛИМ НА 255, блок уже нормализован вычитанием 128
    temp = np.dot(C, block)
    dct_block = np.dot(temp, C.T)
    return dct_block



def apply_quantization(dct_blocks, quant_table):
    """Применяет квантование ко всем блокам изображения"""
    h_blocks, w_blocks, _, _ = dct_blocks.shape
    quantized_blocks = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.int32)

    for i in range(h_blocks):
        for j in range(w_blocks):
            quantized_blocks[i, j] = quantize_dct(dct_blocks[i, j], quant_table)

    print(f"Квантование выполнено для {h_blocks}x{w_blocks} блоков")
    return quantized_blocks

def quantize_dct(dct_block, quant_table):
    """
    Квантование коэффициентов DCT

    dct_block   - блок 8x8 после DCT (float)
    quant_table - матрица квантования 8x8
    """
    # Делим и округляем
    quantized = np.round(dct_block / quant_table)
    return quantized.astype(np.int32)  # обычно хранят как int


def rle_vlc_ac_encode(ac):
    """
    Кодирует AC коэффициенты JPEG с использованием RLE и VLC (переменной длины кода)

    Параметры:
        ac: numpy массив из 63 AC коэффициентов (после квантования и zigzag обхода)

    Возвращает:
        Список словарей, каждый словарь содержит:
            'run_length': количество нулей перед ненулевым коэффициентом (0-15)
            'category': категория коэффициента (размер в битах)
            'amplitude': амплитуда коэффициента (значение)
            'full_code': битовая строка кода Хаффмана + амплитуда
    """

    # Стандартные таблицы Хаффмана для AC коэффициентов JPEG (яркость)
    ac_huffman_table = {
        # (run_length, category): код Хаффмана
        (0, 1): "00",
        (0, 2): "01",
        (0, 3): "100",
        (0, 4): "1011",
        (0, 5): "11010",
        (0, 6): "1111000",
        (0, 7): "11111000",
        (0, 8): "1111110110",
        (0, 9): "1111111110000010",
        (0, 10): "1111111110000011",
        (0, 11): "1111111110000100",
        (0, 12): "1111111110000101",
        (0, 13): "1111111110000110",
        (0, 14): "1111111110000111",
        (0, 15): "1111111110001000",

        (1, 1): "1100",
        (1, 2): "11011",
        (1, 3): "1111001",
        (1, 4): "111110110",
        (1, 5): "11111110110",
        (1, 6): "1111111110001001",
        (1, 7): "1111111110001010",
        (1, 8): "1111111110001011",
        (1, 9): "1111111110001100",
        (1, 10): "1111111110001101",
        (1, 11): "1111111110001110",
        (1, 12): "1111111110001111",
        (1, 13): "1111111110010000",
        (1, 14): "1111111110010001",
        (1, 15): "1111111110010010",

        (2, 1): "11100",
        (2, 2): "11111001",
        (2, 3): "1111110111",
        (2, 4): "111111110100",
        (2, 5): "1111111110010011",
        (2, 6): "1111111110010100",
        (2, 7): "1111111110010101",
        (2, 8): "1111111110010110",
        (2, 9): "1111111110010111",
        (2, 10): "1111111110011000",
        (2, 11): "1111111110011001",
        (2, 12): "1111111110011010",
        (2, 13): "1111111110011011",
        (2, 14): "1111111110011100",
        (2, 15): "1111111110011101",

        (3, 1): "111010",
        (3, 2): "111110111",
        (3, 3): "111111110101",
        (3, 4): "1111111110011110",
        (3, 5): "1111111110011111",
        (3, 6): "1111111110100000",
        (3, 7): "1111111110100001",
        (3, 8): "1111111110100010",
        (3, 9): "1111111110100011",
        (3, 10): "1111111110100100",
        (3, 11): "1111111110100101",
        (3, 12): "1111111110100110",
        (3, 13): "1111111110100111",
        (3, 14): "1111111110101000",
        (3, 15): "1111111110101001",

        (4, 1): "111011",
        (4, 2): "1111111000",
        (4, 3): "1111111110101010",
        (4, 4): "1111111110101011",
        (4, 5): "1111111110101100",
        (4, 6): "1111111110101101",
        (4, 7): "1111111110101110",
        (4, 8): "1111111110101111",
        (4, 9): "1111111110110000",
        (4, 10): "1111111110110001",
        (4, 11): "1111111110110010",
        (4, 12): "1111111110110011",
        (4, 13): "1111111110110100",
        (4, 14): "1111111110110101",
        (4, 15): "1111111110110110",

        (5, 1): "1111010",
        (5, 2): "11111110111",
        (5, 3): "1111111110110111",
        (5, 4): "1111111110111000",
        (5, 5): "1111111110111001",
        (5, 6): "1111111110111010",
        (5, 7): "1111111110111011",
        (5, 8): "1111111110111100",
        (5, 9): "1111111110111101",
        (5, 10): "1111111110111110",
        (5, 11): "1111111110111111",
        (5, 12): "1111111111000000",
        (5, 13): "1111111111000001",
        (5, 14): "1111111111000010",
        (5, 15): "1111111111000011",

        (6, 1): "1111011",
        (6, 2): "111111110110",
        (6, 3): "1111111111000100",
        (6, 4): "1111111111000101",
        (6, 5): "1111111111000110",
        (6, 6): "1111111111000111",
        (6, 7): "1111111111001000",
        (6, 8): "1111111111001001",
        (6, 9): "1111111111001010",
        (6, 10): "1111111111001011",
        (6, 11): "1111111111001100",
        (6, 12): "1111111111001101",
        (6, 13): "1111111111001110",
        (6, 14): "1111111111001111",
        (6, 15): "1111111111010000",

        (7, 1): "11111010",
        (7, 2): "111111110111",
        (7, 3): "1111111111010001",
        (7, 4): "1111111111010010",
        (7, 5): "1111111111010011",
        (7, 6): "1111111111010100",
        (7, 7): "1111111111010101",
        (7, 8): "1111111111010110",
        (7, 9): "1111111111010111",
        (7, 10): "1111111111011000",
        (7, 11): "1111111111011001",
        (7, 12): "1111111111011010",
        (7, 13): "1111111111011011",
        (7, 14): "1111111111011100",
        (7, 15): "1111111111011101",

        (8, 1): "111111000",
        (8, 2): "111111111000000",
        (8, 3): "1111111111011110",
        (8, 4): "1111111111011111",
        (8, 5): "1111111111100000",
        (8, 6): "1111111111100001",
        (8, 7): "1111111111100010",
        (8, 8): "1111111111100011",
        (8, 9): "1111111111100100",
        (8, 10): "1111111111100101",
        (8, 11): "1111111111100110",
        (8, 12): "1111111111100111",
        (8, 13): "1111111111101000",
        (8, 14): "1111111111101001",
        (8, 15): "1111111111101010",

        (9, 1): "111111001",
        (9, 2): "1111111110111110",
        (9, 3): "1111111111101011",
        (9, 4): "1111111111101100",
        (9, 5): "1111111111101101",
        (9, 6): "1111111111101110",
        (9, 7): "1111111111101111",
        (9, 8): "1111111111110000",
        (9, 9): "1111111111110001",
        (9, 10): "1111111111110010",
        (9, 11): "1111111111110011",
        (9, 12): "1111111111110100",
        (9, 13): "1111111111110101",
        (9, 14): "1111111111110110",
        (9, 15): "1111111111110111",

        (10, 1): "111111010",
        (10, 2): "1111111111000000",
        (10, 3): "1111111111111000",
        (10, 4): "1111111111111001",
        (10, 5): "1111111111111010",
        (10, 6): "1111111111111011",
        (10, 7): "1111111111111100",
        (10, 8): "1111111111111101",
        (10, 9): "1111111111111110",
        (10, 10): "1111111111111111",
        (10, 11): "11111111100000000",
        (10, 12): "11111111100000001",
        (10, 13): "11111111100000010",
        (10, 14): "11111111100000011",
        (10, 15): "11111111100000100",

        (11, 1): "1111111001",
        (11, 2): "1111111111110000",
        (11, 3): "11111111100000101",
        (11, 4): "11111111100000110",
        (11, 5): "11111111100000111",
        (11, 6): "11111111100001000",
        (11, 7): "11111111100001001",
        (11, 8): "11111111100001010",
        (11, 9): "11111111100001011",
        (11, 10): "11111111100001100",
        (11, 11): "11111111100001101",
        (11, 12): "11111111100001110",
        (11, 13): "11111111100001111",
        (11, 14): "11111111100010000",
        (11, 15): "11111111100010001",

        (12, 1): "1111111010",
        (12, 2): "1111111110011110",
        (12, 3): "11111111100010010",
        (12, 4): "11111111100010011",
        (12, 5): "11111111100010100",
        (12, 6): "11111111100010101",
        (12, 7): "11111111100010110",
        (12, 8): "11111111100010111",
        (12, 9): "11111111100011000",
        (12, 10): "11111111100011001",
        (12, 11): "11111111100011010",
        (12, 12): "11111111100011011",
        (12, 13): "11111111100011100",
        (12, 14): "11111111100011101",
        (12, 15): "11111111100011110",

        (13, 1): "11111111000",
        (13, 2): "1111111110100000",
        (13, 3): "11111111100011111",
        (13, 4): "11111111100100000",
        (13, 5): "11111111100100001",
        (13, 6): "11111111100100010",
        (13, 7): "11111111100100011",
        (13, 8): "11111111100100100",
        (13, 9): "11111111100100101",
        (13, 10): "11111111100100110",
        (13, 11): "11111111100100111",
        (13, 12): "11111111100101000",
        (13, 13): "11111111100101001",
        (13, 14): "11111111100101010",
        (13, 15): "11111111100101011",

        (14, 1): "111111111110010",
        (14, 2): "1111111110100110",
        (14, 3): "11111111100101100",
        (14, 4): "11111111100101101",
        (14, 5): "11111111100101110",
        (14, 6): "11111111100101111",
        (14, 7): "11111111100110000",
        (14, 8): "11111111100110001",
        (14, 9): "11111111100110010",
        (14, 10): "11111111100110011",
        (14, 11): "11111111100110100",
        (14, 12): "11111111100110101",
        (14, 13): "11111111100110110",
        (14, 14): "11111111100110111",
        (14, 15): "11111111100111000",

        (15, 0): "11111111001"  # EOB (конец блока)
    }

    def get_category(value):
        """Определяет категорию коэффициента (размер в битах)"""
        # Преобразуем numpy.int32 в обычный int
        val = int(value)
        abs_val = abs(val)
        if abs_val == 0:
            return 0
        return abs_val.bit_length()

    def amplitude_to_bits(amplitude):
        """Преобразует амплитуду в битовую строку (дополнительный код)"""
        amp = int(amplitude)
        if amp == 0:
            return ""

        category = get_category(amp)
        if amp > 0:
            # Положительное число
            return bin(amp)[2:].zfill(category)
        else:
            # Отрицательное число - дополнительный код
            return bin((1 << category) + amp)[2:]
    encoded_pairs = []

    # RLE кодирование
    i = 0
    while i < len(ac):
        if ac[i] == 0:
            # Считаем количество нулей
            zero_count = 0
            while i < len(ac) and ac[i] == 0 and zero_count < 16:
                zero_count += 1
                i += 1

            if i >= len(ac):
                # Достигли конца блока, добавляем EOB
                encoded_pairs.append({
                    'run_length': 0,
                    'category': 0,
                    'amplitude': 0,
                    'full_code': ac_huffman_table.get((15, 0), "11111111001")
                })
                break

            # Есть ненулевой коэффициент после нулей
            amplitude = ac[i]
            category = get_category(amplitude)

            # Если нулей 16 или больше, добавляем ZRL (16 нулей)
            while zero_count >= 16:
                encoded_pairs.append({
                    'run_length': 15,
                    'category': 0,
                    'amplitude': 0,
                    'full_code': "11111111001"  # ZRL код
                })
                zero_count -= 16

            # Кодируем пару (run_length, category)
            run = min(zero_count, 15)
            huff_code = ac_huffman_table.get((run, category), "")
            amp_bits = amplitude_to_bits(amplitude)

            encoded_pairs.append({
                'run_length': run,
                'category': category,
                'amplitude': int(amplitude),
                'full_code': huff_code + amp_bits
            })
            i += 1
        else:
            # Ненулевой коэффициент без предшествующих нулей
            amplitude = ac[i]
            category = get_category(amplitude)
            huff_code = ac_huffman_table.get((0, category), "")
            amp_bits = amplitude_to_bits(amplitude)

            encoded_pairs.append({
                'run_length': 0,
                'category': category,
                'amplitude': int(amplitude),
                'full_code': huff_code + amp_bits
            })
            i += 1

    return encoded_pairs


def rle_vlc_dc_encode(dc_values):
    # Стандартная таблица Хаффмана для DC (яркость)
    dc_huffman_table = {
        0: "00",
        1: "010",
        2: "011",
        3: "100",
        4: "101",
        5: "110",
        6: "1110",
        7: "11110",
        8: "111110",
        9: "1111110",
        10: "11111110",
        11: "111111110"
    }

    def get_category(value):
        val = int(value)
        abs_val = abs(val)
        if abs_val == 0:
            return 0
        return abs_val.bit_length()

    def value_to_bits(value, category):
        val = int(value)
        if val >= 0:
            return bin(val)[2:].zfill(category)
        else:
            return bin((1 << category) + val)[2:]

    encoded_dc = []
    prev_dc = 0

    for idx, dc in enumerate(dc_values):
        diff = int(dc) - prev_dc
        category = get_category(diff)

        if idx < 5:  # Отладка
            print(f"  DC[{idx}]: dc={dc}, prev={prev_dc}, diff={diff}, category={category}")

        if category == 0:
            huff_code = dc_huffman_table.get(0, "00")  # Убедись, что есть код
            encoded_dc.append({
                'category': 0,
                'amplitude': diff,
                'full_bitstring': huff_code
            })
        else:
            huff_code = dc_huffman_table.get(category, "")
            if not huff_code:  # Защита от отсутствия кода
                huff_code = "0" * category
            bits = value_to_bits(diff, category)
            encoded_dc.append({
                'category': category,
                'amplitude': diff,
                'full_bitstring': huff_code + bits
            })

        prev_dc = diff

    return encoded_dc

def decompress_image(compressed_filename, output_filename=None):
    """
    Полная декомпрессия изображения из формата XIMG_V1 (raw версия)
    """
    print(f"\n=== Декомпрессия файла {compressed_filename} ===\n")

    # Загружаем данные
    img_data = load_compressed_image(compressed_filename)

    width = img_data['width']
    height = img_data['height']
    quality = img_data['quality']
    quant_y = img_data['quant_y']
    dc_codes = img_data['dc_codes']
    ac_codes_list = img_data['ac_codes_list']

    print(f"Размер изображения: {width}x{height}")
    print(f"Quality: {quality}")

    # RAW декодирование DC
    def raw_dc_decode(dc_codes):
        dc_values = []
        prev_dc = 0
        for code in dc_codes:
            if len(code) >= 16:
                val = int(code[:16], 2)
                if val & 0x8000:
                    val = val - 65536
                dc_values.append(val)
            else:
                dc_values.append(0)
        return dc_values

    # RAW декодирование AC
    def raw_ac_decode(ac_codes_list):
        all_ac = []
        for ac_codes in ac_codes_list:
            full_bits = ''.join(ac_codes)
            ac_vals = []
            for i in range(0, len(full_bits), 8):
                if i+8 <= len(full_bits):
                    byte = int(full_bits[i:i+8], 2)
                    if byte == 255:
                        break
                    if byte & 0x80:
                        byte = byte - 256
                    ac_vals.append(byte)
            while len(ac_vals) < 63:
                ac_vals.append(0)
            all_ac.append(np.array(ac_vals[:63]))
        return all_ac

    # Декодируем DC
    print("Декодирование DC коэффициентов (raw)...")
    dc_values = raw_dc_decode(dc_codes)
    print(f"  Первые 5 DC: {dc_values[:5]}")

    # Декодируем AC
    print("Декодирование AC коэффициентов (raw)...")
    ac_values_list = raw_ac_decode(ac_codes_list)
    if ac_values_list:
        print(f"  Первый блок AC (первые 10): {ac_values_list[0][:10]}")

    # Восстанавливаем блоки
    block_h = (height + 7) // 8
    block_w = (width + 7) // 8
    blocks = np.zeros((block_h, block_w, 8, 8), dtype=np.float32)

    print(f"Восстановление блоков {block_h}x{block_w}...")
    for idx, (dc, ac) in enumerate(zip(dc_values, ac_values_list)):
        i = idx // block_w
        j = idx % block_w
        full = np.zeros(64, dtype=np.float32)
        full[0] = dc
        full[1:] = ac
        blocks[i, j] = inverse_zigzag_scan(full, 8)

    # Обратное квантование
    print("Обратное квантование...")
    dequantized = np.zeros((block_h, block_w, 8, 8), dtype=np.float32)
    for i in range(block_h):
        for j in range(block_w):
            dequantized[i, j] = blocks[i, j] * quant_y

    # Обратное DCT
    print("Обратное DCT...")
    C = create_dct_matrix(8)
    idct_blocks = np.zeros((block_h, block_w, 8, 8), dtype=np.float32)

    for i in range(block_h):
        for j in range(block_w):
            temp = np.dot(C.T, dequantized[i, j])
            idct_blocks[i, j] = np.dot(temp, C)
            # Добавляем 128 обратно (компенсируем вычитание при сжатии)
            idct_blocks[i, j] += 128.0

    # Собираем изображение
    print("Сборка изображения...")
    h = block_h * 8
    w = block_w * 8
    Y_channel = np.zeros((h, w), dtype=np.float32)

    for i in range(block_h):
        for j in range(block_w):
            Y_channel[i*8:(i+1)*8, j*8:(j+1)*8] = idct_blocks[i, j]

    Y_channel = Y_channel[:height, :width]
    Y_channel = np.clip(Y_channel, 0, 255).astype(np.uint8)

    print(f"Диапазон после клиппинга: min={Y_channel.min()}, max={Y_channel.max()}")

    if output_filename:
        img = Image.fromarray(Y_channel, mode='L')
        img.save(output_filename)
        print(f"Изображение сохранено как {output_filename}")

    return Y_channel

def decode_dc_coefficients(dc_codes):
    """Декодирует DC коэффициенты из кодов Хаффмана"""
    dc_huffman_reverse = {
        "00": 0, "010": 1, "011": 2, "100": 3, "101": 4,
        "110": 5, "1110": 6, "11110": 7, "111110": 8,
        "1111110": 9, "11111110": 10, "111111110": 11
    }

    def decode_single_dc(bitstring):
        if not bitstring:
            return 0

        huff_code = ""
        category = None
        pos = 0
        for i, bit in enumerate(bitstring):
            huff_code += bit
            if huff_code in dc_huffman_reverse:
                category = dc_huffman_reverse[huff_code]
                pos = i + 1
                break

        if category is None or category == 0:
            return 0

        if len(bitstring) < pos + category:
            return 0

        amp_bits = bitstring[pos:pos + category]
        amp = int(amp_bits, 2)
        # Дополнительный код: если amp < 2^(category-1) -> положительное, иначе отрицательное
        if amp < (1 << (category - 1)):
            result = amp
        else:
            result = amp - (1 << category)

    dc_values = []
    prev_dc = 0

    for idx, dc_code in enumerate(dc_codes):
        diff = decode_single_dc(dc_code)
        dc_value = prev_dc + diff

        # Отладка
        if idx < 5:
            print(f"  DC[{idx}]: code={dc_code[:30]}..., diff={diff}, value={dc_value}")

        dc_values.append(dc_value)
        prev_dc = dc_value

    return dc_values

def decode_ac_coefficients(ac_codes_list):
    """Декодирует AC коэффициенты из кодов Хаффмана"""
    ac_huffman_reverse = {
        "00": (0, 1), "01": (0, 2), "100": (0, 3), "1011": (0, 4),
        "11010": (0, 5), "1111000": (0, 6), "11111000": (0, 7),
        "1111110110": (0, 8), "1100": (1, 1), "11011": (1, 2),
        "1111001": (1, 3), "111110110": (1, 4), "11111110110": (1, 5),
        "11100": (2, 1), "11111001": (2, 2), "1111110111": (2, 3),
        "111111110100": (2, 4), "111010": (3, 1), "111110111": (3, 2),
        "111111110101": (3, 3), "111011": (4, 1), "1111111000": (4, 2),
        "1111010": (5, 1), "11111110111": (5, 2), "1111011": (6, 1),
        "111111110110": (6, 2), "11111010": (7, 1), "111111110111": (7, 2),
        "111111000": (8, 1), "111111111000000": (8, 2), "111111001": (9, 1),
        "1111111110111110": (9, 2), "111111010": (10, 1), "1111111111000000": (10, 2),
        "1111111001": (11, 1), "1111111111110000": (11, 2), "1111111010": (12, 1),
        "1111111110011110": (12, 2), "11111111000": (13, 1), "1111111110100000": (13, 2),
        "111111111110010": (14, 1), "1111111110100110": (14, 2), "11111111001": (15, 0),
    }

    def decode_amplitude(bitstring, category):
        if category == 0 or len(bitstring) < category:
            return 0
        amp_bits = bitstring[:category]
        amp = int(amp_bits, 2)
        if amp < (1 << (category - 1)):
            return amp
        else:
            return amp - (1 << category)

    all_ac_values = []

    for block_idx, ac_codes in enumerate(ac_codes_list):
        ac_values = []
        full_bitstring = ''.join(ac_codes)
        pos = 0
        bit_len = len(full_bitstring)

        while pos < bit_len and len(ac_values) < 63:
            found = False
            for huff_len in range(1, min(25, bit_len - pos + 1)):
                huff_code = full_bitstring[pos:pos + huff_len]
                if huff_code in ac_huffman_reverse:
                    run, category = ac_huffman_reverse[huff_code]
                    pos += huff_len
                    found = True

                    if category == 0:
                        while len(ac_values) < 63:
                            ac_values.append(0)
                        break

                    if pos + category <= bit_len:
                        amplitude = decode_amplitude(full_bitstring[pos:pos + category], category)
                        pos += category
                        ac_values.extend([0] * run)
                        ac_values.append(amplitude)
                    break

            if not found:
                break

        while len(ac_values) < 63:
            ac_values.append(0)

        all_ac_values.append(np.array(ac_values[:63]))

    return all_ac_values


def inverse_zigzag_scan(zigzag_array, N=8):
    """Обратный зигзаг-обход"""
    matrix = np.zeros((N, N), dtype=np.int32)

    # Правильный порядок индексов для зигзага
    zigzag_indices = []
    for s in range(2 * N - 1):
        if s % 2 == 0:  # четная диагональ - снизу вверх
            for i in range(min(s, N - 1), max(-1, s - N), -1):
                j = s - i
                if 0 <= i < N and 0 <= j < N:
                    zigzag_indices.append((i, j))
        else:  # нечетная диагональ - сверху вниз
            for i in range(max(0, s - N + 1), min(s + 1, N)):
                j = s - i
                if 0 <= i < N and 0 <= j < N:
                    zigzag_indices.append((i, j))

    # Заполняем матрицу
    for idx, val in enumerate(zigzag_array):
        if idx < len(zigzag_indices):
            i, j = zigzag_indices[idx]
            matrix[i, j] = val

    return matrix

    # ========== RAW КОДИРОВАНИЕ (без Хаффмана) ==========
def raw_dc_encode(dc_values):
    """Сохраняет DC значения как 16-битные числа"""
    encoded = []
    for dc in dc_values:
        val = int(dc)
        bits = bin(val & 0xFFFF)[2:].zfill(16)
        encoded.append({'category': 0, 'amplitude': val, 'full_bitstring': bits})
    return encoded

def raw_ac_encode(ac):
    """Сохраняет AC коэффициенты как 8-битные числа, добавляет маркер 255 в конце"""
    encoded_bytes = []
    for val in ac:
        byte = int(val) & 0xFF
        encoded_bytes.append(bin(byte)[2:].zfill(8))
    encoded_bytes.append('11111111')  # маркер конца блока
    return [{'run_length': 0, 'category': 0, 'amplitude': 0, 'full_code': ''.join(encoded_bytes)}]

if __name__ == "__main__":
    print("=== Финальное сохранение сжатого изображения ===\n")

    # 1. Подготовка всех нужных данных
    lena = Image.open('lenna.png').convert('RGB')
    rgb = np.array(lena)
    ycbcr = rgb_to_ycbcr(rgb)
    #Y = ycbcr[:, :, 0].astype(np.float32)  # Берем только яркость для теста
    Y = ycbcr[:, :, 0].astype(np.float32) - 128.0 # Вычитаем 128 для центрирования

    # Таблицы квантования (адаптированные под Quality)
    quality = 75
    quant_y = adjust_quantization_table(quant_table_y, quality)
    quant_c = adjust_quantization_table(quant_table_c, quality)

    # Разбиваем на блоки
    blocks = split_into_8x8_blocks(Y)

    # Применяем DCT (используем правильную версию)
    dct_blocks, dct_matrix = apply_dct_matrix_to_image(Y)

    # Применяем квантование
    quantized_blocks = apply_quantization(dct_blocks, quant_y)

    print(f"Форма quantized_blocks: {quantized_blocks.shape}")
    print(f"Всего блоков: {quantized_blocks.shape[0] * quantized_blocks.shape[1]}")

    # Кодирование
    dc_values = []
    ac_huffman_codes_list = []

    for i in range(quantized_blocks.shape[0]):
        for j in range(quantized_blocks.shape[1]):
            block = quantized_blocks[i, j]
            zig = zigzag_scan(block)
            dc = zig[0]
            ac = zig[1:]

            dc_values.append(dc)
            # ac_encoded = rle_vlc_ac_encode(ac)   <-- заменить на:
            ac_encoded = raw_ac_encode(ac)
            ac_huffman_codes_list.append(ac_encoded)

            if (i * quantized_blocks.shape[1] + j) % 500 == 0:
                print(f"Обработано блоков: {i * quantized_blocks.shape[1] + j}")

    # Кодируем DC
    print(f"\nПервые 5 DC значений ДО кодирования: {dc_values[:5]}")
    dc_huffman_codes = raw_dc_encode(dc_values)
    # Кодируем DC


    print(f"\nВсего DC кодов: {len(dc_huffman_codes)}")
    print(f"Всего AC блоков: {len(ac_huffman_codes_list)}")

    # 2. Сохранение
    save_compressed_image(
        filename="lena_compressed.ximg",
        width=512,
        height=512,
        quality=quality,
        quant_table_y=quant_y,
        quant_table_c=quant_c,
        dc_huffman_codes=dc_huffman_codes,
        ac_huffman_codes_list=ac_huffman_codes_list,
        color_space=1
    )