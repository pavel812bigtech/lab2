from PIL import Image
import numpy as np





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
            dct_blocks[i, j] = dct_8x8_matrix(blocks[i, j], C)

    return dct_blocks, C

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

def dct_8x8_matrix(block, C):
    """Прямое DCT через матрицы: F = C * f * C^T"""
    block = block.astype(np.float32)
    # C * f * C^T
    temp = np.dot(C, block)
    dct_block = np.dot(temp, C.T)
    return dct_block

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


# =============================================
# ФУНКЦИЯ КВАНТОВАНИЯ
# =============================================

def quantize_dct(dct_block, quant_table):
    """
    Квантование коэффициентов DCT

    dct_block   - блок 8x8 после DCT (float)
    quant_table - матрица квантования 8x8
    """
    # Делим и округляем
    quantized = np.round(dct_block / quant_table)
    return quantized.astype(np.int32)  # обычно хранят как int


def dequantize_dct(quantized_block, quant_table):
    """Обратное квантование (для восстановления)"""
    return quantized_block * quant_table


# =============================================
# Применение квантования ко всем блокам
# =============================================

def apply_quantization(dct_blocks, quant_table):
    """Применяет квантование ко всем блокам изображения"""
    h_blocks, w_blocks, _, _ = dct_blocks.shape
    quantized_blocks = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.int32)

    for i in range(h_blocks):
        for j in range(w_blocks):
            quantized_blocks[i, j] = quantize_dct(dct_blocks[i, j], quant_table)

    print(f"Квантование выполнено для {h_blocks}x{w_blocks} блоков")
    return quantized_blocks


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

# Загружаем и подготавливаем
lena = Image.open('lenna.png').convert('RGB')
rgb = np.array(lena)
ycbcr = rgb_to_ycbcr(rgb)
Y = ycbcr[:, :, 0].astype(np.float32)

blocks = split_into_8x8_blocks(Y)
dct_blocks = apply_dct_matrix_to_image(Y)[0]  # используем версию через матрицы

print("\nПример блока после DCT (первый блок):")
print(np.round(dct_blocks[0, 0], 2))

# Квантуем
quantized_blocks = apply_quantization(dct_blocks, quant_table_y)

print("\nПример квантованного блока:")
print(quantized_blocks[0, 0])

print(f"\nКоличество нулей в первом блоке после квантования: {(quantized_blocks[0, 0] == 0).sum()}/64")
