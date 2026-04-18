from PIL import Image
import numpy as np

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



def dct_2d_general(block):
    """Общая примитивная ДКП для блока произвольного размера NxM"""
    block = np.array(block, dtype=np.float32)
    N, M = block.shape
    dct = np.zeros((N, M), dtype=np.float32)

    for u in range(N):
        for v in range(M):
            sum_val = 0.0
            for x in range(N):
                for y in range(M):
                    cos1 = np.cos((2 * x + 1) * u * np.pi / (2 * N))
                    cos2 = np.cos((2 * y + 1) * v * np.pi / (2 * M))
                    sum_val += block[x, y] * cos1 * cos2

            # alpha
            alpha_u = np.sqrt(1.0 / N) if u == 0 else np.sqrt(2.0 / N)
            alpha_v = np.sqrt(1.0 / M) if v == 0 else np.sqrt(2.0 / M)

            dct[u, v] = alpha_u * alpha_v * sum_val

    return dct


# =============================================
# 3. ПРИМЕНЕНИЕ КО ВСЕМУ ИЗОБРАЖЕНИЮ
# =============================================
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


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

lena = Image.open('lenna.png').convert('RGB')
rgb = np.array(lena)
ycbcr = rgb_to_ycbcr(rgb)
Y = ycbcr[:, :, 0]

print(f"Размер Y: {Y.shape}")
print("Тип данных коэффициентов: float32 (могут быть отрицательными и дробными)")

dct_blocks = apply_dct_to_image(Y)

print("\nПример коэффициентов первого блока (округлено):")
print(np.round(dct_blocks[0, 0], decimals=2))

# Временная сложность
print("\nВременная сложность:")
print("• Для одного блока 8x8: O(8²·8²) = O(4096) операций")
print("• Для произвольного NxM: O(N²·M²) операций")
print("• Для всего изображения: O(H·W·N·M)")

