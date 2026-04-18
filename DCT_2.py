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



def dct_8x8_matrix(block, C):
    """Прямое DCT через матрицы: F = C * f * C^T"""
    block = block.astype(np.float32)
    # C * f * C^T
    temp = np.dot(C, block)
    dct_block = np.dot(temp, C.T)
    return dct_block



def idct_8x8_matrix(dct_block, C):
    """Обратное DCT через матрицы: f = C^T * F * C"""
    dct_block = dct_block.astype(np.float32)
    # C^T * F * C
    temp = np.dot(C.T, dct_block)
    block = np.dot(temp, C)
    return block



def apply_dct_matrix_to_image(Y_channel):
    C = create_dct_matrix(8)
    blocks = split_into_8x8_blocks(Y_channel)

    h_blocks, w_blocks, _, _ = blocks.shape
    dct_blocks = np.zeros((h_blocks, w_blocks, 8, 8), dtype=np.float32)

    for i in range(h_blocks):
        for j in range(w_blocks):
            dct_blocks[i, j] = dct_8x8_matrix(blocks[i, j], C)

    return dct_blocks, C


def reconstruct_from_dct_matrix(dct_blocks, C):
    h_blocks, w_blocks, _, _ = dct_blocks.shape
    reconstructed = np.zeros((h_blocks * 8, w_blocks * 8), dtype=np.float32)

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = idct_8x8_matrix(dct_blocks[i, j], C)
            reconstructed[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = block

    return np.clip(reconstructed, 0, 255).astype(np.uint8)


# =============================================
# ТЕСТИРОВАНИЕ И ПРОВЕРКА
# =============================================

lena = Image.open('lenna.png').convert('RGB')
rgb = np.array(lena)
ycbcr = rgb_to_ycbcr(rgb)
Y = ycbcr[:, :, 0].astype(np.float32)

print(f"Оригинал Y-канала: {Y.shape}")

# Применяем DCT через матрицы
dct_blocks, C_matrix = apply_dct_matrix_to_image(Y)
print("Прямое DCT через матрицы выполнено")

# Восстанавливаем
reconstructed_Y = reconstruct_from_dct_matrix(dct_blocks, C_matrix)
print("Обратное IDCT через матрицы выполнено")

# Проверка ошибки
error = np.mean(np.abs(Y.astype(np.float32) - reconstructed_Y.astype(np.float32)))
print(f"\nСредняя ошибка на пиксель: {error:.6f}")

if error < 1e-4:
    print("Отлично! Прямое и обратное ДКП через матрицы работают корректно.")
else:
    print("Ошибка больше ожидаемой — проверь матрицу C")

# Сохраняем восстановленное изображение
Image.fromarray(reconstructed_Y).save('reconstructed_matrix_Y.png')
print("Восстановленное изображение сохранено как reconstructed_matrix_Y.png")

