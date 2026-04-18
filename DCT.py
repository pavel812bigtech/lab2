from PIL import Image
import numpy as np

print("=== ЛАБА 3: Дискретное косинусное преобразование (DCT) ===")
print("1. Разбиение изображения на блоки 8x8\n")


# =============================================
# ФУНКЦИЯ РАЗБИЕНИЯ НА БЛОКИ 8x8
# =============================================
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


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

# Загружаем изображение (лучше использовать Y-канал)
lena = Image.open('lenna.png').convert('RGB')
rgb = np.array(lena)

# Преобразуем в YCbCr и берём только Y (яркость)
ycbcr = rgb_to_ycbcr(rgb)
Y = ycbcr[:, :, 0]  # только канал яркости

print(f"Оригинальный размер Y-канала: {Y.shape}")

blocks = split_into_8x8_blocks(Y)

# Пример: смотрим первый блок
print("\nПример первого блока 8x8:")
print(blocks[0, 0])


# Сохраним один блок для просмотра
Image.fromarray(blocks[0, 0]).save('example_8x8_block.png')
