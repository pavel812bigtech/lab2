
from PIL import Image
import numpy as np
import struct
import os

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

def ycbcr_to_rgb(ycbcr_array):
    """Преобразует YCbCr обратно в RGB"""
    Y = ycbcr_array[:, :, 0].astype(np.float32)
    Cb = ycbcr_array[:, :, 1].astype(np.float32)
    Cr = ycbcr_array[:, :, 2].astype(np.float32)

    R = Y + 1.402 * (Cr - 128)
    G = Y - 0.34414 * (Cb - 128) - 0.71414 * (Cr - 128)
    B = Y + 1.772 * (Cb - 128)

    R = np.clip(R, 0, 255).astype(np.uint8)
    G = np.clip(G, 0, 255).astype(np.uint8)
    B = np.clip(B, 0, 255).astype(np.uint8)

    return np.stack([R, G, B], axis=2)


def save_to_raw_color(img, color_space, filename):
    """color_space: 0 = RGB, 1 = YCbCr"""
    if isinstance(img, Image.Image):
        rgb = np.array(img.convert('RGB'))
        if color_space == 1:
            data = rgb_to_ycbcr(rgb)
        else:
            data = rgb
    else:
        data = img

    h, w = data.shape[:2]

    # ИСПРАВЛЕННЫЙ ЗАГОЛОВОК: B(type) + B(color_space) + I(width) + I(height)
    header = struct.pack('<BBII', 2, color_space, w, h)  # ← теперь правильно!

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(data.tobytes())

    space_name = "RGB" if color_space == 0 else "YCbCr"
    total_size = len(header) + data.nbytes
    print(f"✓ {filename} сохранён → {space_name} | Размер: {total_size:,} байт")


def load_from_raw(filename):
    """Загружает raw с новым заголовком"""
    with open(filename, 'rb') as f:
        header = f.read(11)  # 1+1+4+4 = 10 байт? Подождите: B B I I = 1+1+4+4 = 10 байт
        img_type, color_space, w, h = struct.unpack('<BBII', header)

        bytes_per_pixel = 3
        raw_data = f.read()
        data = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, w, bytes_per_pixel))

    space_name = "RGB" if color_space == 0 else "YCbCr"
    print(f"Загружено: {w}x{h} | Цветовое пространство: {space_name}")
    return data, color_space


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

print("Тестируем преобразования на Lena...")

lena = Image.open('lenna.png')
rgb_array = np.array(lena.convert('RGB'))

ycbcr = rgb_to_ycbcr(rgb_array)
rgb_back = ycbcr_to_rgb(ycbcr)

Image.fromarray(ycbcr).save('lena_ycbcr.png')
Image.fromarray(rgb_back).save('lena_back_to_rgb.png')

diff = np.abs(rgb_array.astype(int) - rgb_back.astype(int)).mean()
print(f"\nСредняя ошибка на пиксель: {diff:.3f}")

if diff < 2.0:
    print("Преобразования работают нормально.")
else:
    print("Ошибка большая.")

# Сохраняем в raw оба варианта
save_to_raw_color(lena, 0, 'lena_rgb.raw')  # RGB
save_to_raw_color(lena, 1, 'lena_ycbcr.raw')  # YCbCr

