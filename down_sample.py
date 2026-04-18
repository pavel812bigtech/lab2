from PIL import Image
import os
import numpy as np
import struct

def downsample(image_array):
    """
        Простая децимация с коэффициентом 2.
        Берём каждый второй пиксель по ширине и высоте.
        Работает с RGB, YCbCr и даже grayscale.
        """
    if len(image_array.shape) == 2:  # если grayscale (высота, ширина)
        return image_array[::2, ::2]
    else:  # если цветное (высота, ширина, 3)
        return image_array[::2, ::2, :]


def check_even_dimensions(image_array):
    """Проверяет, что размеры чётные"""
    h, w = image_array.shape[:2]
    if h % 2 != 0 or w % 2 != 0:
        print(f"Предупреждение: размеры {w}x{h} — нечётные! Децимация может работать некорректно.")
        return False
    return True


def save_to_raw(img_array, color_space, filename):
    """Сохраняет массив в raw формат"""
    h, w = img_array.shape[:2]

    if len(img_array.shape) == 3 and img_array.shape[2] == 3:
        img_type = 2  # цветное
    else:
        img_type = 1  # серый

    # Заголовок: type + color_space + width + height
    header = struct.pack('<BBII', img_type, color_space, w, h)

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(img_array.tobytes())

    space_name = "RGB" if color_space == 0 else "YCbCr" if img_type == 2 else "Gray"
    print(f"✓ {filename} сохранён | {w}x{h} | {space_name} | Размер: {len(header) + img_array.nbytes:,} байт")

lena = Image.open('lenna.png')
rgb = np.array(lena.convert('RGB'))

print(f"Оригинальный размер: {rgb.shape[1]}x{rgb.shape[0]}")

# Проверка размеров
check_even_dimensions(rgb)

# Делаем даунсэмплинг
down_rgb = downsample(rgb)

print(f"После децимации ×2: {down_rgb.shape[1]}x{down_rgb.shape[0]}")

# Сохраняем результаты
Image.fromarray(down_rgb).save('lena_downsampled.png')

# Сохраняем в raw
save_to_raw(rgb, 0, 'lena_full_rgb.raw')           # оригинал
save_to_raw(down_rgb, 0, 'lena_down_rgb.raw')      # уменьшенное RGB

print("\n=== ГОТОВО ===")
print("Созданные файлы:")
print("• lena_downsampled.png     — для визуальной проверки")
print("• lena_down_rgb.raw        — уменьшенное изображение")
print("• lena_full_rgb.raw        — оригинал")

