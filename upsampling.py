from PIL import Image
import numpy as np
import struct

print("=== ЛАБА 3: Downsampling + Upsampling + Артефакты пикселизации ===\n")


# =============================================
# 1. ДАУНСЭМПЛИНГ (из прошлого пункта)
# =============================================
def downsample_decimation(image_array, factor=2):
    """Децимация с коэффициентом factor (по умолчанию 2)"""
    if len(image_array.shape) == 2:  # grayscale
        return image_array[::factor, ::factor]
    else:  # color
        return image_array[::factor, ::factor, :]


# =============================================
# 2. АПСЕМПЛИНГ (НОВОЕ)
# =============================================
def upsample_nearest(image_array, factor=2):
    """
    Простой апсемплинг методом ближайшего соседа (Nearest Neighbor)
    Увеличивает изображение в factor раз
    """
    h, w = image_array.shape[:2]
    new_h, new_w = h * factor, w * factor

    if len(image_array.shape) == 2:  # grayscale
        up = np.repeat(image_array, factor, axis=0)
        up = np.repeat(up, factor, axis=1)
    else:  # color (RGB или YCbCr)
        up = np.repeat(image_array, factor, axis=0)
        up = np.repeat(up, factor, axis=1)

    return up


# =============================================
# 3. СОХРАНЕНИЕ В RAW
# =============================================
def save_to_raw(img_array, color_space, filename):
    h, w = img_array.shape[:2]
    img_type = 2 if len(img_array.shape) == 3 else 1

    header = struct.pack('<BBII', img_type, color_space, w, h)

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(img_array.tobytes())

    name = "RGB" if color_space == 0 else "YCbCr"
    print(f"✓ {filename} сохранён | {w}x{h} | {name}")


# =============================================
# ТЕСТИРОВАНИЕ И ПРОВЕРКА АРТЕФАКТОВ
# =============================================

lena = Image.open('lenna.png')
rgb = np.array(lena.convert('RGB'))

print(f"Оригинал: {rgb.shape[1]}x{rgb.shape[0]}")

# ==================== Тест 1: Коэффициент 2 ====================
print("\n--- Тест с коэффициентом 2 ---")
down2 = downsample_decimation(rgb, factor=2)
up2 = upsample_nearest(down2, factor=2)

Image.fromarray(down2).save('lena_down_x2.png')
Image.fromarray(up2).save('lena_up_x2.png')

diff2 = np.mean(np.abs(rgb.astype(int) - up2.astype(int)))
print(f"Средняя ошибка после down+up (x2): {diff2:.2f}")

# ==================== Тест 2: Коэффициент 4 (сильнее) ====================
print("\n--- Тест с коэффициентом 4 (сильная пикселизация) ---")
down4 = downsample_decimation(rgb, factor=4)
up4 = upsample_nearest(down4, factor=4)

Image.fromarray(down4).save('lena_down_x4.png')
Image.fromarray(up4).save('lena_up_x4.png')

diff4 = np.mean(np.abs(rgb.astype(int) - up4.astype(int)))
print(f"Средняя ошибка после down+up (x4): {diff4:.2f}")

print("\nСравнение:")
print("• x2 — артефакты заметны, но терпимо")
print("• x4 — сильная пикселизация, блоками, качество сильно упало")

# Сохраняем в raw
save_to_raw(rgb, 0, 'lena_original.raw')
save_to_raw(down2, 0, 'lena_down_x2.raw')
save_to_raw(up2, 0, 'lena_up_x2.raw')

print("\n=== ГОТОВО ===")
print("Посмотри файлы:")
print("• lena_up_x2.png  — после увеличения ×2")
print("• lena_up_x4.png  — после увеличения ×4 (очень сильная пикселизация)")
print("Чем больше коэффициент децимации — тем сильнее артефакты.")