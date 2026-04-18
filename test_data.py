
import os
from PIL import Image
import struct


# Функция для сохранения в НАШ raw-формат
# Метаданные в начале файла:
# 1 байт — тип (0=ЧБ, 1=оттенки серого, 2=цветное)
# 4 байта — ширина (uint32 little-endian)
# 4 байта — высота (uint32 little-endian)
# Потом идут сами пиксели: 1 байт на пиксель (ЧБ/серый) или 3 байта (RGB)
def save_to_raw(img, img_type, filename):
    w, h = img.size
    if img_type == 2:  # цветное
        pixel_data = img.convert('RGB').tobytes()
        bpp = 3
    else:  # 0 или 1 — ЧБ или серый
        # Даже если картинка в mode='1', переводим в 'L', чтобы было ровно 1 байт на пиксель (0 или 255)
        pixel_data = img.convert('L').tobytes()
        bpp = 1

    # Собираем заголовок
    header = struct.pack('<BII', img_type, w, h)  # B = 1 байт, I = 4 байта

    with open(filename, 'wb') as f:
        f.write(header)
        f.write(pixel_data)

    raw_size = len(header) + len(pixel_data)
    print(f"✓ {filename} готов (размер raw = {raw_size} байт)")
    return raw_size


# ====================== 1. Lena.png (512x512) ======================
lena_path = 'lenna.png'
if not os.path.exists(lena_path):
    print("изображение не найдено!")
    exit()

lena = Image.open(lena_path)
print(f"Lena загружена: {lena.size} px, режим {lena.mode}")

lena_raw_size = save_to_raw(lena, 2, 'lena.raw')  # сохраняем как цветное
lena_png_size = os.path.getsize(lena_path)
print(f"Lena.png размер: {lena_png_size} байт → коэффициент сжатия PNG ≈ {lena_raw_size / lena_png_size:.2f} раз\n")

# ====================== 2. Цветное изображение (>=512x512) ======================
# Если хочешь своё цветное — замени путь ниже на свой файл (например, 'my_photo.jpg')
color_path = 'color.jpg'  # ←←← здесь можно поменять на своё цветное изображение
color = Image.open(color_path)
print(f"Цветное изображение загружено: {color.size} px, режим {color.mode}")

color_raw_size = save_to_raw(color, 2, 'color.raw')
color_png_size = os.path.getsize(color_path)
print(f"color.png размер: {color_png_size} байт → коэффициент сжатия PNG ≈ {color_raw_size / color_png_size:.2f} раз\n")

# ====================== 3. Оттенки серого ======================
gray = color.convert('L')
gray.save('gray.png')
print("✓ gray.png сохранён (оттенки серого)")

gray_raw_size = save_to_raw(gray, 1, 'gray.raw')
gray_png_size = os.path.getsize('gray.png')
print(f"gray.png размер: {gray_png_size} байт → коэффициент сжатия PNG ≈ {gray_raw_size / gray_png_size:.2f} раз\n")

# ====================== 4. ЧБ без дизеринга (round) ======================
bw_no_dither = color.convert('L').convert('1', dither=Image.Dither.NONE)
bw_no_dither.save('bw_no_dither.png')
print("✓ bw_no_dither.png сохранён (ЧБ, без дизеринга)")

bw_no_raw_size = save_to_raw(bw_no_dither, 0, 'bw_no_dither.raw')
bw_no_png_size = os.path.getsize('bw_no_dither.png')
print(
    f"bw_no_dither.png размер: {bw_no_png_size} байт → коэффициент сжатия PNG ≈ {bw_no_raw_size / bw_no_png_size:.2f} раз\n")

# ====================== 5. ЧБ с дизерингом ======================
bw_dither = color.convert('1')  # по умолчанию с дизерингом
bw_dither.save('bw_dither.png')
print("✓ bw_dither.png сохранён (ЧБ, с дизерингом)")

bw_dither_raw_size = save_to_raw(bw_dither, 0, 'bw_dither.raw')
bw_dither_png_size = os.path.getsize('bw_dither.png')
print(
    f"bw_dither.png размер: {bw_dither_png_size} байт → коэффициент сжатия PNG ≈ {bw_dither_raw_size / bw_dither_png_size:.2f} раз\n")


