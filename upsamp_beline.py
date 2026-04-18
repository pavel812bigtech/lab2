from PIL import Image
import numpy as np




# =============================================
# 1. Билинейная интерполяция для одной точки (из прошлого пункта)
# =============================================
def bilinear_interpolation(x1, y1, x2, y2,
                           z11, z12, z21, z22,
                           x, y):
    """Билинейная интерполяция для одной точки"""
    if x2 - x1 == 0 or y2 - y1 == 0:
        return z11

    dx = (x - x1) / (x2 - x1)
    dy = (y - y1) / (y2 - y1)

    return (z11 * (1 - dx) * (1 - dy) +
            z12 * (1 - dx) * dy +
            z21 * dx * (1 - dy) +
            z22 * dx * dy)


# =============================================
# 2. ГЛАВНАЯ ФУНКЦИЯ — Resize с билинейной интерполяцией
# =============================================
def resize_bilinear(image_array, new_width, new_height):
    """
    Изменяет размер изображения с помощью билинейной интерполяции

    image_array: numpy массив (height, width) или (height, width, 3)
    new_width, new_height: желаемый новый размер
    """
    old_height, old_width = image_array.shape[:2]

    # Если изображение цветное — будем обрабатывать каждый канал отдельно
    if len(image_array.shape) == 3:
        channels = image_array.shape[2]
        resized = np.zeros((new_height, new_width, channels), dtype=np.uint8)

        for c in range(channels):
            resized[:, :, c] = resize_bilinear(image_array[:, :, c], new_width, new_height)
        return resized

    # Для grayscale или одного канала
    resized = np.zeros((new_height, new_width), dtype=np.float32)

    # Масштабные коэффициенты
    scale_x = (old_width - 1) / (new_width - 1) if new_width > 1 else 0
    scale_y = (old_height - 1) / (new_height - 1) if new_height > 1 else 0

    for i in range(new_height):
        for j in range(new_width):
            # Находим координаты в старом изображении
            x = j * scale_x
            y = i * scale_y

            # Находим координаты четырёх ближайших пикселей
            x1 = int(np.floor(x))
            y1 = int(np.floor(y))
            x2 = min(x1 + 1, old_width - 1)
            y2 = min(y1 + 1, old_height - 1)

            # Берём значения четырёх углов
            z11 = image_array[y1, x1]
            z12 = image_array[y2, x1]
            z21 = image_array[y1, x2]
            z22 = image_array[y2, x2]

            # Применяем билинейную интерполяцию
            resized[i, j] = bilinear_interpolation(x1, y1, x2, y2, z11, z12, z21, z22, x, y)

    return np.clip(resized, 0, 255).astype(np.uint8)


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

lena = Image.open('lenna.png')
img = np.array(lena.convert('RGB'))

print(f"Оригинальный размер: {img.shape[1]}x{img.shape[0]}")

# Тест ресайза
resized = resize_bilinear(img, new_width=256, new_height=256)
Image.fromarray(resized).save('lena_resized_256x256.png')

resized2 = resize_bilinear(img, new_width=800, new_height=600)
Image.fromarray(resized2).save('lena_resized_800x600.png')

print("Изображения сохранены:")
print("- lena_resized_256x256.png")
print("- lena_resized_800x600.png")

