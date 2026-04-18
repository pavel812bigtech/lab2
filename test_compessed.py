import numpy as np
from PIL import Image
import os

# Импортируем все функции из твоего файла
from compressed_image import (
    rgb_to_ycbcr, split_into_8x8_blocks, apply_dct_matrix_to_image,
    apply_quantization, adjust_quantization_table, zigzag_scan,
    rle_vlc_ac_encode, rle_vlc_dc_encode, save_compressed_image,
    quant_table_y, quant_table_c
)

print("=== Начинаем тестирование компрессора ===\n")

# Загружаем изображение
original_image = Image.open('lenna.png').convert('RGB')
rgb = np.array(original_image)
width, height = original_image.size
print(f"Размер изображения: {width}x{height}")

# Список Quality для тестирования
quality_list = [10, 50, 75, 90, 100]

# Словарь для хранения результатов
results = {}

# Тестируем каждое значение Quality
for quality in quality_list:
    print(f"\n--- Тестируем Quality = {quality} ---")

    # Конвертируем в YCbCr и берем только яркость
    ycbcr = rgb_to_ycbcr(rgb)
    Y = ycbcr[:, :, 0].astype(np.float32)

    # Адаптируем таблицы квантования
    quant_y = adjust_quantization_table(quant_table_y, quality)
    quant_c = adjust_quantization_table(quant_table_c, quality)

    # DCT
    dct_blocks, _ = apply_dct_matrix_to_image(Y)

    # Квантование
    quantized_blocks = apply_quantization(dct_blocks, quant_y)

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
            ac_encoded = rle_vlc_ac_encode(ac)
            ac_huffman_codes_list.append(ac_encoded)

    dc_huffman_codes = rle_vlc_dc_encode(dc_values)

    # Сохраняем сжатый файл
    filename = f"lena_compressed_q{quality}.ximg"
    save_compressed_image(
        filename=filename,
        width=width,
        height=height,
        quality=quality,
        quant_table_y=quant_y,
        quant_table_c=quant_c,
        dc_huffman_codes=dc_huffman_codes,
        ac_huffman_codes_list=ac_huffman_codes_list,
        color_space=1
    )

    # Получаем размер файла
    file_size = os.path.getsize(filename)

    # Вычисляем коэффициент сжатия
    original_size = width * height * 3  # 3 байта на пиксель (RGB)
    compression_ratio = original_size / file_size

    # Сохраняем результаты
    results[quality] = {
        'file_size': file_size,
        'compression_ratio': compression_ratio
    }

    print(f"  Размер файла: {file_size} байт")
    print(f"  Коэффициент сжатия: {compression_ratio:.2f}")
    print(f"  Файл сохранён: {filename}")

print("\n" + "=" * 60)
print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
print("=" * 60)
print("Quality | Размер файла (байт) | Коэффициент сжатия | Визуальное качество")
print("-" * 60)

quality_desc = {
    10: "сильные артефакты",
    50: "хорошее",
    75: "очень хорошее",
    90: "отличное",
    100: "почти без потерь"
}

for q in sorted(results.keys()):
    size = results[q]['file_size']
    ratio = results[q]['compression_ratio']
    desc = quality_desc[q]
    print(f"{q:7} | {size:18} | {ratio:18.2f} | {desc}")

print("=" * 60)