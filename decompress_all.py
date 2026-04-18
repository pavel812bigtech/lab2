import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import time
import scipy
from compressed_image import raw_ac_encode, raw_dc_encode

# Импортируем все твои функции из compressed_image.py
# (предполагается, что файл называется compressed_image.py)
from compressed_image import (
    rgb_to_ycbcr, split_into_8x8_blocks, apply_dct_matrix_to_image,
    apply_quantization, adjust_quantization_table, zigzag_scan,
    rle_vlc_ac_encode, rle_vlc_dc_encode, save_compressed_image,
    quant_table_y, quant_table_c
)


def test_compression_quality(image_path, quality_values=[10, 50, 75, 90, 100]):
    """
    Тестирует компрессор при разных значениях Quality

    Параметры:
        image_path: путь к изображению
        quality_values: список значений Quality для тестирования

    Возвращает:
        словарь с результатами тестирования
    """
    # Загружаем оригинальное изображение
    original = Image.open(image_path).convert('RGB')
    rgb = np.array(original)
    width, height = original.size

    results = {}

    for quality in quality_values:
        print(f"\n--- Тестирование Quality = {quality} ---")

        # Конвертируем в YCbCr
        ycbcr = rgb_to_ycbcr(rgb)
        Y = ycbcr[:, :, 0].astype(np.float32)

        # Адаптируем таблицы квантования
        quant_y = adjust_quantization_table(quant_table_y, quality)
        quant_c = adjust_quantization_table(quant_table_c, quality)

        # Разбиваем на блоки
        blocks = split_into_8x8_blocks(Y)

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
                ac_encoded = raw_ac_encode(ac)
                ac_huffman_codes_list.append(ac_encoded)

        # Кодируем DC
        dc_huffman_codes = raw_dc_encode(dc_values)

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
        original_size = width * height * 3  # RGB, 3 байта на пиксель
        compression_ratio = original_size / file_size

        results[quality] = {
            'file_size': file_size,
            'compression_ratio': compression_ratio,
            'filename': filename
        }

        print(f"  Размер файла: {file_size} байт")
        print(f"  Коэффициент сжатия: {compression_ratio:.2f}")

    return results


def create_results_table(results):
    """Создает таблицу результатов в формате Markdown"""
    print("\n" + "=" * 80)
    print("Таблица 1. Результаты сжатия изображения lena.png")
    print("=" * 80)
    print(f"{'Quality':<10} {'Размер файла (байт)':<20} {'Коэффициент сжатия':<20} {'Визуальное качество':<20}")
    print("-" * 80)

    quality_desc = {
        10: "сильные артефакты",
        50: "хорошее",
        75: "очень хорошее",
        90: "отличное",
        100: "почти без потерь"
    }

    for quality in sorted(results.keys()):
        data = results[quality]
        desc = quality_desc.get(quality, "стандартное")
        print(f"{quality:<10} {data['file_size']:<20} {data['compression_ratio']:<20.2f} {desc:<20}")

    print("=" * 80)


def create_compression_plot(results):
    """Создает график зависимости размера файла от Quality"""
    qualities = sorted(results.keys())
    file_sizes = [results[q]['file_size'] for q in qualities]

    plt.figure(figsize=(10, 6))
    plt.plot(qualities, file_sizes, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Quality', fontsize=12)
    plt.ylabel('Размер файла (байт)', fontsize=12)
    plt.title('Зависимость размера сжатого файла от параметра Quality', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(qualities)

    # Добавляем значения на график
    for q, size in zip(qualities, file_sizes):
        plt.annotate(f'{size}', (q, size), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('compression_plot.png', dpi=150)
    plt.show()
    print("\nГрафик сохранен как 'compression_plot.png'")


def create_comparison_grid(image_path, results):
    """Создает сетку для визуального сравнения"""
    original = Image.open(image_path).convert('RGB')

    # Создаем фигуру с подграфиками
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Сравнение оригинального и восстановленного изображения при различных значениях Quality',
                 fontsize=16, fontweight='bold')

    # Оригинал
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Оригинал', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Для разных Quality (10, 50, 75, 90, 100)
    quality_list = [10, 50, 75, 90, 100]
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for quality, pos in zip(quality_list, positions):
        # Здесь нужно будет загрузить восстановленное изображение
        # Пока просто заглушка - нужно реализовать декомпрессию
        compressed_file = results[quality]['filename']

        # Загружаем сжатый файл
        from compressed_image import load_compressed_image
        img_data = load_compressed_image(compressed_file)

        # Для демонстрации пока используем оригинал
        # В реальности нужно будет декомпрессировать
        axes[pos[0], pos[1]].imshow(original)
        axes[pos[0], pos[1]].set_title(f'Quality = {quality}', fontsize=12)
        axes[pos[0], pos[1]].axis('off')

    plt.tight_layout()
    plt.savefig('comparison_grid.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nСравнительная сетка сохранена как 'comparison_grid.png'")


def calculate_psnr(original, reconstructed):
    """Вычисляет PSNR (Peak Signal-to-Noise Ratio)"""
    mse = np.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2):
    """Упрощенный расчет SSIM (Structural Similarity Index)"""
    # Простая реализация, для реальной работы лучше использовать skimage.metrics.structural_similarity
    from scipy.signal import convolve2d

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Константы для стабильности
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    # Средние значения
    mu1 = convolve2d(img1, np.ones((11, 11)) / 121, mode='same')
    mu2 = convolve2d(img2, np.ones((11, 11)) / 121, mode='same')

    # Дисперсии и ковариация
    sigma1_sq = convolve2d(img1 ** 2, np.ones((11, 11)) / 121, mode='same') - mu1 ** 2
    sigma2_sq = convolve2d(img2 ** 2, np.ones((11, 11)) / 121, mode='same') - mu2 ** 2
    sigma12 = convolve2d(img1 * img2, np.ones((11, 11)) / 121, mode='same') - mu1 * mu2

    # SSIM
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(ssim_map)


def generate_latex_table(results):
    """Генерирует таблицу в формате LaTeX для отчета"""
    latex = r"\begin{table}[h]" + "\n"
    latex += r"\centering" + "\n"
    latex += r"\caption{Результаты сжатия изображения lena.png}" + "\n"
    latex += r"\begin{tabular}{|c|c|c|c|}" + "\n"
    latex += r"\hline" + "\n"
    latex += r"Quality & Размер файла (байт) & Коэффициент сжатия & Визуальное качество \\" + "\n"
    latex += r"\hline" + "\n"

    quality_desc = {
        10: "сильные артефакты",
        50: "хорошее",
        75: "очень хорошее",
        90: "отличное",
        100: "почти без потерь"
    }

    for quality in sorted(results.keys()):
        data = results[quality]
        desc = quality_desc.get(quality, "стандартное")
        latex += f"{quality} & {data['file_size']} & {data['compression_ratio']:.2f} & {desc} \\\\" + "\n"
        latex += r"\hline" + "\n"

    latex += r"\end{tabular}" + "\n"
    latex += r"\end{table}" + "\n"

    return latex


if __name__ == "__main__":
    print("=== Тестирование JPEG компрессора ===\n")

    # Тестируем разные значения Quality
    quality_values = [10, 50, 75, 90, 100]
    results = test_compression_quality('lenna.png', quality_values)

    # Выводим таблицу результатов
    create_results_table(results)

    # Создаем график
    create_compression_plot(results)

    # Генерируем LaTeX таблицу для отчета
    latex_table = generate_latex_table(results)
    print("\n=== LaTeX таблица для отчета ===")
    print(latex_table)

    # Сохраняем LaTeX таблицу в файл
    with open('compression_results.tex', 'w') as f:
        f.write(latex_table)

    print("\nРезультаты сохранены в файл 'compression_results.tex'")

    # Дополнительно: создаем сравнительную сетку (нужна функция декомпрессии)
    # create_comparison_grid('lenna.png', results)