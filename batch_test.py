import numpy as np
from PIL import Image, ImageDraw
import os
import matplotlib.pyplot as plt

from compressed_image import (
    rgb_to_ycbcr, split_into_8x8_blocks, apply_dct_matrix_to_image,
    apply_quantization, adjust_quantization_table, zigzag_scan,
    rle_vlc_ac_encode, rle_vlc_dc_encode, save_compressed_image,
    quant_table_y, quant_table_c, decompress_image
)


def test_image(image_path, quality_values=[10, 50, 75, 90, 100]):
    """Тестирует сжатие одного изображения"""

    # Загружаем изображение
    original = Image.open(image_path).convert('RGB')
    rgb = np.array(original)
    width, height = original.size
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    results = {}

    print(f"\n{'=' * 60}")
    print(f"Тестирование изображения: {image_name}")
    print(f"Размер: {width}x{height}")
    print(f"{'=' * 60}")

    for quality in quality_values:
        print(f"\n--- Quality = {quality} ---")

        # Конвертируем в YCbCr и берем яркость
        ycbcr = rgb_to_ycbcr(rgb)
        Y = ycbcr[:, :, 0].astype(np.float32)

        # Адаптируем таблицы квантования
        quant_y = adjust_quantization_table(quant_table_y, quality)
        quant_c = adjust_quantization_table(quant_table_c, quality)

        # DCT и квантование
        dct_blocks, _ = apply_dct_matrix_to_image(Y)
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

        # Сохраняем
        filename = f"{image_name}_q{quality}.ximg"
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

        file_size = os.path.getsize(filename)
        original_size = width * height * 3
        compression_ratio = original_size / file_size

        results[quality] = {
            'file_size': file_size,
            'compression_ratio': compression_ratio,
            'filename': filename
        }

        print(f"  Размер: {file_size} байт")
        print(f"  Коэф. сжатия: {compression_ratio:.2f}")

        # Декомпрессия и сохранение восстановленного изображения
        decompress_image(filename, f"{image_name}_restored_q{quality}.png")

    return results, image_name


def plot_results(all_results):
    """Строит графики для всех изображений"""
    plt.figure(figsize=(12, 6))

    colors = ['blue', 'red', 'green']
    markers = ['o', 's', '^']

    for idx, (image_name, results) in enumerate(all_results.items()):
        qualities = sorted(results.keys())
        file_sizes = [results[q]['file_size'] for q in qualities]

        plt.plot(qualities, file_sizes,
                 color=colors[idx % len(colors)],
                 marker=markers[idx % len(markers)],
                 linewidth=2, markersize=8,
                 label=image_name)

    plt.xlabel('Quality', fontsize=12)
    plt.ylabel('Размер файла (байт)', fontsize=12)
    plt.title('Зависимость размера сжатого файла от параметра Quality', fontsize=14)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks([10, 50, 75, 90, 100])

    plt.tight_layout()
    plt.savefig('compression_comparison.png', dpi=150)
    plt.show()

    print("\nГрафик сохранен как 'compression_comparison.png'")


def create_comparison_grid(image_path, quality_values=[10, 50, 75, 90, 100]):
    """Создает сетку с оригиналом и восстановленными изображениями"""
    original = Image.open(image_path).convert('RGB')
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Сравнение оригинального и восстановленных изображений\n({image_name})',
                 fontsize=14, fontweight='bold')

    # Оригинал
    axes[0, 0].imshow(original)
    axes[0, 0].set_title('Оригинал', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    # Восстановленные изображения
    positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]

    for quality, pos in zip(quality_values, positions):
        try:
            restored = Image.open(f'{image_name}_restored_q{quality}.png')
            axes[pos[0], pos[1]].imshow(restored)
            axes[pos[0], pos[1]].set_title(f'Quality = {quality}', fontsize=12)
            axes[pos[0], pos[1]].axis('off')
        except Exception as e:
            axes[pos[0], pos[1]].text(0.5, 0.5, f'Ошибка: {e}',
                                      ha='center', va='center')
            axes[pos[0], pos[1]].axis('off')

    plt.tight_layout()
    plt.savefig(f'{image_name}_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Сравнительная сетка сохранена как '{image_name}_comparison.png'")


if __name__ == "__main__":
    print("=== БАТЧНОЕ ТЕСТИРОВАНИЕ JPEG КОМПРЕССОРА ===\n")

    # Список тестовых изображений
    test_images = ['lenna.png']  # Добавь другие изображения

    # Если нет второго изображения, создаем его
    if not os.path.exists('test_image.png'):
        print("Создаем тестовое изображение...")
        img = Image.new('RGB', (512, 512), color='white')
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 200, 200], fill='red')
        draw.rectangle([300, 50, 450, 200], fill='green')
        draw.ellipse([50, 300, 200, 450], fill='blue')
        draw.ellipse([300, 300, 450, 450], fill='yellow')
        img.save('test_image.png')
        test_images.append('test_image.png')

    quality_values = [10, 50, 75, 90, 100]
    all_results = {}

    # Тестируем каждое изображение
    for image_path in test_images:
        if os.path.exists(image_path):
            results, image_name = test_image(image_path, quality_values)
            all_results[image_name] = results

            # Создаем визуальное сравнение для каждого изображения
            create_comparison_grid(image_path, quality_values)

    # Строим общий график
    plot_results(all_results)

    # Выводим итоговую таблицу
    print("\n" + "=" * 80)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("=" * 80)

    for image_name, results in all_results.items():
        print(f"\n{image_name}:")
        print(f"{'Quality':<10} {'Размер (байт)':<15} {'Коэф. сжатия':<15}")
        print("-" * 40)
        for q in sorted(results.keys()):
            data = results[q]
            print(f"{q:<10} {data['file_size']:<15} {data['compression_ratio']:<15.2f}")