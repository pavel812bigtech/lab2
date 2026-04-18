from compressed_image import decompress_image
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import os

print("=== Декомпрессия сжатых файлов ===\n")

# Список Quality, для которых есть сжатые файлы
quality_list = [10, 50, 75, 90, 100]

# Декомпрессируем каждый файл
for quality in quality_list:
    input_file = f"lena_compressed_q{quality}.ximg"
    output_file = f"lena_restored_q{quality}.png"

    if not os.path.exists(input_file):
        print(f"Файл {input_file} не найден! Сначала запусти тестирование.")
        continue

    print(f"\n--- Декомпрессия Quality = {quality} ---")
    try:
        Y_channel = decompress_image(input_file, output_file)
        print(f"✓ Сохранено: {output_file}")
    except Exception as e:
        print(f"✗ Ошибка: {e}")

print("\n=== Создание визуального сравнения ===\n")

# Загружаем оригинал (переводим в черно-белый для сравнения)
original = Image.open('lenna.png').convert('L')
original_array = np.array(original)

# Создаем сетку для сравнения
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Сравнение оригинального и восстановленных изображений', fontsize=16, fontweight='bold')

# Оригинал
axes[0, 0].imshow(original_array, cmap='gray')
axes[0, 0].set_title('Оригинал (512×512)', fontsize=12, fontweight='bold')
axes[0, 0].axis('off')

# Восстановленные изображения
positions = [(0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
quality_order = [10, 50, 75, 90, 100]

for quality, pos in zip(quality_order, positions):
    restored_file = f'lena_restored_q{quality}.png'
    if os.path.exists(restored_file):
        restored = Image.open(restored_file)
        axes[pos[0], pos[1]].imshow(restored, cmap='gray')
        axes[pos[0], pos[1]].set_title(f'Quality = {quality}', fontsize=12)
        axes[pos[0], pos[1]].axis('off')
    else:
        axes[pos[0], pos[1]].text(0.5, 0.5, f'Файл не найден\n{restored_file}',
                                  ha='center', va='center')
        axes[pos[0], pos[1]].axis('off')

plt.tight_layout()
plt.savefig('visual_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n✓ Сравнительная сетка сохранена как 'visual_comparison.png'")