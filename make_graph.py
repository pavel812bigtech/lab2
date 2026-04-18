import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Для сохранения без открытия окна

# Данные из твоих результатов
quality = [10, 50, 75, 90, 100]
file_sizes = [92877, 127039, 154175, 206712, 497005]
compression_ratios = [8.47, 6.19, 5.10, 3.80, 1.58]

# Создаем фигуру с двумя подграфиками
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Зависимость размера файла от Quality
ax1.plot(quality, file_sizes, 'bo-', linewidth=2, markersize=8, color='blue')
ax1.set_xlabel('Quality', fontsize=12)
ax1.set_ylabel('Размер файла (байт)', fontsize=12)
ax1.set_title('Зависимость размера сжатого файла от Quality', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(quality)

# Добавляем значения на график
for q, size in zip(quality, file_sizes):
    ax1.annotate(f'{size:,}', (q, size), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9)

# График 2: Зависимость коэффициента сжатия от Quality
ax2.plot(quality, compression_ratios, 'ro-', linewidth=2, markersize=8, color='red')
ax2.set_xlabel('Quality', fontsize=12)
ax2.set_ylabel('Коэффициент сжатия', fontsize=12)
ax2.set_title('Зависимость коэффициента сжатия от Quality', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(quality)

# Добавляем значения на график
for q, ratio in zip(quality, compression_ratios):
    ax2.annotate(f'{ratio:.2f}', (q, ratio), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('compression_graphs.png', dpi=150, bbox_inches='tight')
print("График сохранён как 'compression_graphs.png'")

# Также сохраняем отдельно первый график (для отчета)
plt.figure(figsize=(10, 6))
plt.plot(quality, file_sizes, 'bo-', linewidth=2, markersize=8)
plt.xlabel('Quality', fontsize=14)
plt.ylabel('Размер файла (байт)', fontsize=14)
plt.title('Зависимость размера сжатого файла от параметра Quality', fontsize=14)
plt.grid(True, alpha=0.3)
plt.xticks(quality)

for q, size in zip(quality, file_sizes):
    plt.annotate(f'{size:,}', (q, size), textcoords="offset points",
                xytext=(0, 10), ha='center', fontsize=10)

plt.savefig('file_size_vs_quality.png', dpi=150, bbox_inches='tight')
print("График 'file_size_vs_quality.png' сохранён")