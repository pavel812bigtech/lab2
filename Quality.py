import numpy as np


def adjust_quantization_table(quant_table, quality):
    """
    Адаптирует таблицу квантования под заданный уровень качества

    quality: от 1 до 100
    """
    if quality < 1 or quality > 100:
        raise ValueError("Quality должен быть от 1 до 100")

    # Вычисляем масштабный коэффициент S
    if quality < 50:
        S = 5000 / quality
    else:
        S = 200 - 2 * quality

    # Создаём новую таблицу
    new_table = np.zeros_like(quant_table, dtype=np.int32)

    for i in range(8):
        for j in range(8):
            # Основная формула
            temp = (quant_table[i, j] * S) / 100
            new_table[i, j] = np.ceil(temp)  # округление вверх

            # Защита от нуля (нельзя делить на 0 при квантовании)
            if new_table[i, j] < 1:
                new_table[i, j] = 1

    return new_table


def print_quant_table(table, title=""):
    print(f"\n{title}")
    print("   ", end="")
    for j in range(8):
        print(f"{j:3}", end="")
    print()
    for i in range(8):
        print(f"{i:2} ", end="")
        for j in range(8):
            print(f"{table[i, j]:3}", end="")
        print()


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

# Стандартная таблица для яркости
quant_y = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
])

print("Оригинальная таблица квантования (Quality = 50):")
print_quant_table(quant_y)

# Тестируем разные уровни качества
for q in [10, 25, 50, 75, 90, 100]:
    new_table = adjust_quantization_table(quant_y, q)
    print(f"\n=== Quality = {q} ===")
    print_quant_table(new_table, f"Адаптированная таблица (Quality={q})")

