import numpy as np

print("=== 3. RLE-кодирование AC коэффициентов (JPEG стандарт) ===\n")


def rle_ac_encode(ac_coefficients):
    """
    RLE-кодирование для 63 AC коэффициентов одного блока
    Вход: список или массив из ТОЧНО 63 элементов
    """
    if len(ac_coefficients) != 63:
        raise ValueError(f"Должно быть ровно 63 AC коэффициента, а пришло {len(ac_coefficients)}")

    ac = np.array(ac_coefficients, dtype=int)
    rle = []
    zero_count = 0

    for val in ac:
        if val == 0:
            zero_count += 1
        else:
            # Обрабатываем длинные последовательности нулей (больше 15)
            while zero_count >= 16:
                rle.append((15, 0))  # 15 нулей + 1 "фиктивный" ноль = 16 нулей
                zero_count -= 16

            rle.append((zero_count, int(val)))
            zero_count = 0

    # Если в конце блока остались только нули — ставим EOB
    if zero_count > 0 or len(rle) == 0:
        rle.append((0, 0))  # End Of Block

    return rle


def print_rle(rle_codes):
    print(f"RLE-коды ({len(rle_codes)} пар):")
    for run, val in rle_codes:
        if run == 0 and val == 0:
            print("   (0, 0)  ← EOB (End Of Block)")
        else:
            print(f"   ({run:2d}, {val:4d})")


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

# 63 AC коэффициент
ac_example = [
    15, 0, 0, -3, 0, 0, 0, 0, 0, 0, 5, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
]  # ровно 63 элемента

print("AC коэффициенты (63 шт):")
print(ac_example)
print(f"Нулей: {ac_example.count(0)} из 63\n")

rle_result = rle_ac_encode(ac_example)
print_rle(rle_result)

print(f"\nВсего RLE-пар: {len(rle_result)}")

