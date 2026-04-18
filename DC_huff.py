import numpy as np


# =============================================
# Стандартные таблицы Хаффмана для DC (из ITU-T T.81)
# =============================================

# Для яркости (Luminance - Y)
dc_huffman_luminance = {
    0: '00',  # Category 0
    1: '010',  # Category 1
    2: '011',  # Category 2
    3: '100',  # Category 3
    4: '101',  # Category 4
    5: '110',  # Category 5
    6: '1110',  # Category 6
    7: '11110',  # Category 7
    8: '111110',  # Category 8
    9: '1111110',  # Category 9
    10: '11111110',  # Category 10
    11: '111111110'  # Category 11
}

# Для цветности (Chrominance - Cb/Cr)
dc_huffman_chrominance = {
    0: '00',  # Category 0
    1: '01',  # Category 1
    2: '10',  # Category 2
    3: '110',  # Category 3
    4: '1110',  # Category 4
    5: '11110',  # Category 5
    6: '111110',  # Category 6
    7: '1111110',  # Category 7
    8: '11111110',  # Category 8
    9: '111111110',  # Category 9
    10: '1111111110',  # Category 10
    11: '11111111110'  # Category 11
}


def get_dc_category(value):
    """Возвращает категорию (SIZE) для разностного DC коэффициента"""
    if value == 0:
        return 0
    abs_val = abs(value)
    category = 1
    while abs_val >= (1 << category):
        category += 1
    return category


def get_dc_additional_bits(value, category):
    """Дополнительные биты для представления значения"""
    if category == 0:
        return ""
    if value >= 0:
        return format(value, f'0{category}b')
    else:
        # Для отрицательных чисел — инвертируем биты
        return format((1 << category) - 1 + value, f'0{category}b')


def huffman_dc_encode(dc_dpcm_list, is_luminance=True):
    """
    Кодирование DC коэффициентов с помощью Huffman (ITU-T T.81)

    dc_dpcm_list - список разностных DC коэффициентов
    is_luminance - True для Y (яркость), False для Cb/Cr
    """
    table = dc_huffman_luminance if is_luminance else dc_huffman_chrominance
    result = []

    for value in dc_dpcm_list:
        category = get_dc_category(value)
        huffman_code = table[category]
        additional_bits = get_dc_additional_bits(value, category)

        result.append({
            'dc_diff': value,
            'category': category,
            'huffman_code': huffman_code,
            'additional_bits': additional_bits,
            'full_bitstring': huffman_code + additional_bits
        })

    return result


def print_huffman_dc(codes):
    print(f"{'DC diff':>8} | {'Cat':>3} | {'Huffman':>10} | {'Additional bits':>18} | {'Full bits':>20}")
    print("-" * 75)
    for item in codes:
        print(f"{item['dc_diff']:8} | {item['category']:3} | "
              f"{item['huffman_code']:>10} | "
              f"{item['additional_bits']:>18} | "
              f"{item['full_bitstring']:>20}")


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

dc_dpcm = [120, 5, -2, 7, -2, -1, 8, -3, 0, 4, -5]

print("Разностные DC коэффициенты:")
print(dc_dpcm)

print("\n--- Кодирование для яркости (Y) ---")
codes_y = huffman_dc_encode(dc_dpcm, is_luminance=True)
print_huffman_dc(codes_y)

total_bits_y = sum(len(item['full_bitstring']) for item in codes_y)
print(f"\nВсего бит (Y): {total_bits_y}")

