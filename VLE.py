import numpy as np

# =============================================
# Таблица Huffman для DC коэффициентов (Luminance - Y)
# =============================================
dc_luminance_huffman = {
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


def get_category(value):
    """Определяет категорию (Size) для DC разности"""
    if value == 0:
        return 0
    abs_val = abs(value)
    category = 1
    while abs_val > (1 << (category - 1)):
        category += 1
    return category


def get_additional_bits(value, category):
    """Возвращает дополнительные биты"""
    if category == 0:
        return ''

    if value >= 0:
        return format(value, f'0{category}b')
    else:
        # Для отрицательных: инвертируем биты
        return format((1 << category) - 1 + value, f'0{category}b')


def vlc_dc_encode(dc_dpcm_values, is_luminance=True):
    """
    Кодирование переменной длины для DC коэффициентов (DPCM)

    dc_dpcm_values - список разностных DC коэффициентов
    is_luminance - True для Y (яркость), False для цветовых каналов
    """
    if is_luminance:
        huffman_table = dc_luminance_huffman
    else:
        # Для простоты пока используем ту же таблицу (в реальности чуть другая)
        huffman_table = dc_luminance_huffman

    result = []

    for value in dc_dpcm_values:
        category = get_category(value)
        huffman_code = huffman_table[category]
        additional_bits = get_additional_bits(value, category)

        result.append({
            'value': value,
            'category': category,
            'huffman': huffman_code,
            'bits': additional_bits,
            'full_code': huffman_code + additional_bits
        })

    return result


def print_vlc_dc(vlc_result):
    print(f"{'DC diff':>8} | {'Cat':>3} | {'Huffman':>10} | {'Additional':>12} | Full Code")
    print("-" * 70)
    for item in vlc_result:
        print(f"{item['value']:8} | {item['category']:3} | {item['huffman']:>10} | "
              f"{item['bits']:>12} | {item['full_code']}")


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

dc_dpcm = [120, 5, -2, 7, -2, -1, 8, -3, 0, 4]

print("Разностные DC коэффициенты (DPCM):")
print(dc_dpcm)

vlc_codes = vlc_dc_encode(dc_dpcm, is_luminance=True)

print_vlc_dc(vlc_codes)

print(f"\nВсего бит в коде: {sum(len(item['full_code']) for item in vlc_codes)}")

