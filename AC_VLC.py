import numpy as np

# Таблица Huffman для AC коэффициентов (Luminance - Яркость)
ac_luminance_huffman = {
    (0, 0): '1010',  # EOB - End Of Block
    (0, 1): '00',
    (0, 2): '01',
    (0, 3): '100',
    (0, 4): '1011',
    (0, 5): '11010',
    (0, 6): '111000',
    (0, 7): '1111000',
    (0, 8): '1111110110',
    (0, 9): '1111111110000010',
    (0, 10): '1111111110000011',

    (1, 1): '1100',
    (1, 2): '11011',
    (1, 3): '1111001',
    (1, 4): '111110110',
    (1, 5): '11111110110',
    (1, 6): '1111111110000100',

    (2, 1): '11100',
    (2, 2): '11111000',
    (2, 3): '1111110111',
    (2, 4): '111111110100',

    (3, 1): '111010',
    (3, 2): '111110111',
    (3, 3): '111111110101',

    (4, 1): '111011',
    (4, 2): '1111111000',

    (5, 1): '1111010',
    (5, 2): '1111111001',

    (6, 1): '1111011',
    (6, 2): '11111110111',

    (7, 1): '11111001',
    (8, 1): '11111010',
    (9, 1): '11111011',
    (10, 1): '11111100',
    (11, 1): '11111101',
    (12, 1): '11111110',
    (15, 0): '111111111000000',  # ZRL - 16 нулей
}


def get_category(value):
    """Определяет категорию (SIZE) для AC коэффициента"""
    if value == 0:
        return 0
    abs_val = abs(value)
    cat = 1
    while abs_val >= (1 << cat):
        cat += 1
    return cat


def get_additional_bits(value, category):
    """Дополнительные биты для AC коэффициента"""
    if category == 0:
        return ''
    if value >= 0:
        return format(value, f'0{category}b')
    else:
        return format((1 << category) - 1 + value, f'0{category}b')


def rle_vlc_ac_encode(ac_coefficients):
    """
    Объединённая функция: RLE + VLC для 63 AC коэффициентов
    """
    if len(ac_coefficients) != 63:
        raise ValueError("Должно быть 63 AC коэффициента")

    ac = np.array(ac_coefficients, dtype=int)
    result = []
    zero_count = 0

    for val in ac:
        if val == 0:
            zero_count += 1
        else:
            while zero_count >= 16:
                # ZRL - 16 нулей
                huff_code = ac_luminance_huffman[(15, 0)]
                result.append({
                    'run': 15,
                    'category': 0,
                    'huffman': huff_code,
                    'bits': '',
                    'full_code': huff_code
                })
                zero_count -= 16

            category = get_category(val)
            huff_key = (zero_count, category)

            # Если такой комбинации нет в таблице — используем ближайшую (упрощённо)
            if huff_key not in ac_luminance_huffman:
                huff_key = (15, 0)  # fallback

            huff_code = ac_luminance_huffman[huff_key]
            additional = get_additional_bits(val, category)

            result.append({
                'run': zero_count,
                'category': category,
                'value': val,
                'huffman': huff_code,
                'bits': additional,
                'full_code': huff_code + additional
            })

            zero_count = 0

    # End Of Block
    if zero_count > 0 or not result:
        eob_code = ac_luminance_huffman[(0, 0)]
        result.append({
            'run': 0,
            'category': 0,
            'huffman': eob_code,
            'bits': '',
            'full_code': eob_code,
            'eob': True
        })

    return result


def print_ac_vlc(codes):
    print(f"{'Run':>3} | {'Cat':>3} | {'Value':>6} | {'Huffman':>12} | {'Additional':>12} | Full Code")
    print("-" * 85)
    for item in codes:
        val = item.get('value', 0)
        print(f"{item['run']:3} | {item['category']:3} | {val:6} | "
              f"{item['huffman']:>12} | {item['bits']:>12} | {item['full_code']}")


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

ac_test = [15, 0, 0, -3, 0, 0, 0, 5, 0, 0, 0, 0] + [0] * 51  # 63 элемента

print("AC коэффициенты (63):")
print(ac_test[:30], "...")

codes = rle_vlc_ac_encode(ac_test)
print_ac_vlc(codes)

total_bits = sum(len(item['full_code']) for item in codes)
print(f"\nВсего бит после VLC: {total_bits}")

