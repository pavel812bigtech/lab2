import numpy as np


# Таблица Huffman для AC Luminance (Яркость)

ac_huffman_luminance = {
    (0, 0): '1010',  # EOB
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

    (3, 1): '111010',
    (3, 2): '111110111',

    (4, 1): '111011',
    (5, 1): '1111010',
    (6, 1): '1111011',
    (7, 1): '11111001',
    (8, 1): '11111010',
    (9, 1): '11111011',
    (10, 1): '11111100',
    (11, 1): '11111101',
    (12, 1): '11111110',
    (15, 0): '111111111000000'  # ZRL - 16 нулей
}


def get_category_ac(value):
    """Категория (SIZE) для AC коэффициента"""
    if value == 0:
        return 0
    absv = abs(value)
    cat = 1
    while absv >= (1 << cat):
        cat += 1
    return cat


def get_additional_bits_ac(value, category):
    """Дополнительные биты для AC коэффициента"""
    if category == 0:
        return ''
    if value >= 0:
        return format(value, f'0{category}b')
    else:
        return format((1 << category) - 1 + value, f'0{category}b')


def huffman_ac_encode(rle_pairs, is_luminance=True):
    """
    Кодирование (RUN, SIZE) с помощью Huffman
    rle_pairs - список пар из RLE [(run, value), ...]
    """
    table = ac_huffman_luminance  # пока используем только для Y

    result = []

    for run, value in rle_pairs:
        category = get_category_ac(value)

        if run == 0 and value == 0:  # EOB
            huff_code = table[(0, 0)]
            result.append({
                'run': 0,
                'category': 0,
                'value': 0,
                'huffman': huff_code,
                'bits': '',
                'full_code': huff_code,
                'comment': 'EOB'
            })
            continue

        # Для обычных пар
        key = (run, category)

        # Если комбинации нет в таблице — используем ZRL (15,0)
        if key not in table:
            key = (15, 0)

        huff_code = table[key]
        additional = get_additional_bits_ac(value, category)

        result.append({
            'run': run,
            'category': category,
            'value': value,
            'huffman': huff_code,
            'bits': additional,
            'full_code': huff_code + additional
        })

    return result


def print_ac_huffman(codes):
    print(f"{'Run':>3} | {'Cat':>3} | {'Value':>6} | {'Huffman':>12} | {'Additional':>12} | Full Code")
    print("-" * 85)
    for item in codes:
        comment = item.get('comment', '')
        print(f"{item['run']:3} | {item['category']:3} | {item['value']:6} | "
              f"{item['huffman']:>12} | {item['bits']:>12} | {item['full_code']} {comment}")


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

# Пример RLE-пар (из предыдущего пункта)
rle_pairs = [
    (0, 15),
    (2, -3),
    (6, 5),
    (0, 0)  # EOB
]

print("Входные RLE-пары (RUN, VALUE):")
for p in rle_pairs:
    print(p)

print("\nПосле Huffman-кодирования AC:")
huffman_codes = huffman_ac_encode(rle_pairs, is_luminance=True)
print_ac_huffman(huffman_codes)

total_bits = sum(len(item['full_code']) for item in huffman_codes)
print(f"\nВсего бит после Huffman AC: {total_bits}")

