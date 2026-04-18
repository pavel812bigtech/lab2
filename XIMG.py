import numpy as np
import struct
import os


def save_compressed_image(filename,
                          width, height,
                          quality,
                          quant_table_y,  # таблица 8x8 для яркости
                          quant_table_c,  # таблица 8x8 для цвета
                          dc_huffman_codes,  # список словарей с DC
                          ac_huffman_codes,  # список списков словарей с AC (по блокам)
                          color_space=1):  # 0=RGB, 1=YCbCr
    """
    Сохраняет сжатое изображение в собственный формат с полными метаданными
    """
    with open(filename, 'wb') as f:

        # ====================== ЗАГОЛОВОК ======================
        f.write(b'XIMG_V1')  # Сигнатура (7 байт) - наша "визитка"

        # Основная информация
        f.write(struct.pack('<B', color_space))  # 1 байт
        f.write(struct.pack('<H', width))  # 2 байта
        f.write(struct.pack('<H', height))  # 2 байта
        f.write(struct.pack('<B', quality))  # 1 байт

        # Размер изображения в блоках
        block_h = (height + 7) // 8
        block_w = (width + 7) // 8
        f.write(struct.pack('<H', block_h))
        f.write(struct.pack('<H', block_w))

        # ====================== ТАБЛИЦЫ КВАНТОВАНИЯ ======================
        # Сохраняем обе таблицы (Y и C)
        f.write(quant_table_y.astype(np.uint16).flatten().tobytes())
        f.write(quant_table_c.astype(np.uint16).flatten().tobytes())

        # ====================== СЖАТЫЕ ДАННЫЕ ======================
        # 1. DC коэффициенты
        f.write(struct.pack('<I', len(dc_huffman_codes)))  # сколько DC

        for dc in dc_huffman_codes:
            bitstring = dc['full_bitstring']
            byte_length = (len(bitstring) + 7) // 8
            f.write(struct.pack('<B', len(bitstring)))  # длина в битах
            f.write(int(bitstring, 2).to_bytes(byte_length, 'big'))

        # 2. AC коэффициенты (для каждого блока)
        f.write(struct.pack('<I', len(ac_huffman_codes)))  # сколько блоков

        for block_ac in ac_huffman_codes:
            f.write(struct.pack('<H', len(block_ac)))  # сколько RLE+VLC пар в блоке

            for ac in block_ac:
                bitstring = ac['full_code']
                byte_length = (len(bitstring) + 7) // 8
                f.write(struct.pack('<B', len(bitstring)))  # длина в битах
                f.write(int(bitstring, 2).to_bytes(byte_length, 'big'))

    file_size = os.path.getsize(filename)
    print(f"   Файл успешно сохранён: {filename}")
    print(f"   Размер изображения: {width}x{height}")
    print(f"   Quality: {quality}")
    print(f"   Размер файла: {file_size:,} байт")


# ====================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ======================

if __name__ == "__main__":
    # Пример вызова (замени на свои переменные)
    save_compressed_image(
        filename="lena_compressed.ximg",
        width=512,
        height=512,
        quality=75,
        quant_table_y=quant_table_y,  # твоя таблица после adjust
        quant_table_c=quant_table_c,
        dc_huffman_codes=dc_vlc_result,  # результат huffman_dc_encode
        ac_huffman_codes=ac_vlc_result,  # результат huffman_ac_encode
        color_space=1  # 1 = YCbCr
    )