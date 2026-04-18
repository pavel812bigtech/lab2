import numpy as np





def dpcm_dc_encode(dc_coefficients):
    """
    Разностное кодирование DC коэффициентов

    dc_coefficients - список или массив DC коэффициентов (один из каждого блока)
    """
    if len(dc_coefficients) == 0:
        return np.array([])

    dc = np.array(dc_coefficients, dtype=np.int32)
    dpcm = np.zeros_like(dc)

    dpcm[0] = dc[0]  # первый DC сохраняется как есть
    dpcm[1:] = dc[1:] - dc[:-1]  # остальные — разница с предыдущим

    return dpcm


def dpcm_dc_decode(dpcm_codes):
    """Обратное разностное декодирование (для восстановления)"""
    if len(dpcm_codes) == 0:
        return np.array([])

    dc = np.zeros_like(dpcm_codes, dtype=np.int32)
    dc[0] = dpcm_codes[0]

    for i in range(1, len(dpcm_codes)):
        dc[i] = dc[i - 1] + dpcm_codes[i]

    return dc


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

# Пример из реальной лабы
dc_values = [120, 125, 123, 130, 128, 127, 135, 130, 128]

print("Исходные DC коэффициенты:")
print(dc_values)

encoded = dpcm_dc_encode(dc_values)
print("\nПосле разностного кодирования (DPCM):")
print(list(encoded))

decoded = dpcm_dc_decode(encoded)
print("\nПосле обратного декодирования:")
print(list(decoded))

print("\nСовпадает с оригиналом?", np.array_equal(decoded, dc_values))

