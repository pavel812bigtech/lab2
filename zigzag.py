import numpy as np




def zigzag_scan(square_matrix):
    """Зигзаг-обход квадратной матрицы NxN"""
    N = square_matrix.shape[0]
    result = []

    for sum_diag in range(2 * N - 1):  # сумма индексов по диагоналям
        diag = []
        for i in range(max(0, sum_diag - N + 1), min(sum_diag + 1, N)):
            j = sum_diag - i
            diag.append(square_matrix[i, j])

        # Чередуем направление: чётные диагонали — в одну сторону, нечётные — в другую
        if sum_diag % 2 == 0:
            result.extend(diag[::-1])  # разворачиваем
        else:
            result.extend(diag)

    return np.array(result)



def zigzag_scan_rect(matrix):
    """Зигзаг-обход прямоугольной матрицы N×M"""
    N, M = matrix.shape
    result = []

    for sum_diag in range(N + M - 1):
        diag = []
        for i in range(max(0, sum_diag - M + 1), min(sum_diag + 1, N)):
            j = sum_diag - i
            diag.append(matrix[i, j])

        if sum_diag % 2 == 0:
            result.extend(diag[::-1])
        else:
            result.extend(diag)

    return np.array(result)


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

# Тест на маленькой матрице
test = np.arange(16).reshape(4, 4)
print("Исходная матрица 4x4:")
print(test)

print("\nZigzag-обход (квадратная):")
print(zigzag_scan(test))

# Тест на прямоугольной
test_rect = np.arange(12).reshape(3, 4)
print("\nИсходная матрица 3x4:")
print(test_rect)

print("\nZigzag-обход (прямоугольная 3x4):")
print(zigzag_scan_rect(test_rect))

