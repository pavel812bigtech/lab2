from typing import List, Union

print("=== ЛАБА: Линейный сплайн (кусочно-линейная интерполяция) ===\n")


# =============================================
# 1. Уже готовая функция линейной интерполяции
# =============================================
def linear_interpolation(x1, y1, x2, y2, x):
    """Линейная интерполяция между двумя точками"""
    if x2 - x1 == 0:
        return y1
    return y1 + (y2 - y1) * (x - x1) / (x2 - x1)


# =============================================
# 2. ФУНКЦИЯ ЛИНЕЙНОГО СПЛАЙНА (главная)
# =============================================
def linear_spline(x_nodes: List[float], y_nodes: List[float], x: float) -> float:
    """
    Вычисляет значение линейного сплайна в точке x

    x_nodes — список координат x (узлы): [x1, x2, ..., xn]
    y_nodes — список значений y:         [y1, y2, ..., yn]
    x       — точка, в которой ищем значение
    """

    # Проверки
    if len(x_nodes) != len(y_nodes):
        raise ValueError("Количество x и y должно быть одинаковым!")

    if len(x_nodes) < 2:
        raise ValueError("Должно быть минимум 2 узла для сплайна")

    # Проверка, что x_nodes отсортированы по возрастанию
    for i in range(1, len(x_nodes)):
        if x_nodes[i] < x_nodes[i - 1]:
            raise ValueError("Узлы x должны быть отсортированы по возрастанию!")

    # Если x точно совпадает с одним из узлов — возвращаем значение
    for i in range(len(x_nodes)):
        if abs(x_nodes[i] - x) < 1e-9:  # небольшая погрешность на float
            return y_nodes[i]

    # Если x вне диапазона
    if x < x_nodes[0] or x > x_nodes[-1]:
        print(f"⚠️ Предупреждение: x = {x} выходит за пределы [{x_nodes[0]}, {x_nodes[-1]}]")
        # Можно вернуть ближайшее значение (экстраполяция), но по условию — лучше предупредить
        if x < x_nodes[0]:
            return y_nodes[0]
        else:
            return y_nodes[-1]

    # Находим, между какими двумя узлами лежит x
    for i in range(len(x_nodes) - 1):
        if x_nodes[i] <= x <= x_nodes[i + 1]:
            # Применяем линейную интерполяцию к этому отрезку
            return linear_interpolation(
                x_nodes[i], y_nodes[i],
                x_nodes[i + 1], y_nodes[i + 1],
                x
            )

    # Если по какой-то причине не нашли (не должно произойти)
    return y_nodes[-1]


# =============================================
# ТЕСТИРОВАНИЕ
# =============================================

print("Тесты линейного сплайна:\n")

# Тест 1
x_nodes = [0, 2, 5, 8, 10]
y_nodes = [0, 4, 10, 3, 7]

print("Узлы:", list(zip(x_nodes, y_nodes)))
print("Значение в x=1   →", linear_spline(x_nodes, y_nodes, 1))
print("Значение в x=3   →", linear_spline(x_nodes, y_nodes, 3))
print("Значение в x=6   →", linear_spline(x_nodes, y_nodes, 6))
print("Значение в x=9   →", linear_spline(x_nodes, y_nodes, 9))
print("Значение в x=5   →", linear_spline(x_nodes, y_nodes, 5))  # точно в узле

print("\n=== Функция линейного сплайна готова ===")