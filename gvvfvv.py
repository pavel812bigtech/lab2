from PIL import Image, ImageDraw

# Создаем тестовое изображение с цветными полосами
img = Image.new('RGB', (512, 512))
draw = ImageDraw.Draw(img)

# Цветные полосы
colors = [(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255)]
bar_width = 512 // len(colors)

for i, color in enumerate(colors):
    draw.rectangle([i*bar_width, 0, (i+1)*bar_width, 512], fill=color)

img.save('color_bars.png')
print("Создано изображение color_bars.png")