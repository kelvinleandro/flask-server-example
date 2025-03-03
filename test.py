from app import aplicar_otsu, apply_window, carregar_imagem
import cv2
import matplotlib.pyplot as plt

img = carregar_imagem(r"C:\Users\Kelvi\Downloads\90.dcm")
imagem_hu = apply_window(img)

imagem_suavizada = cv2.GaussianBlur(imagem_hu, (5, 5), 0)
_, mascara_pulmao = cv2.threshold(
    imagem_suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
)

# Obter tanto a imagem com contornos quanto o dicionário de contornos válidos
imagem_contornos_somente, contornos_validos_dict = aplicar_otsu(imagem_hu)

print(contornos_validos_dict.keys())

# Plotar as imagens
plt.figure(figsize=(5, 5))
plt.imshow(imagem_hu, cmap="gray")
plt.axis("off")
plt.title("Imagem Original")

plt.figure(figsize=(5, 5))
plt.imshow(imagem_contornos_somente, cmap="gray")
plt.axis("off")
plt.title("Contorno Original")

plt.show()
