from flask import Flask, request, jsonify
from flask_cors import CORS
import pydicom
import numpy as np
import base64
import cv2
from random import random


def carregar_imagem(input_path: str) -> np.ndarray:
    """
    Carrega imagem do tipo DICOM e retorna a imagem em Hounsfield Units (HU).

    args:
        input_path: str - Caminho para a imagem DICOM
    return:
        np.ndarray - Imagem em Hounsfield Units
    """
    ds = pydicom.dcmread(input_path)

    assert ds.Modality == "CT", "Somente imagens do tipo CT são suportadas"

    rescale_intercept = getattr(ds, "RescaleIntercept", 0)
    rescale_slope = getattr(ds, "RescaleSlope", 1)

    hu_image = ds.pixel_array * rescale_slope + rescale_intercept
    return hu_image


def apply_window(image, img_min=-1000, img_max=2000):
    image = np.clip(image, img_min, img_max)

    image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)

    return image


def aplicar_otsu(imagem_cinza: np.ndarray) -> tuple:
    """
    Aplica o algoritmo de Otsu para segmentação dos pulmões em imagens.

    Parâmetros:
        imagem_cinza (np.ndarray): Pixels da imagem de entrada em escala de cinza.

    Retorna:
        tuple:
            - Imagem original com os contornos dos pulmões destacados em vermelho.
            - Imagem com apenas os contornos dos pulmões em branco sobre fundo preto.
    """
    # Aplicar um filtro Gaussiano para suavizar a imagem
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5, 5), 0)

    # Aplicar threshold de Otsu para segmentação
    _, mascara_pulmao = cv2.threshold(
        imagem_suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return remove_fundo(mascara_pulmao)


def remove_fundo(
    mascara: np.ndarray, area_minima: int = 3000, area_maxima: int = 40000
) -> tuple:
    """
    Mantém apenas contornos fechados cujas áreas estão dentro do intervalo especificado e que não tocam a borda da imagem.

    Parâmetros:
        mascara (np.ndarray): Máscara binária com os contornos.
        area_minima (int): Área mínima permitida para os contornos (default: 3000).
        area_maxima (int): Área máxima permitida para os contornos (default: 40000).

    Retorna:
        tuple:
            - np.ndarray: Imagem com os novos contornos preenchidos em vermelho.
            - dict: Dicionário onde cada chave é uma string (e.g., "contorno_0") e o valor é o contorno válido.
    """
    # Encontrar contornos na máscara
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Criar uma imagem para os contornos vermelhos
    pulmao_contornado = np.zeros(
        (mascara.shape[0], mascara.shape[1], 3), dtype=np.uint8
    )  # Imagem em preto

    # Dicionário para armazenar os contornos válidos
    contornos_validos_dict = {}

    # Obter as dimensões da imagem
    altura, largura = mascara.shape

    # Filtrar e desenhar apenas os contornos fechados que não tocam a borda e têm áreas dentro do intervalo especificado
    for i, contorno in enumerate(contornos):
        # Verificar se o contorno é fechado
        if (
            cv2.arcLength(contorno, True) > 0
        ):  # Verifica se o contorno tem comprimento positivo
            # Verificar se o contorno toca a borda da imagem
            toca_borda = False
            for ponto in contorno:
                x, y = ponto[0]
                if x == 0 or x == largura - 1 or y == 0 or y == altura - 1:
                    toca_borda = True
                    break

            # Se o contorno não tocar a borda e tiver área dentro do intervalo, desenhar em vermelho e adicionar ao dicionário
            if not toca_borda:
                area = cv2.contourArea(contorno)
                if area_minima <= area <= area_maxima:
                    # Desenhar o contorno válido em vermelho na imagem de contornos vermelhos
                    cv2.drawContours(
                        pulmao_contornado, [contorno], -1, (0, 0, 255), 2
                    )  # Vermelho
                    # Adicionar o contorno ao dicionário de contornos válidos
                    contornos_validos_dict[f"contorno_{i}"] = (
                        contorno.squeeze().tolist()
                    )

    return pulmao_contornado, contornos_validos_dict


app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return "Hello, Flask!"


@app.route("/upload", methods=["POST"])
def upload_file():
    if "dicom" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["dicom"]

    try:
        dicom_data = carregar_imagem(file)
        dicom_data = apply_window(dicom_data)

        # Processa a imagem e obtém os contornos
        _, todos_contornos = aplicar_otsu(dicom_data)
        contornos_validos = {
            key: value for key, value in todos_contornos.items() if random() < 0.7
        }

        rotated_image = cv2.rotate(dicom_data, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # Convert the rotated image to Base64
        _, buffer = cv2.imencode(".png", rotated_image)
        base64_image = base64.b64encode(buffer).decode("utf-8")

        return (
            jsonify(
                {
                    "preprocessed": base64_image,
                    "valid_contours": contornos_validos,
                    "todos_contornos": todos_contornos,
                }
            ),
            200,
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
