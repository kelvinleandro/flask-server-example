from flask import Flask, request, jsonify
from flask_cors import CORS
import pydicom
import numpy as np
import random
import math
import cv2

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

def aplicar_otsu(imagem_cinza: np.ndarray) -> np.ndarray:
    """
    Aplica o algoritmo de Otsu para segmentação dos pulmões em imagens através da biblioteca cv2.

    Parâmetros:
        imagem_cinza (np.ndarray): Pixels da imagem de entrada em escala de cinza.

    Retorna:
        np.ndarray: Pixels da imagem original com os contornos dos pulmões
        destacados em vermelho.
    """

    # Aplica um filtro Gaussiano para suavizar a imagem
    imagem_suavizada = cv2.GaussianBlur(imagem_cinza, (5,5), 0)

    # Aplica threshold de Otsu para segmentação
    _, mascara_pulmao = cv2.threshold(imagem_suavizada, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Encontra contornos
    contornos, _ = cv2.findContours(mascara_pulmao, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contornos_dict = {f"contorno_{i}": contorno.squeeze().tolist() for i, contorno in enumerate(contornos)}

    return contornos_dict

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
        contornos_dict = aplicar_otsu(dicom_data)

        return jsonify(contornos_dict), 200
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()