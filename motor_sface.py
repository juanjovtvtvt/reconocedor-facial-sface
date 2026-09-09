"""Detección YuNet y extracción de características SFace."""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


RAIZ = Path(__file__).resolve().parent
MODELOS = RAIZ / "modelos"
MODELO_YUNET = MODELOS / "face_detection_yunet_2023mar.onnx"
MODELO_SFACE = MODELOS / "face_recognition_sface_2021dec.onnx"
MAX_ANCHO_DETECCION = 640


def normalizar_vector(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    return vector / (float(np.linalg.norm(vector)) + 1e-12)


def normalizar_nombre(nombre: str) -> str:
    valor = unicodedata.normalize("NFKD", nombre.casefold().strip())
    return "".join(c for c in valor if c.isalnum())


def nombres_parecidos(a: str, b: str) -> bool:
    a, b = normalizar_nombre(a), normalizar_nombre(b)
    if not a or not b:
        return False
    corto, largo = sorted((a, b), key=len)
    return (
        (len(corto) >= 4 and largo.startswith(corto))
        or SequenceMatcher(None, a, b).ratio() >= 0.58
    )


class MotorFacial:
    """Convierte cada rostro alineado en un vector SFace de 128 valores."""

    def __init__(self) -> None:
        faltantes = [p.name for p in (MODELO_YUNET, MODELO_SFACE) if not p.exists()]
        if faltantes:
            raise FileNotFoundError(
                "Faltan modelos en la carpeta modelos: " + ", ".join(faltantes)
            )
        self.detector = cv2.FaceDetectorYN.create(
            str(MODELO_YUNET), "", (320, 320), 0.88, 0.30, 5000
        )
        self.reconocedor = cv2.FaceRecognizerSF.create(str(MODELO_SFACE), "")

    def detectar(self, imagen: np.ndarray) -> list[np.ndarray]:
        alto, ancho = imagen.shape[:2]
        escala = min(1.0, MAX_ANCHO_DETECCION / float(ancho))
        if escala < 1.0:
            entrada = cv2.resize(
                imagen,
                (int(round(ancho * escala)), int(round(alto * escala))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            entrada = imagen
        entrada_alto, entrada_ancho = entrada.shape[:2]
        self.detector.setInputSize((entrada_ancho, entrada_alto))
        _, caras = self.detector.detect(entrada)
        if caras is None:
            return []
        resultado = [fila.copy() for fila in caras]
        if escala < 1.0:
            for cara in resultado:
                cara[:14] /= escala
        resultado.sort(key=lambda f: float(f[2] * f[3]), reverse=True)
        return resultado

    def alinear_y_extraer(
        self, imagen: np.ndarray, cara: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        alineada = self.reconocedor.alignCrop(imagen, cara)
        vector = self.reconocedor.feature(alineada)
        return alineada, normalizar_vector(vector)

    def extraer_alineada(self, imagen: np.ndarray) -> np.ndarray:
        if imagen.ndim == 2:
            imagen = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
        imagen = cv2.resize(imagen, (112, 112), interpolation=cv2.INTER_AREA)
        return normalizar_vector(self.reconocedor.feature(imagen))

    def extraer_archivo(self, ruta: Path, ya_alineada: bool) -> Optional[np.ndarray]:
        imagen = cv2.imread(str(ruta), cv2.IMREAD_COLOR)
        if imagen is None:
            return None
        if ya_alineada:
            return self.extraer_alineada(imagen)
        caras = self.detectar(imagen)
        if not caras:
            return self.extraer_alineada(imagen)
        _, vector = self.alinear_y_extraer(imagen, caras[0])
        return vector

    @staticmethod
    def caja(cara: np.ndarray) -> tuple[int, int, int, int]:
        return tuple(int(round(float(v))) for v in cara[:4])

    @staticmethod
    def evaluar_calidad(cara: np.ndarray, alineada: np.ndarray) -> tuple[bool, str, float]:
        gris = cv2.cvtColor(alineada, cv2.COLOR_BGR2GRAY)
        brillo = float(np.mean(gris))
        nitidez = float(cv2.Laplacian(gris, cv2.CV_64F).var())
        tamano = min(float(cara[2]), float(cara[3]))
        ojo_a, ojo_b = cara[4:6], cara[6:8]
        nariz = cara[8:10]
        distancia_ojos = float(np.linalg.norm(ojo_b - ojo_a)) + 1e-6
        centro_ojos = (ojo_a + ojo_b) / 2.0
        giro = float((nariz[0] - centro_ojos[0]) / distancia_ojos)
        inclinacion = abs(math_degrees(ojo_b[1] - ojo_a[1], ojo_b[0] - ojo_a[0]))

        if tamano < 95:
            return False, "Acércate un poco a la cámara", giro
        if brillo < 38:
            return False, "Falta luz en el rostro", giro
        if brillo > 220:
            return False, "Hay demasiada luz en el rostro", giro
        if nitidez < 28:
            return False, "Quédate quieta: la imagen está borrosa", giro
        if inclinacion > 22:
            return False, "Endereza un poco la cabeza", giro
        if abs(giro) > 0.52:
            return False, "Gira un poco hacia el frente", giro
        return True, "Calidad correcta", giro


def math_degrees(y: float, x: float) -> float:
    return float(np.degrees(np.arctan2(y, x)))
