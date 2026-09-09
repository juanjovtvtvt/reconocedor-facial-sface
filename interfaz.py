"""Interfaz, voz, registro guiado y estabilización temporal."""

from __future__ import annotations

import queue
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

from base_sface import Resultado


MUESTRAS_POR_REGISTRO = 30
MUESTRAS_POR_ANGULO = 10
VENTANA_VOTOS = 7
VOTOS_CONFIRMACION = 5
COOLDOWN_VOZ = 10


def texto(
    imagen: np.ndarray,
    mensaje: str,
    posicion: tuple[int, int],
    color: tuple[int, int, int] = (255, 255, 255),
    escala: float = 0.55,
) -> None:
    cv2.putText(
        imagen, mensaje, posicion, cv2.FONT_HERSHEY_SIMPLEX,
        escala, (0, 0, 0), 3, cv2.LINE_AA,
    )
    cv2.putText(
        imagen, mensaje, posicion, cv2.FONT_HERSHEY_SIMPLEX,
        escala, color, 1, cv2.LINE_AA,
    )


def dibujar_partes(frame: np.ndarray, resultado_mesh) -> None:
    alto, ancho = frame.shape[:2]
    estilo = mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
    conexiones = mp.solutions.face_mesh.FACEMESH_TESSELATION
    for rostro in resultado_mesh.multi_face_landmarks or []:
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            rostro,
            conexiones,
            landmark_drawing_spec=None,
            connection_drawing_spec=estilo,
        )
        puntos = rostro.landmark
        for nombre, indice in {
            "Frente": 10,
            "Nariz": 1,
            "Menton": 152,
            "Ojo izq.": 33,
            "Ojo der.": 263,
            "Boca": 13,
        }.items():
            x = int(puntos[indice].x * ancho)
            y = int(puntos[indice].y * alto)
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)
            texto(frame, nombre, (x + 5, y - 5), (0, 255, 255))


def pedir_nombre() -> Optional[str]:
    try:
        import tkinter as tk
        from tkinter import simpledialog

        raiz = tk.Tk()
        raiz.withdraw()
        raiz.attributes("-topmost", True)
        nombre = simpledialog.askstring(
            "Registrar o mejorar persona",
            "Escribe el nombre exacto. Si ya existe, mejorará ese perfil:",
            parent=raiz,
        )
        raiz.destroy()
        return nombre.strip() if nombre and nombre.strip() else None
    except Exception:
        nombre = input("Nombre de la persona (vacío para cancelar): ").strip()
        return nombre or None


class Voz:
    """Un único hilo de voz evita anuncios simultáneos."""

    def __init__(self) -> None:
        self.ultimos: dict[str, float] = {}
        self.cola: queue.Queue[str] = queue.Queue()
        threading.Thread(target=self._bucle, daemon=True).start()

    def anunciar(self, nombre: str) -> None:
        ahora = time.monotonic()
        if ahora - self.ultimos.get(nombre, 0.0) < COOLDOWN_VOZ:
            return
        self.ultimos[nombre] = ahora
        self.cola.put(f"Hola, {nombre}")

    def _bucle(self) -> None:
        try:
            motor = pyttsx3.init()
            for voz in motor.getProperty("voices"):
                idiomas = " ".join(str(x) for x in getattr(voz, "languages", []))
                if "spanish" in voz.name.lower() or "es" in idiomas.lower():
                    motor.setProperty("voice", voz.id)
                    break
            motor.setProperty("rate", 165)
            while True:
                motor.say(self.cola.get())
                motor.runAndWait()
        except Exception as error:
            print(f"La voz no pudo iniciarse: {error}")


@dataclass
class SesionRegistro:
    persona_id: str
    nombre: str
    carpeta: Path
    cantidad: int = 0
    ultimo_guardado: float = 0.0
    signo_primer_lado: Optional[float] = None
    mensaje: str = "Mira al frente"
    vectores: list[np.ndarray] = field(default_factory=list)

    @property
    def terminada(self) -> bool:
        return self.cantidad >= MUESTRAS_POR_REGISTRO

    def instruccion(self) -> str:
        if self.cantidad < MUESTRAS_POR_ANGULO:
            return "Mira al frente"
        if self.cantidad < 2 * MUESTRAS_POR_ANGULO:
            return "Gira un poco hacia un lado"
        return "Ahora gira hacia el lado contrario"

    def _pose_correcta(self, giro: float) -> bool:
        if self.cantidad < MUESTRAS_POR_ANGULO:
            return abs(giro) <= 0.18
        if self.cantidad < 2 * MUESTRAS_POR_ANGULO:
            if abs(giro) < 0.10:
                return False
            if self.signo_primer_lado is None:
                self.signo_primer_lado = 1.0 if giro >= 0 else -1.0
            return giro * self.signo_primer_lado >= 0.08
        if self.signo_primer_lado is None:
            return abs(giro) >= 0.10
        return giro * self.signo_primer_lado <= -0.08

    def intentar_guardar(
        self,
        alineada: np.ndarray,
        vector: np.ndarray,
        calidad_ok: bool,
        mensaje_calidad: str,
        giro: float,
    ) -> bool:
        ahora = time.monotonic()
        self.mensaje = mensaje_calidad if not calidad_ok else self.instruccion()
        if not calidad_ok or ahora - self.ultimo_guardado < 0.18:
            return False
        if not self._pose_correcta(giro):
            self.mensaje = self.instruccion()
            return False
        if self.vectores:
            maxima = max(float(v @ vector) for v in self.vectores[-8:])
            if maxima > 0.9995:
                self.mensaje = "Mueve levemente la cabeza"
                return False

        ruta = self.carpeta / f"{time.time_ns()}_{self.cantidad:02d}.png"
        if not cv2.imwrite(str(ruta), alineada):
            self.mensaje = "No se pudo guardar la muestra"
            return False
        self.vectores.append(vector.copy())
        self.cantidad += 1
        self.ultimo_guardado = ahora
        self.mensaje = self.instruccion() if not self.terminada else "Registro completo"
        return True


@dataclass
class Pista:
    pista_id: int
    caja: tuple[int, int, int, int]
    historial: deque[tuple[Optional[str], float]] = field(
        default_factory=lambda: deque(maxlen=VENTANA_VOTOS)
    )
    confirmado: Optional[str] = None
    score_confirmado: float = 0.0
    faltantes: int = 0
    actual: Optional[Resultado] = None
    recien_confirmado: bool = False


def _centro(caja: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = caja
    return x + w / 2.0, y + h / 2.0


class EstabilizadorTemporal:
    """No muestra ni anuncia un nombre hasta acumular 5 de 7 votos."""

    def __init__(self) -> None:
        self.pistas: dict[int, Pista] = {}
        self.siguiente = 1

    def limpiar(self) -> None:
        self.pistas.clear()

    def _buscar_pista(
        self,
        caja: tuple[int, int, int, int],
        disponibles: set[int],
    ) -> Optional[int]:
        cx, cy = _centro(caja)
        mejor_id: Optional[int] = None
        mejor_distancia = float("inf")
        for pista_id in disponibles:
            pista = self.pistas[pista_id]
            px, py = _centro(pista.caja)
            escala = max(caja[2], caja[3], pista.caja[2], pista.caja[3], 1)
            distancia = ((cx - px) ** 2 + (cy - py) ** 2) ** 0.5 / escala
            if distancia < 0.75 and distancia < mejor_distancia:
                mejor_id = pista_id
                mejor_distancia = distancia
        return mejor_id

    def _votar(self, pista: Pista, resultado: Resultado) -> None:
        pista.actual = resultado
        nombre = resultado.nombre
        pista.historial.append((nombre, resultado.similitud))
        pista.recien_confirmado = False
        nombres = [n for n, _ in pista.historial if n]
        if nombres:
            candidato, votos = Counter(nombres).most_common(1)[0]
            if votos >= VOTOS_CONFIRMACION:
                scores = [s for n, s in pista.historial if n == candidato]
                if pista.confirmado != candidato:
                    pista.confirmado = candidato
                    pista.recien_confirmado = True
                pista.score_confirmado = float(np.mean(scores))
                return
        if pista.confirmado and sum(n is None for n, _ in pista.historial) >= VOTOS_CONFIRMACION:
            pista.confirmado = None
            pista.score_confirmado = 0.0

    def actualizar(
        self,
        observaciones: list[tuple[tuple[int, int, int, int], Resultado]],
    ) -> list[Pista]:
        for pista in self.pistas.values():
            pista.faltantes += 1
        disponibles = set(self.pistas)
        visibles: list[Pista] = []
        for caja, resultado in observaciones:
            pista_id = self._buscar_pista(caja, disponibles)
            if pista_id is None:
                pista_id = self.siguiente
                self.siguiente += 1
                self.pistas[pista_id] = Pista(pista_id=pista_id, caja=caja)
            else:
                disponibles.remove(pista_id)
            pista = self.pistas[pista_id]
            pista.caja = caja
            pista.faltantes = 0
            self._votar(pista, resultado)
            visibles.append(pista)
        self.pistas = {i: p for i, p in self.pistas.items() if p.faltantes <= 12}
        return visibles
