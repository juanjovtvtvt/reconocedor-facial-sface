"""Interfaz gráfica para el reconocedor facial local SFace."""

from __future__ import annotations

import argparse
import queue
import threading
import tkinter as tk
import time
from dataclasses import dataclass
from typing import Optional

import cv2
import customtkinter as ctk
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk

from base_sface import BaseRostrosSFace, Resultado
from interfaz import EstabilizadorTemporal, SesionRegistro, Voz, dibujar_partes, texto
from motor_sface import MotorFacial


COLOR_FONDO = "#090D16"
COLOR_PANEL = "#111827"
COLOR_TARJETA = "#182235"
COLOR_PRIMARIO = "#2F80ED"
COLOR_EXITO = "#21C58E"
COLOR_AVISO = "#F5B942"
COLOR_ERROR = "#FF5A70"
COLOR_TEXTO = "#F4F7FB"
COLOR_TEXTO_SEC = "#94A3B8"


@dataclass
class Observacion:
    cara: np.ndarray
    caja: tuple[int, int, int, int]
    alineada: np.ndarray
    vector: np.ndarray
    resultado: Resultado


def abrir_camara() -> cv2.VideoCapture:
    camara = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camara.isOpened():
        camara.release()
        camara = cv2.VideoCapture(0)
    if not camara.isOpened():
        raise RuntimeError(
            "No se pudo abrir la cámara. Revisa los permisos de Windows y "
            "cierra otras aplicaciones que la estén usando."
        )
    camara.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camara.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    camara.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return camara


def crear_observaciones(
    frame: np.ndarray,
    motor: MotorFacial,
    base: BaseRostrosSFace,
) -> list[Observacion]:
    observaciones: list[Observacion] = []
    for cara in motor.detectar(frame):
        try:
            alineada, vector = motor.alinear_y_extraer(frame, cara)
        except cv2.error:
            continue
        observaciones.append(
            Observacion(
                cara=cara,
                caja=motor.caja(cara),
                alineada=alineada,
                vector=vector,
                resultado=base.identificar(vector),
            )
        )
    return observaciones


def dibujar_pistas(frame, pistas, voz: Voz, registrando: bool) -> Optional[str]:
    identidad_principal: Optional[str] = None
    for pista in pistas:
        x, y, w, h = pista.caja
        if pista.confirmado:
            color = (142, 197, 33)
            etiqueta = f"{pista.confirmado} | similitud {pista.score_confirmado * 100:.0f}"
            identidad_principal = identidad_principal or pista.confirmado
            if pista.recien_confirmado and not registrando:
                voz.anunciar(pista.confirmado)
        elif pista.actual and pista.actual.aceptado:
            color = (66, 185, 245)
            etiqueta = "Verificando identidad..."
        else:
            color = (112, 90, 255)
            etiqueta = (
                "Resultado ambiguo"
                if pista.actual and pista.actual.motivo == "ambiguo"
                else "Desconocido"
            )
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        texto(frame, etiqueta, (x, max(25, y - 10)), color)
    return identidad_principal


class AplicacionFacial(ctk.CTk):
    def __init__(self, reconstruir: bool = False) -> None:
        super().__init__(fg_color=COLOR_FONDO)
        escala_dpi = max(float(self._get_window_scaling()), 1.0)
        ctk.set_window_scaling(1.0 / escala_dpi)
        ctk.set_widget_scaling(1.0 / escala_dpi)
        self.title("Face Local · Reconocimiento facial privado")
        ancho = min(1180, max(1000, self.winfo_screenwidth() - 80))
        alto = min(650, max(600, self.winfo_screenheight() - 120))
        self.geometry(f"{ancho}x{alto}+0+0")
        self.minsize(min(1000, ancho), min(600, alto))

        self.motor = MotorFacial()
        self.base = BaseRostrosSFace(self.motor, reconstruir=reconstruir)
        self.voz = Voz()
        self.estabilizador = EstabilizadorTemporal()
        self.malla = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=5,
            refine_landmarks=True,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self.camara: Optional[cv2.VideoCapture] = None
        self.camara_activa = False
        self.cerrando = False
        self.actualizando_base = False
        self.reanudar_tras_actualizar = False
        self.sesion: Optional[SesionRegistro] = None
        self.imagen_tk: Optional[ImageTk.PhotoImage] = None
        self.eventos: queue.Queue[tuple[str, object]] = queue.Queue()
        self.contador_frames = 0
        self.ultimo_resultado_malla = None
        self.ultimo_tiempo_frame = time.perf_counter()
        self.fps_suavizados = 0.0

        self._crear_interfaz()
        self._refrescar_personas()
        self.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.bind("<KeyPress-r>", lambda _e: self.registrar())
        self.bind("<KeyPress-t>", lambda _e: self.reconstruir_base())
        self.bind("<Escape>", lambda _e: self.cerrar())
        self.after(250, self.iniciar_camara)
        self.after(150, self._procesar_eventos)

    @staticmethod
    def _tarjeta(padre, **opciones):
        return ctk.CTkFrame(
            padre,
            fg_color=COLOR_TARJETA,
            corner_radius=16,
            border_width=1,
            border_color="#243249",
            **opciones,
        )

    def _crear_interfaz(self) -> None:
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        encabezado = ctk.CTkFrame(self, fg_color="transparent", height=92)
        encabezado.grid(row=0, column=0, sticky="ew", padx=28, pady=(18, 8))
        encabezado.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            encabezado,
            text="FACE·LOCAL",
            text_color=COLOR_PRIMARIO,
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, rowspan=2, padx=(0, 24))
        ctk.CTkLabel(
            encabezado,
            text="Reconocimiento facial inteligente",
            text_color=COLOR_TEXTO,
            font=ctk.CTkFont(size=28, weight="bold"),
            anchor="w",
        ).grid(row=0, column=1, sticky="sw")
        ctk.CTkLabel(
            encabezado,
            text="SFace · confirmación temporal · datos almacenados en este equipo",
            text_color=COLOR_TEXTO_SEC,
            font=ctk.CTkFont(size=13),
            anchor="w",
        ).grid(row=1, column=1, sticky="nw")
        ctk.CTkLabel(
            encabezado,
            text="●  PROCESAMIENTO LOCAL",
            text_color=COLOR_EXITO,
            fg_color="#102B28",
            corner_radius=12,
            padx=14,
            pady=8,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).grid(row=0, column=2, rowspan=2, padx=(20, 0))

        contenido = ctk.CTkFrame(self, fg_color="transparent")
        contenido.grid(row=1, column=0, sticky="nsew", padx=28, pady=(0, 24))
        contenido.grid_columnconfigure(0, weight=1)
        contenido.grid_columnconfigure(1, weight=0, minsize=335)
        contenido.grid_rowconfigure(0, weight=1)

        video = self._tarjeta(contenido)
        video.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        video.grid_columnconfigure(0, weight=1)
        video.grid_rowconfigure(1, weight=1)
        ctk.CTkLabel(
            video,
            text="CÁMARA EN VIVO",
            text_color=COLOR_TEXTO_SEC,
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=18, pady=(14, 10))
        self.video_label = tk.Label(
            video,
            bg="#05070C",
            fg="#64748B",
            text="Iniciando cámara...",
            font=("Segoe UI", 14),
            borderwidth=0,
        )
        self.video_label.grid(row=1, column=0, sticky="nsew", padx=12)
        self.video_estado = ctk.CTkLabel(
            video,
            text="● Cámara desconectada",
            text_color=COLOR_TEXTO_SEC,
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self.video_estado.grid(row=2, column=0, sticky="ew", padx=18, pady=12)

        self.panel = ctk.CTkScrollableFrame(
            contenido,
            width=320,
            fg_color=COLOR_PANEL,
            corner_radius=16,
            scrollbar_button_color="#334155",
        )
        self.panel.grid(row=0, column=1, sticky="nsew")
        self.panel.grid_columnconfigure(0, weight=1)
        self._crear_panel_control()

    def _crear_panel_control(self) -> None:
        ctk.CTkLabel(
            self.panel,
            text="ESTADO DEL SISTEMA",
            text_color=COLOR_TEXTO_SEC,
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=4, pady=(4, 8))

        estado = self._tarjeta(self.panel)
        estado.grid(row=1, column=0, sticky="ew", pady=(0, 14))
        estado.grid_columnconfigure(0, weight=1)
        self.estado_label = ctk.CTkLabel(
            estado,
            text="Preparando reconocimiento",
            text_color=COLOR_AVISO,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        self.estado_label.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        self.rostros_label = ctk.CTkLabel(
            estado,
            text="0 rostros detectados",
            text_color=COLOR_TEXTO_SEC,
            font=ctk.CTkFont(size=12),
            anchor="w",
        )
        self.rostros_label.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))

        identidad = self._tarjeta(self.panel)
        identidad.grid(row=2, column=0, sticky="ew", pady=(0, 14))
        identidad.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            identidad,
            text="IDENTIDAD CONFIRMADA",
            text_color=COLOR_TEXTO_SEC,
            font=ctk.CTkFont(size=10, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 2))
        self.identidad_label = ctk.CTkLabel(
            identidad,
            text="—",
            text_color=COLOR_TEXTO,
            font=ctk.CTkFont(size=25, weight="bold"),
            anchor="w",
        )
        self.identidad_label.grid(row=1, column=0, sticky="ew", padx=16)
        self.similitud_label = ctk.CTkLabel(
            identidad,
            text="Esperando una coincidencia estable",
            text_color=COLOR_TEXTO_SEC,
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self.similitud_label.grid(row=2, column=0, sticky="ew", padx=16, pady=(2, 14))

        registro = self._tarjeta(self.panel)
        registro.grid(row=3, column=0, sticky="ew", pady=(0, 14))
        registro.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(
            registro,
            text="REGISTRO GUIADO",
            text_color=COLOR_TEXTO_SEC,
            font=ctk.CTkFont(size=10, weight="bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 4))
        self.registro_label = ctk.CTkLabel(
            registro,
            text="Sin registro activo",
            text_color=COLOR_TEXTO,
            font=ctk.CTkFont(size=13, weight="bold"),
            anchor="w",
        )
        self.registro_label.grid(row=1, column=0, sticky="ew", padx=16)
        self.registro_detalle = ctk.CTkLabel(
            registro,
            text="La app te guiará por tres ángulos.",
            text_color=COLOR_TEXTO_SEC,
            font=ctk.CTkFont(size=11),
            anchor="w",
            wraplength=285,
            justify="left",
        )
        self.registro_detalle.grid(row=2, column=0, sticky="ew", padx=16, pady=(2, 10))
        self.progreso = ctk.CTkProgressBar(
            registro,
            height=10,
            corner_radius=5,
            progress_color=COLOR_PRIMARIO,
            fg_color="#263247",
        )
        self.progreso.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 16))
        self.progreso.set(0)

        self.boton_registrar = ctk.CTkButton(
            self.panel,
            text="＋  Registrar o mejorar persona",
            command=self.registrar,
            height=44,
            corner_radius=12,
            fg_color=COLOR_PRIMARIO,
            hover_color="#2568C1",
            font=ctk.CTkFont(size=13, weight="bold"),
        )
        self.boton_registrar.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        self.boton_reconstruir = ctk.CTkButton(
            self.panel,
            text="↻  Recalibrar base facial",
            command=self.reconstruir_base,
            height=40,
            corner_radius=12,
            fg_color="#263247",
            hover_color="#334155",
            border_width=1,
            border_color="#3B4A63",
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.boton_reconstruir.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        self.boton_camara = ctk.CTkButton(
            self.panel,
            text="Pausar cámara",
            command=self.alternar_camara,
            height=38,
            corner_radius=12,
            fg_color="transparent",
            hover_color="#1E293B",
            border_width=1,
            border_color="#3B4A63",
            text_color=COLOR_TEXTO_SEC,
        )
        self.boton_camara.grid(row=6, column=0, sticky="ew", pady=(0, 18))

        ctk.CTkLabel(
            self.panel,
            text="PERSONAS EN LA BASE",
            text_color=COLOR_TEXTO_SEC,
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w",
        ).grid(row=7, column=0, sticky="ew", padx=4, pady=(0, 8))
        self.personas_contenedor = ctk.CTkFrame(
            self.panel, fg_color="transparent"
        )
        self.personas_contenedor.grid(row=8, column=0, sticky="ew")
        self.personas_contenedor.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            self.panel,
            text="Privacidad: imágenes y vectores permanecen únicamente en este equipo.",
            text_color="#64748B",
            font=ctk.CTkFont(size=10),
            wraplength=300,
            justify="left",
        ).grid(row=9, column=0, sticky="ew", padx=4, pady=(18, 10))

    def _refrescar_personas(self) -> None:
        for elemento in self.personas_contenedor.winfo_children():
            elemento.destroy()
        if not self.base.grupos:
            ctk.CTkLabel(
                self.personas_contenedor,
                text="Todavía no hay personas registradas.",
                text_color=COLOR_TEXTO_SEC,
                font=ctk.CTkFont(size=11),
            ).grid(row=0, column=0, sticky="w", padx=4)
            return
        for fila, grupo in enumerate(self.base.grupos):
            tarjeta = ctk.CTkFrame(
                self.personas_contenedor,
                fg_color=COLOR_TARJETA,
                corner_radius=10,
            )
            tarjeta.grid(row=fila, column=0, sticky="ew", pady=3)
            tarjeta.grid_columnconfigure(1, weight=1)
            ctk.CTkLabel(
                tarjeta,
                text="●",
                text_color=COLOR_EXITO,
                font=ctk.CTkFont(size=12),
            ).grid(row=0, column=0, rowspan=2, padx=(12, 8), pady=9)
            ctk.CTkLabel(
                tarjeta,
                text=grupo.nombre,
                text_color=COLOR_TEXTO,
                font=ctk.CTkFont(size=12, weight="bold"),
                anchor="w",
            ).grid(row=0, column=1, sticky="sw", pady=(7, 0))
            ctk.CTkLabel(
                tarjeta,
                text=f"{len(grupo.plantillas)} plantillas seleccionadas",
                text_color=COLOR_TEXTO_SEC,
                font=ctk.CTkFont(size=9),
                anchor="w",
            ).grid(row=1, column=1, sticky="nw", pady=(0, 7))

    def _estado(self, mensaje: str, color: str = COLOR_TEXTO_SEC) -> None:
        self.estado_label.configure(text=mensaje, text_color=color)

    def iniciar_camara(self) -> None:
        if self.cerrando or self.actualizando_base or self.camara_activa:
            return
        try:
            self.camara = abrir_camara()
        except RuntimeError as error:
            self._estado("Cámara no disponible", COLOR_ERROR)
            self.video_estado.configure(text=f"● {error}", text_color=COLOR_ERROR)
            self.video_label.configure(text="No fue posible abrir la cámara", image="")
            self.boton_camara.configure(text="Reintentar cámara")
            return
        self.camara_activa = True
        self.video_estado.configure(text="● Cámara activa", text_color=COLOR_EXITO)
        self.boton_camara.configure(text="Pausar cámara")
        self._estado("Reconocimiento activo", COLOR_EXITO)
        self.after(5, self._actualizar_video)

    def pausar_camara(self) -> None:
        self.camara_activa = False
        if self.camara is not None:
            self.camara.release()
            self.camara = None
        self.video_estado.configure(text="● Cámara en pausa", text_color=COLOR_AVISO)
        self.boton_camara.configure(text="Reanudar cámara")
        self.video_label.configure(image="", text="Cámara en pausa")
        self.imagen_tk = None

    def alternar_camara(self) -> None:
        if self.camara_activa:
            self.pausar_camara()
        else:
            self.iniciar_camara()

    def _mostrar_frame(self, frame: np.ndarray) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagen = Image.fromarray(rgb)
        ancho = max(500, min(720, self.winfo_width() - 430))
        alto = max(280, min(370, self.winfo_height() - 280))
        imagen.thumbnail((ancho, alto), Image.Resampling.BILINEAR)
        lienzo = Image.new("RGB", (ancho, alto), "#05070C")
        x = (ancho - imagen.width) // 2
        y = (alto - imagen.height) // 2
        lienzo.paste(imagen, (x, y))
        self.imagen_tk = ImageTk.PhotoImage(lienzo)
        self.video_label.configure(image=self.imagen_tk, text="")

    def _actualizar_panel_deteccion(self, pistas, cantidad: int) -> None:
        texto_rostros = "rostro detectado" if cantidad == 1 else "rostros detectados"
        self.rostros_label.configure(text=f"{cantidad} {texto_rostros}")
        confirmadas = [p for p in pistas if p.confirmado]
        if confirmadas:
            principal = max(confirmadas, key=lambda p: p.caja[2] * p.caja[3])
            self.identidad_label.configure(
                text=principal.confirmado, text_color=COLOR_EXITO
            )
            self.similitud_label.configure(
                text=f"Similitud estable: {principal.score_confirmado * 100:.0f} / 100"
            )
        else:
            verificando = any(p.actual and p.actual.aceptado for p in pistas)
            self.identidad_label.configure(
                text="Verificando..." if verificando else "—",
                text_color=COLOR_AVISO if verificando else COLOR_TEXTO,
            )
            self.similitud_label.configure(
                text=(
                    "Esperando 5 coincidencias consistentes"
                    if verificando
                    else "Sin identidad confirmada"
                )
            )

    def _procesar_malla(self, frame: np.ndarray, hay_rostros: bool):
        if not hay_rostros:
            self.ultimo_resultado_malla = None
            return None
        self.contador_frames += 1
        if self.contador_frames % 2 == 0 or self.ultimo_resultado_malla is None:
            alto, ancho = frame.shape[:2]
            escala = min(1.0, 640.0 / ancho)
            if escala < 1.0:
                reducido = cv2.resize(
                    frame,
                    (int(ancho * escala), int(alto * escala)),
                    interpolation=cv2.INTER_AREA,
                )
            else:
                reducido = frame
            rgb = cv2.cvtColor(reducido, cv2.COLOR_BGR2RGB)
            self.ultimo_resultado_malla = self.malla.process(rgb)
        return self.ultimo_resultado_malla

    def _procesar_registro(self, observaciones: list[Observacion]) -> None:
        if self.sesion is None:
            return
        if len(observaciones) != 1:
            self.sesion.mensaje = (
                "Colócate frente a la cámara"
                if not observaciones
                else "Debe aparecer solamente la persona registrada"
            )
        else:
            observacion = observaciones[0]
            calidad, mensaje, giro = self.motor.evaluar_calidad(
                observacion.cara, observacion.alineada
            )
            self.sesion.intentar_guardar(
                observacion.alineada,
                observacion.vector,
                calidad,
                mensaje,
                giro,
            )

        self.registro_label.configure(
            text=f"{self.sesion.nombre} · {self.sesion.cantidad}/30",
            text_color=COLOR_AVISO,
        )
        self.registro_detalle.configure(text=self.sesion.mensaje)
        self.progreso.set(self.sesion.cantidad / 30.0)
        if not self.sesion.terminada:
            return

        nombre = self.sesion.nombre
        persona_id = self.sesion.persona_id
        correcto = self.base.incorporar_perfil_color(persona_id)
        self.sesion = None
        self.estabilizador.limpiar()
        self.boton_registrar.configure(text="＋  Registrar o mejorar persona")
        self.registro_label.configure(
            text="Registro completado" if correcto else "No se pudo actualizar",
            text_color=COLOR_EXITO if correcto else COLOR_ERROR,
        )
        self.registro_detalle.configure(
            text=(
                f"El perfil de {nombre} ya usa muestras SFace a color."
                if correcto
                else "Revisa la cámara e inténtalo nuevamente."
            )
        )
        self.progreso.set(1 if correcto else 0)
        self._refrescar_personas()
        self._estado(
            "Perfil actualizado" if correcto else "Error al actualizar perfil",
            COLOR_EXITO if correcto else COLOR_ERROR,
        )

    def _actualizar_video(self) -> None:
        if self.cerrando or not self.camara_activa or self.actualizando_base:
            return
        if self.camara is None:
            return
        ok, frame = self.camara.read()
        if not ok:
            self.pausar_camara()
            self._estado("Se perdió la señal de la cámara", COLOR_ERROR)
            return

        try:
            frame = cv2.flip(frame, 1)
            observaciones = crear_observaciones(frame, self.motor, self.base)
            self._procesar_registro(observaciones)
            entradas = [(o.caja, o.resultado) for o in observaciones]
            pistas = self.estabilizador.actualizar(entradas)

            resultado_malla = self._procesar_malla(frame, bool(observaciones))
            if resultado_malla is not None:
                dibujar_partes(frame, resultado_malla)
            identidad = dibujar_pistas(
                frame, pistas, self.voz, self.sesion is not None
            )
            self._actualizar_panel_deteccion(pistas, len(observaciones))
            if self.sesion is not None:
                self._estado("Capturando perfil facial", COLOR_AVISO)
            elif identidad:
                self._estado("Identidad confirmada", COLOR_EXITO)
            elif observaciones:
                self._estado("Analizando identidad", COLOR_AVISO)
            else:
                self._estado("Reconocimiento activo", COLOR_EXITO)
            self._mostrar_frame(frame)
            ahora = time.perf_counter()
            fps = 1.0 / max(ahora - self.ultimo_tiempo_frame, 1e-6)
            self.ultimo_tiempo_frame = ahora
            self.fps_suavizados = (
                fps if self.fps_suavizados == 0 else 0.88 * self.fps_suavizados + 0.12 * fps
            )
            self.video_estado.configure(
                text=f"● Cámara activa · {self.fps_suavizados:.0f} FPS",
                text_color=COLOR_EXITO,
            )
        except Exception as error:
            self._estado("Error procesando la imagen", COLOR_ERROR)
            self.video_estado.configure(text=f"● {error}", text_color=COLOR_ERROR)
        if self.camara_activa and not self.cerrando:
            self.after(10, self._actualizar_video)

    def registrar(self) -> None:
        if self.actualizando_base:
            return
        if self.sesion is not None:
            nombre = self.sesion.nombre
            self.sesion = None
            self.estabilizador.limpiar()
            self.boton_registrar.configure(text="＋  Registrar o mejorar persona")
            self.registro_label.configure(text="Registro cancelado", text_color=COLOR_ERROR)
            self.registro_detalle.configure(text=f"No se completó el perfil de {nombre}.")
            self.progreso.set(0)
            return
        if not self.camara_activa:
            self.iniciar_camara()
            if not self.camara_activa:
                return
        dialogo = ctk.CTkInputDialog(
            title="Registrar o mejorar persona",
            text=(
                "Escribe el nombre. Si ya existe, usa exactamente el mismo "
                "para mejorar ese perfil:"
            ),
        )
        nombre = dialogo.get_input()
        if not nombre or not nombre.strip():
            return
        nombre = nombre.strip()
        persona_id, carpeta = self.base.iniciar_registro(nombre)
        self.sesion = SesionRegistro(persona_id, nombre, carpeta)
        self.estabilizador.limpiar()
        self.boton_registrar.configure(text="×  Cancelar registro", fg_color="#7F3042")
        self.registro_label.configure(text=f"{nombre} · 0/30", text_color=COLOR_AVISO)
        self.registro_detalle.configure(text="Mira al frente")
        self.progreso.set(0)
        self._estado("Registro guiado activo", COLOR_AVISO)

    def reconstruir_base(self) -> None:
        if self.actualizando_base:
            return
        if self.sesion is not None:
            self._estado("Termina o cancela el registro primero", COLOR_AVISO)
            return
        self.reanudar_tras_actualizar = self.camara_activa
        if self.camara_activa:
            self.pausar_camara()
        self.actualizando_base = True
        self._estado("Recalculando todos los perfiles...", COLOR_AVISO)
        self.video_label.configure(text="Recalibrando la base facial", image="")
        self.boton_reconstruir.configure(state="disabled", text="Recalculando...")
        self.boton_registrar.configure(state="disabled")
        self.boton_camara.configure(state="disabled")

        def trabajo() -> None:
            try:
                nueva = BaseRostrosSFace(MotorFacial(), reconstruir=True)
                self.eventos.put(("base_lista", nueva))
            except Exception as error:
                self.eventos.put(("error_base", str(error)))

        threading.Thread(target=trabajo, daemon=True).start()

    def _procesar_eventos(self) -> None:
        if self.cerrando:
            return
        while True:
            try:
                tipo, contenido = self.eventos.get_nowait()
            except queue.Empty:
                break
            self.actualizando_base = False
            self.boton_reconstruir.configure(
                state="normal", text="↻  Recalibrar base facial"
            )
            self.boton_registrar.configure(state="normal")
            self.boton_camara.configure(state="normal")
            if tipo == "base_lista":
                self.base = contenido
                self.estabilizador.limpiar()
                self._refrescar_personas()
                self._estado("Base facial recalibrada", COLOR_EXITO)
                self.registro_label.configure(text="Base actualizada", text_color=COLOR_EXITO)
                self.registro_detalle.configure(
                    text=f"{len(self.base.grupos)} identidades consolidadas."
                )
            else:
                self._estado("No se pudo recalibrar la base", COLOR_ERROR)
                self.registro_detalle.configure(text=str(contenido))
            if self.reanudar_tras_actualizar:
                self.reanudar_tras_actualizar = False
                self.iniciar_camara()
        self.after(150, self._procesar_eventos)

    def cerrar(self) -> None:
        if self.cerrando:
            return
        self.cerrando = True
        self.camara_activa = False
        if self.camara is not None:
            self.camara.release()
            self.camara = None
        self.malla.close()
        self.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reconstruir",
        action="store_true",
        help="Recalcula la galería antes de abrir la interfaz.",
    )
    parser.add_argument(
        "--solo-reconstruir",
        action="store_true",
        help="Recalcula la galería y termina sin abrir la cámara.",
    )
    argumentos = parser.parse_args()
    if argumentos.solo_reconstruir:
        base = BaseRostrosSFace(MotorFacial(), reconstruir=True)
        for linea in base.resumen():
            print(linea)
        return
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    AplicacionFacial(reconstruir=argumentos.reconstruir).mainloop()


if __name__ == "__main__":
    main()
