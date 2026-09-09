"""Galería facial y algoritmo de decisión multi-muestra."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from motor_sface import MotorFacial, nombres_parecidos, normalizar_nombre, normalizar_vector


RAIZ = Path(__file__).resolve().parent
DATOS = RAIZ / "datos"
ROSTROS_ANTIGUOS = DATOS / "rostros"
ROSTROS_COLOR = DATOS / "rostros_sface"
ARCHIVO_PERSONAS = DATOS / "personas.json"
ARCHIVO_GALERIA = DATOS / "galeria_sface.npz"

MIN_MUESTRAS_COLOR = 8
UMBRAL_BASE = 0.48
MARGEN_MINIMO = 0.07


@dataclass
class GrupoIdentidad:
    nombre: str
    ids: tuple[str, ...]
    plantillas: np.ndarray
    centro: np.ndarray
    umbral: float = UMBRAL_BASE


@dataclass
class Resultado:
    candidato: Optional[str]
    aceptado: bool
    similitud: float
    segunda_similitud: float
    margen: float
    motivo: str

    @property
    def nombre(self) -> Optional[str]:
        return self.candidato if self.aceptado else None


class BaseRostrosSFace:
    """Agrupa sesiones, quita atípicos y exige separación entre candidatos."""

    def __init__(self, motor: MotorFacial, reconstruir: bool = False) -> None:
        DATOS.mkdir(exist_ok=True)
        ROSTROS_ANTIGUOS.mkdir(exist_ok=True)
        ROSTROS_COLOR.mkdir(exist_ok=True)
        self.motor = motor
        self.personas = self._cargar_personas()
        self.caracteristicas = np.empty((0, 128), dtype=np.float32)
        self.ids = np.empty((0,), dtype="U20")
        self.perfiles_color: set[str] = set()
        self.grupos: list[GrupoIdentidad] = []

        if reconstruir or not self._cargar_galeria():
            print("Construyendo la galería SFace a partir de las muestras...")
            self.entrenar()

    @staticmethod
    def _cargar_personas() -> dict[str, str]:
        if not ARCHIVO_PERSONAS.exists():
            return {}
        try:
            datos = json.loads(ARCHIVO_PERSONAS.read_text(encoding="utf-8"))
            return {str(k): str(v) for k, v in datos.items()}
        except (OSError, json.JSONDecodeError):
            print("No se pudo leer personas.json; se usará una base vacía.")
            return {}

    def _guardar_personas(self) -> None:
        ARCHIVO_PERSONAS.write_text(
            json.dumps(self.personas, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def _huella_datos(self) -> str:
        resumen = hashlib.sha256()
        resumen.update(json.dumps(self.personas, sort_keys=True).encode("utf-8"))
        for carpeta in (ROSTROS_ANTIGUOS, ROSTROS_COLOR):
            for ruta in sorted(carpeta.rglob("*.png")):
                estado = ruta.stat()
                relativo = ruta.relative_to(RAIZ).as_posix()
                resumen.update(
                    f"{relativo}|{estado.st_size}|{estado.st_mtime_ns}".encode("utf-8")
                )
        return resumen.hexdigest()

    def _cargar_galeria(self) -> bool:
        if not ARCHIVO_GALERIA.exists():
            return False
        try:
            with np.load(ARCHIVO_GALERIA, allow_pickle=False) as datos:
                if str(datos["huella"].item()) != self._huella_datos():
                    return False
                self.caracteristicas = datos["caracteristicas"].astype(np.float32)
                self.ids = datos["ids"].astype("U20")
                self.perfiles_color = set(datos["perfiles_color"].astype(str).tolist())
            self._construir_grupos()
            return bool(self.grupos)
        except (OSError, ValueError, KeyError):
            return False

    def _guardar_galeria(self) -> None:
        np.savez_compressed(
            ARCHIVO_GALERIA,
            caracteristicas=self.caracteristicas,
            ids=self.ids,
            perfiles_color=np.array(sorted(self.perfiles_color), dtype="U20"),
            huella=np.array(self._huella_datos()),
        )

    def _rutas_perfil(self, persona_id: str) -> list[tuple[Path, bool]]:
        modernas = sorted((ROSTROS_COLOR / persona_id).glob("*.png"))
        antiguas = sorted((ROSTROS_ANTIGUOS / persona_id).glob("*.png"))
        if len(modernas) >= MIN_MUESTRAS_COLOR:
            self.perfiles_color.add(persona_id)
            return [(ruta, True) for ruta in modernas[-90:]]
        return [(ruta, True) for ruta in modernas] + [(ruta, False) for ruta in antiguas]

    @staticmethod
    def _filtrar_atipicos(vectores: list[np.ndarray]) -> np.ndarray:
        if not vectores:
            return np.empty((0, 128), dtype=np.float32)
        matriz = np.stack([normalizar_vector(v) for v in vectores])
        if len(matriz) < 6:
            return matriz
        centro = normalizar_vector(np.mean(matriz, axis=0))
        similitudes = matriz @ centro
        mediana = float(np.median(similitudes))
        mad = float(np.median(np.abs(similitudes - mediana)))
        limite = max(0.35, mediana - max(0.10, 3.0 * 1.4826 * mad))
        filtrada = matriz[similitudes >= limite]
        return filtrada if len(filtrada) >= 3 else matriz

    def entrenar(self) -> bool:
        todas: list[np.ndarray] = []
        etiquetas: list[str] = []
        self.perfiles_color.clear()

        for persona_id in sorted(self.personas, key=lambda x: int(x)):
            vectores: list[np.ndarray] = []
            for ruta, ya_alineada in self._rutas_perfil(persona_id):
                try:
                    vector = self.motor.extraer_archivo(ruta, ya_alineada)
                except cv2.error:
                    vector = None
                if vector is not None:
                    vectores.append(vector)
            filtrados = self._filtrar_atipicos(vectores)
            if len(filtrados):
                todas.extend(filtrados)
                etiquetas.extend([persona_id] * len(filtrados))
                tipo = "color" if persona_id in self.perfiles_color else "migradas"
                print(
                    f"  {self.personas[persona_id]}: "
                    f"{len(filtrados)} muestras {tipo}"
                )

        if not todas:
            self.caracteristicas = np.empty((0, 128), dtype=np.float32)
            self.ids = np.empty((0,), dtype="U20")
            self.grupos = []
            return False

        self.caracteristicas = np.stack(todas).astype(np.float32)
        self.ids = np.array(etiquetas, dtype="U20")
        self._construir_grupos()
        self._guardar_galeria()
        print(f"Galería lista: {len(self.grupos)} identidades consolidadas.")
        return True

    @staticmethod
    def _seleccionar_representativas(
        matriz: np.ndarray, limite: int = 30
    ) -> np.ndarray:
        """Elige centro y variaciones distintas, evitando fotogramas repetidos."""
        if len(matriz) <= limite:
            return matriz
        centro = normalizar_vector(np.mean(matriz, axis=0))
        elegidos = [int(np.argmax(matriz @ centro))]
        disponibles = set(range(len(matriz))) - set(elegidos)
        while disponibles and len(elegidos) < limite:
            candidatos = np.array(sorted(disponibles), dtype=int)
            parecido = matriz[candidatos] @ matriz[elegidos].T
            novedad = 1.0 - np.max(parecido, axis=1)
            nuevo = int(candidatos[int(np.argmax(novedad))])
            elegidos.append(nuevo)
            disponibles.remove(nuevo)
        return matriz[elegidos]

    def _nombre_del_grupo(self, ids_grupo: list[str]) -> str:
        normalizados = [normalizar_nombre(self.personas[i]) for i in ids_grupo]
        conteos = Counter(normalizados)
        mejor = max(
            conteos,
            key=lambda n: (conteos[n], max(int(i) for i in ids_grupo if normalizar_nombre(self.personas[i]) == n)),
        )
        candidatos = [i for i in ids_grupo if normalizar_nombre(self.personas[i]) == mejor]
        return self.personas[max(candidatos, key=int)]

    def _construir_grupos(self) -> None:
        perfiles: dict[str, np.ndarray] = {}
        centros: dict[str, np.ndarray] = {}
        for persona_id in self.personas:
            matriz = self.caracteristicas[self.ids == persona_id]
            if len(matriz):
                perfiles[persona_id] = matriz
                centros[persona_id] = normalizar_vector(np.mean(matriz, axis=0))

        ids_validos = list(perfiles)
        padre = {persona_id: persona_id for persona_id in ids_validos}

        def buscar(x: str) -> str:
            while padre[x] != x:
                padre[x] = padre[padre[x]]
                x = padre[x]
            return x

        def unir(a: str, b: str) -> None:
            raiz_a, raiz_b = buscar(a), buscar(b)
            if raiz_a != raiz_b:
                padre[raiz_b] = raiz_a

        for indice, a in enumerate(ids_validos):
            for b in ids_validos[indice + 1:]:
                similitud = float(centros[a] @ centros[b])
                mismo_nombre = normalizar_nombre(self.personas[a]) == normalizar_nombre(self.personas[b])
                if (
                    mismo_nombre
                    or similitud >= 0.86
                    or (similitud >= 0.72 and nombres_parecidos(self.personas[a], self.personas[b]))
                ):
                    unir(a, b)

        componentes: dict[str, list[str]] = {}
        for persona_id in ids_validos:
            componentes.setdefault(buscar(persona_id), []).append(persona_id)

        grupos: list[GrupoIdentidad] = []
        for ids_grupo in componentes.values():
            ids_color = [i for i in ids_grupo if i in self.perfiles_color]
            ids_usados = ids_color or ids_grupo
            partes = [
                self._seleccionar_representativas(perfiles[i]) for i in ids_usados
            ]
            plantillas = np.concatenate(partes, axis=0)
            plantillas = self._seleccionar_representativas(plantillas, limite=60)
            centro = normalizar_vector(np.mean(plantillas, axis=0))
            grupos.append(
                GrupoIdentidad(
                    nombre=self._nombre_del_grupo(ids_grupo),
                    ids=tuple(sorted(ids_grupo, key=int)),
                    plantillas=plantillas,
                    centro=centro,
                )
            )

        # Un impostor parecido eleva automáticamente el requisito de ese perfil.
        for grupo in grupos:
            impostores = [
                float(np.max(otro.plantillas @ grupo.centro))
                for otro in grupos
                if otro is not grupo
            ]
            max_impostor = max(impostores, default=0.0)
            grupo.umbral = min(0.66, max(UMBRAL_BASE, max_impostor + 0.10))
        self.grupos = grupos

    def identificar(self, vector: np.ndarray) -> Resultado:
        if not self.grupos:
            return Resultado(None, False, 0.0, 0.0, 0.0, "sin_base")
        vector = normalizar_vector(vector)
        puntuaciones: list[tuple[float, GrupoIdentidad]] = []
        for grupo in self.grupos:
            similitudes = grupo.plantillas @ vector
            k = min(5, len(similitudes))
            mejores = np.partition(similitudes, len(similitudes) - k)[-k:]
            score_local = float(np.mean(mejores))
            score_centro = float(grupo.centro @ vector)
            score = 0.85 * score_local + 0.15 * score_centro
            puntuaciones.append((score, grupo))

        puntuaciones.sort(key=lambda item: item[0], reverse=True)
        mejor_score, mejor_grupo = puntuaciones[0]
        segundo = puntuaciones[1][0] if len(puntuaciones) > 1 else -1.0
        margen = mejor_score - segundo
        if mejor_score < mejor_grupo.umbral:
            motivo = "similitud_baja"
            aceptado = False
        elif margen < MARGEN_MINIMO:
            motivo = "ambiguo"
            aceptado = False
        else:
            motivo = "aceptado"
            aceptado = True
        return Resultado(
            candidato=mejor_grupo.nombre,
            aceptado=aceptado,
            similitud=mejor_score,
            segunda_similitud=segundo,
            margen=margen,
            motivo=motivo,
        )

    def siguiente_id(self) -> int:
        return max((int(i) for i in self.personas), default=0) + 1

    def iniciar_registro(self, nombre: str) -> tuple[str, Path]:
        buscado = normalizar_nombre(nombre)
        existentes = [
            i for i, valor in self.personas.items()
            if normalizar_nombre(valor) == buscado
        ]
        if existentes:
            persona_id = max(existentes, key=int)
        else:
            persona_id = str(self.siguiente_id())
            self.personas[persona_id] = nombre.strip()
            self._guardar_personas()
        carpeta = ROSTROS_COLOR / persona_id
        carpeta.mkdir(parents=True, exist_ok=True)
        return persona_id, carpeta

    def incorporar_perfil_color(self, persona_id: str) -> bool:
        rutas = sorted((ROSTROS_COLOR / persona_id).glob("*.png"))[-90:]
        vectores: list[np.ndarray] = []
        for ruta in rutas:
            vector = self.motor.extraer_archivo(ruta, ya_alineada=True)
            if vector is not None:
                vectores.append(vector)
        nuevos = self._filtrar_atipicos(vectores)
        if not len(nuevos):
            return False

        conservar = self.ids != persona_id
        anteriores = self.caracteristicas[conservar]
        ids_anteriores = self.ids[conservar]
        self.caracteristicas = (
            np.concatenate((anteriores, nuevos), axis=0)
            if len(anteriores) else nuevos
        ).astype(np.float32)
        self.ids = np.concatenate(
            (ids_anteriores, np.array([persona_id] * len(nuevos), dtype="U20"))
        )
        self.perfiles_color.add(persona_id)
        self._construir_grupos()
        self._guardar_galeria()
        return True

    def resumen(self) -> list[str]:
        return [
            f"{g.nombre}: ids {','.join(g.ids)}, {len(g.plantillas)} plantillas, umbral {g.umbral:.3f}"
            for g in self.grupos
        ]
