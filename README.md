# Reconocedor facial SFace

Aplicación de escritorio local en Python que detecta y etiqueta partes de la cara, registra personas con su consentimiento, reconoce varias caras y anuncia por voz los nombres confirmados.

La interfaz incluye cámara en vivo, estado del análisis, identidad confirmada, similitud, progreso del registro guiado, controles de cámara y una lista de perfiles consolidados.

## Ejecutar

La forma más sencilla es hacer doble clic en `iniciar.bat`. La primera vez también crea el entorno e instala las dependencias automáticamente si hace falta.

Desde Visual Studio Code también puedes pulsar `F5` y escoger **Iniciar reconocedor facial SFace**, o ejecutar en la terminal:

```powershell
.\.venv\Scripts\python.exe app.py
```

Controles de la interfaz:

- Botón **Registrar o mejorar persona**: inicia o cancela el registro guiado.
- Botón **Recalibrar base facial**: reconstruye la galería sin congelar la ventana.
- Botón **Pausar/Reanudar cámara**: controla la cámara.
- Atajos: `R` registra, `T` recalibra y `Esc` cierra.

## Mejorar los perfiles actuales

Los registros antiguos fueron migrados automáticamente, pero eran recortes en escala de grises. Para conseguir la máxima precisión con `ADRI`, `yura` y `darlenis`:

1. Pulsa `R`.
2. Escribe exactamente el nombre que ya aparece en pantalla.
3. Debe aparecer una sola persona, con buena iluminación y sin gafas oscuras ni objetos que oculten la cara.
4. Sigue los mensajes: primero de frente, después hacia un lado y finalmente hacia el lado contrario.
5. Al completar 30 muestras, el nuevo perfil a color sustituirá las muestras antiguas de ese registro.

Puedes repetir el proceso en otro momento o con una iluminación diferente. El sistema conserva las muestras a color y selecciona automáticamente las más representativas.

## Cómo mejora la exactitud

Esta versión ya no usa LBPH ni su antiguo umbral de distancia. Utiliza:

- **YuNet** para detectar el rostro y cinco puntos faciales.
- **Alineación geométrica** antes de comparar caras.
- **SFace** para crear un vector de identidad de 128 características.
- Eliminación automática de muestras atípicas y fotogramas casi repetidos.
- Agrupación de registros duplicados de la misma persona.
- Comparación contra varias muestras y contra el centro del perfil.
- Umbral adaptativo: una persona parecida exige una coincidencia más alta.
- Margen mínimo contra el segundo candidato; si dos resultados son cercanos, muestra `Resultado ambiguo` en lugar de adivinar.
- Confirmación temporal de 5 resultados coincidentes dentro de 7 fotogramas antes de mostrar o pronunciar un nombre.

El número mostrado es similitud, no una probabilidad matemática. Ningún sistema biométrico es perfecto; iluminación, ángulo, distancia y oclusiones siguen importando.

## Datos y privacidad

Todo se procesa localmente. Las muestras están en `datos/rostros/` y `datos/rostros_sface/`; la galería calculada está en `datos/galeria_sface.npz`. No compartas esas carpetas sin autorización de las personas registradas.

Los modelos proceden del repositorio oficial [OpenCV Zoo: YuNet](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet) y [OpenCV Zoo: SFace](https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface).

## Diagnóstico

Para recalcular la galería sin abrir la cámara:

```powershell
.\.venv\Scripts\python.exe app.py --solo-reconstruir
```

La versión anterior permanece en `app_lbph_anterior.py` únicamente como respaldo.
