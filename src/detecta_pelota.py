# importaciones
import cv2
import numpy as np
import os
import json

# Variables globales
pausar_video = False
mostrar_pixel = False
mostrar_hsv = True
x, y = 0, 0
trayectoria = []

# PARÁMETROS HSV INICIALES (se cargarán desde archivo si existe)
H_MIN = 50
H_MAX = 124
S_MIN = 92
S_MAX = 223
V_MIN = 0
V_MAX = 255
KERNEL_SIZE = 1
AREA_MIN = 1000
# Parámetros fijos de dilatación
DILATION_ITERATIONS = 1
DILATION_KERNEL_SIZE = 3

# Archivo para guardar configuración
CONFIG_FILE = 'parametros_hsv.json'

# Funciones para cargar y guardar configuración
def cargar_parametros():
    global H_MIN, H_MAX, S_MIN, S_MAX, V_MIN, V_MAX, KERNEL_SIZE, AREA_MIN
    try:
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, 'r') as f:
                config = json.load(f)
                H_MIN = config.get('H_MIN', H_MIN)
                H_MAX = config.get('H_MAX', H_MAX)
                S_MIN = config.get('S_MIN', S_MIN)
                S_MAX = config.get('S_MAX', S_MAX)
                V_MIN = config.get('V_MIN', V_MIN)
                V_MAX = config.get('V_MAX', V_MAX)
                KERNEL_SIZE = config.get('KERNEL_SIZE', KERNEL_SIZE)
                AREA_MIN = config.get('AREA_MIN', AREA_MIN)
                print(f"Parámetros cargados desde {CONFIG_FILE}")
    except Exception as e:
        print(f"Error cargando parámetros: {e}")

def guardar_parametros():
    try:
        config = {
            'H_MIN': cv2.getTrackbarPos('H Min', 'Controles'),
            'H_MAX': cv2.getTrackbarPos('H Max', 'Controles'),
            'S_MIN': cv2.getTrackbarPos('S Min', 'Controles'),
            'S_MAX': cv2.getTrackbarPos('S Max', 'Controles'),
            'V_MIN': cv2.getTrackbarPos('V Min', 'Controles'),
            'V_MAX': cv2.getTrackbarPos('V Max', 'Controles'),
            'KERNEL_SIZE': cv2.getTrackbarPos('Kernel', 'Controles'),
            'AREA_MIN': cv2.getTrackbarPos('Area Min', 'Controles')
        }
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
        print(f"Parámetros guardados en {CONFIG_FILE}")
    except Exception as e:
        print(f"Error guardando parámetros: {e}")

# Función dummy para trackbars
def nothing(val):
    pass

# Función de callback de mouse
def mouse_callback(event, _x, _y, flags, param):
    global pausar_video, mostrar_pixel, x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        x, y = _x, _y
        mostrar_pixel = True
        print(f"Pixel seleccionado: ({x}, {y})")
    if event == cv2.EVENT_RBUTTONDOWN:
        pausar_video = not pausar_video
        print("Pausado" if pausar_video else "Reanudado")

# Cargar parámetros guardados
cargar_parametros()

# Cargar el video
video_path = 'data/tiro_parabolico.mp4'
cap = cv2.VideoCapture(video_path)

# Validar si el video se abrió
if not cap.isOpened():
    print("Error: no se pudo abrir video")
else:
    print("video abierto correctamente")

# Crear ventanas
cv2.namedWindow('Video')
cv2.namedWindow('Controles')
cv2.setMouseCallback('Video', mouse_callback)

# Crear trackbars con valores cargados
cv2.createTrackbar('H Min', 'Controles', H_MIN, 179, nothing)
cv2.createTrackbar('H Max', 'Controles', H_MAX, 179, nothing)
cv2.createTrackbar('S Min', 'Controles', S_MIN, 255, nothing)
cv2.createTrackbar('S Max', 'Controles', S_MAX, 255, nothing)
cv2.createTrackbar('V Min', 'Controles', V_MIN, 255, nothing)
cv2.createTrackbar('V Max', 'Controles', V_MAX, 255, nothing)
cv2.createTrackbar('Kernel', 'Controles', KERNEL_SIZE, 10, nothing)
cv2.createTrackbar('Area Min', 'Controles', AREA_MIN, 5000, nothing)

nuevo_alto = 480
nuevo_ancho = 680
frame_num = 0
coords = []
frame_actual = None

print("=== DETECCIÓN DE PELOTA CON TRACKBARS ===")
print(f"Parámetros iniciales: H[{H_MIN},{H_MAX}] S[{S_MIN},{S_MAX}] V[{V_MIN},{V_MAX}]")
print(f"Área mínima: {AREA_MIN}, Kernel: {KERNEL_SIZE}")
print("")
print("Controles:")
print("- ESPACIO: Pausar/reanudar")
print("- S: Guardar parámetros actuales")
print("- Click en pelota: Ver información")
print("- ESC: Salir")

os.makedirs('results', exist_ok=True)

# Leer primer frame
ret, frame_actual = cap.read()
if not ret:
    print("Error: No se pudo leer el primer frame")
    exit()

while True:
    if not pausar_video:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_num = 0
            trayectoria.clear()
            ret, frame = cap.read()
            if not ret:
                break
        frame_actual = frame.copy()
        frame_num += 1
    else:
        frame = frame_actual.copy()
    
    frame3 = cv2.resize(frame, (nuevo_ancho, nuevo_alto)) 
    hsv = cv2.cvtColor(frame3, cv2.COLOR_BGR2HSV)
    
    # Obtener valores actuales de los trackbars
    h_min = cv2.getTrackbarPos('H Min', 'Controles')
    h_max = cv2.getTrackbarPos('H Max', 'Controles')
    s_min = cv2.getTrackbarPos('S Min', 'Controles')
    s_max = cv2.getTrackbarPos('S Max', 'Controles')
    v_min = cv2.getTrackbarPos('V Min', 'Controles')
    v_max = cv2.getTrackbarPos('V Max', 'Controles')
    kernel_size = cv2.getTrackbarPos('Kernel', 'Controles')
    area_min = cv2.getTrackbarPos('Area Min', 'Controles')
    
    # Crear máscara con parámetros de trackbars
    bajo = np.array([h_min, s_min, v_min])
    alto = np.array([h_max, s_max, v_max])
    mascara_normal = cv2.inRange(hsv, bajo, alto)
    
    # Aplicar filtros morfológicos para mejorar la forma de la pelota
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    if kernel_size < 1:
        kernel_size = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    
    # Aplicar dilatación con parámetros fijos
    dilation_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (DILATION_KERNEL_SIZE, DILATION_KERNEL_SIZE))
    mascara_normal = cv2.dilate(mascara_normal, dilation_kernel, iterations=DILATION_ITERATIONS)
    
    # Secuencia mejorada de operaciones morfológicas
    # 1. CLOSE: Rellena huecos pequeños dentro de la pelota
    mascara_normal = cv2.morphologyEx(mascara_normal, cv2.MORPH_CLOSE, kernel)
    
    # 2. OPEN: Elimina ruido pequeño fuera de la pelota
    mascara_normal = cv2.morphologyEx(mascara_normal, cv2.MORPH_OPEN, kernel)
    
    # 3. Segunda dilatación para mejorar la forma circular
    kernel_circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mascara_normal = cv2.morphologyEx(mascara_normal, cv2.MORPH_CLOSE, kernel_circle)
    
    # 4. Suavizado final con filtro gaussiano
    mascara_normal = cv2.GaussianBlur(mascara_normal, (3, 3), 0)
    
    # 5. Re-binarizar después del suavizado
    _, mascara_normal = cv2.threshold(mascara_normal, 127, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mascara_normal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_deteccion = frame3.copy()
    
    # Mostrar información del pixel seleccionado
    if mostrar_pixel and 0 <= x < frame3.shape[1] and 0 <= y < frame3.shape[0]:
        hsv_pixel = hsv[y, x]
        bgr_pixel = frame3[y, x]
        
        cv2.putText(frame_deteccion, f'BGR: {bgr_pixel}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_deteccion, f'HSV: {hsv_pixel}', (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        cv2.circle(frame_deteccion, (x, y), 8, (255, 0, 0), 2)
    
    # Mostrar parámetros actuales
    cv2.putText(frame_deteccion, f'HSV: H[{h_min},{h_max}] S[{s_min},{s_max}] V[{v_min},{v_max}]', 
               (10, nuevo_alto - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Detectar pelota
    contornos_validos = [c for c in contours if cv2.contourArea(c) >= area_min]
    
    cv2.putText(frame_deteccion, f'Contornos: {len(contours)} | Válidos: {len(contornos_validos)}', 
               (10, nuevo_alto - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Dibujar la pelota detectada
    if contornos_validos:
        # Tomar el contorno más grande
        pelota_contorno = max(contornos_validos, key=cv2.contourArea)
        x_c, y_c, w, h = cv2.boundingRect(pelota_contorno)
        area = cv2.contourArea(pelota_contorno)
        
        # Dibujar detección
        cv2.rectangle(frame_deteccion, (x_c, y_c), (x_c+w, y_c+h), (0, 255, 0), 2)
        cv2.putText(frame_deteccion, f'Pelota - Area: {int(area)}', (x_c, y_c-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Calcular centro para trayectoria
        centro_x = x_c + w // 2
        centro_y = y_c + h // 2
        trayectoria.append((centro_x, centro_y))
        
        # Limitar trayectoria a últimos 30 puntos
        if len(trayectoria) > 30:
            trayectoria.pop(0)
        
        # Dibujar trayectoria
        for i in range(1, len(trayectoria)):
            cv2.line(frame_deteccion, trayectoria[i-1], trayectoria[i], (0, 0, 255), 2)
    
    # Estado de pausa
    if pausar_video:
        cv2.putText(frame_deteccion, 'PAUSADO', 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # Mostrar ventanas
    cv2.imshow('Mascara', mascara_normal)
    cv2.imshow('Video', frame_deteccion)
    
    key = cv2.waitKey(33) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # ESPACIO
        pausar_video = not pausar_video
        print("Pausado" if pausar_video else "Reanudado")
    elif key == ord('s'):  # S - Guardar parámetros
        guardar_parametros()
    elif key == ord('t'):  # Limpiar trayectoria
        trayectoria.clear()
        print("Trayectoria limpiada")

cap.release()
cv2.destroyAllWindows()

print("Detección completada!")
print(f"Puntos de trayectoria capturados: {len(trayectoria)}")
