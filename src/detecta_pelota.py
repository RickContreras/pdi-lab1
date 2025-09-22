# importaciones
import cv2
import numpy as np
import os
import json
from datetime import datetime

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

def test_ball_detection(video_path):
    """Probar detección de pelota en tiempo real"""
    
    if not os.path.exists(video_path):
        print(f"Error: No se encuentra el video en {video_path}")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: No se puede abrir el video")
        return
    
    # Parámetros de detección para pelota (ajustar según el color)
    # Para pelota naranja/amarilla
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    
    # Para pelota blanca
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    
    frame_count = 0
    detections = 0
    
    print("Presiona 'q' para salir, 's' para pausar")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Crear máscaras
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
        mask_white = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combinar máscaras
        mask = cv2.bitwise_or(mask_orange, mask_white)
        
        # Operaciones morfológicas
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        ball_detected = False
        
        if contours:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 50:  # Área mínima
                    # Calcular centro y radio
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    center = (int(x), int(y))
                    radius = int(radius)
                    
                    if radius > 5:  # Radio mínimo
                        # Dibujar círculo
                        cv2.circle(frame, center, radius, (0, 255, 0), 2)
                        cv2.circle(frame, center, 2, (0, 0, 255), -1)
                        
                        # Mostrar información
                        cv2.putText(frame, f"Pelota: ({center[0]}, {center[1]})", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        ball_detected = True
                        detections += 1
                        break
        
        # Mostrar estado de detección
        status = "DETECTADA" if ball_detected else "NO DETECTADA"
        color = (0, 255, 0) if ball_detected else (0, 0, 255)
        cv2.putText(frame, f"Frame {frame_count}: {status}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Mostrar estadísticas
        detection_rate = (detections / (frame_count + 1)) * 100
        cv2.putText(frame, f"Deteccion: {detection_rate:.1f}%", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Mostrar frame
        cv2.imshow('Deteccion de Pelota', frame)
        cv2.imshow('Mascara', mask)
        
        frame_count += 1
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.waitKey(0)  # Pausar
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\nResultados:")
    print(f"Total de frames procesados: {frame_count}")
    print(f"Frames con detección: {detections}")
    print(f"Tasa de detección: {(detections/frame_count)*100:.2f}%")

def capture_detection_frames(video_path, output_dir="capturas"):
    """Capturar frames mostrando el proceso de detección"""
    
    # Crear directorio de salida
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Error: No se puede abrir el video")
        return
    
    # Parámetros de detección
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([255, 30, 255])
    
    frame_count = 0
    capture_interval = 30  # Capturar cada 30 frames
    
    print(f"Capturando frames en: {output_dir}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Procesar solo algunos frames para captura
        if frame_count % capture_interval == 0:
            
            # Frame original
            original = frame.copy()
            
            # Convertir a HSV
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Crear máscaras
            mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            mask = cv2.bitwise_or(mask_orange, mask_white)
            
            # Operaciones morfológicas
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            mask_processed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_OPEN, kernel)
            
            # Encontrar contornos
            contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Frame con detección
            detection_frame = frame.copy()
            ball_detected = False
            
            if contours:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 50:
                        (x, y), radius = cv2.minEnclosingCircle(contour)
                        center = (int(x), int(y))
                        radius = int(radius)
                        
                        if radius > 5:
                            cv2.circle(detection_frame, center, radius, (0, 255, 0), 2)
                            cv2.circle(detection_frame, center, 2, (0, 0, 255), -1)
                            cv2.putText(detection_frame, f"Pelota: ({center[0]}, {center[1]})", 
                                       (center[0]-50, center[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            ball_detected = True
            
            # Agregar información al frame
            status = "DETECTADA" if ball_detected else "NO DETECTADA"
            color = (0, 255, 0) if ball_detected else (0, 0, 255)
            cv2.putText(detection_frame, f"Frame {frame_count}: {status}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Crear composición de 4 imágenes
            # Redimensionar para composición
            h, w = original.shape[:2]
            new_h, new_w = h//2, w//2
            
            img1 = cv2.resize(original, (new_w, new_h))
            img2 = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (new_w, new_h))
            img3 = cv2.resize(cv2.cvtColor(mask_processed, cv2.COLOR_GRAY2BGR), (new_w, new_h))
            img4 = cv2.resize(detection_frame, (new_w, new_h))
            
            # Agregar títulos
            cv2.putText(img1, "Original", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img2, "Mascara Color", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img3, "Mascara Procesada", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(img4, "Deteccion", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Combinar imágenes
            top_row = np.hstack([img1, img2])
            bottom_row = np.hstack([img3, img4])
            combined = np.vstack([top_row, bottom_row])
            
            # Guardar captura
            filename = f"{output_dir}/frame_{frame_count:06d}_{'detected' if ball_detected else 'not_detected'}.jpg"
            cv2.imwrite(filename, combined)
            print(f"Capturado: {filename}")
        
        frame_count += 1
    
    cap.release()
    print(f"\nCapturas guardadas en: {output_dir}")
    print(f"Total de frames procesados: {frame_count}")

def show_existing_captures(directory="capturas"):
    """Mostrar capturas existentes"""
    if not os.path.exists(directory):
        print(f"No existe el directorio: {directory}")
        return
    
    files = [f for f in os.listdir(directory) if f.endswith('.jpg')]
    files.sort()
    
    if not files:
        print(f"No hay capturas en: {directory}")
        return
    
    print(f"\nCapturas encontradas en {directory}:")
    for i, file in enumerate(files):
        detected = "✓ DETECTADA" if "detected" in file and "not_detected" not in file else "✗ NO DETECTADA"
        print(f"{i+1:2d}. {file} - {detected}")
    
    # Mostrar las capturas una por una
    print(f"\nPresiona cualquier tecla para ver las capturas, 'q' para salir")
    
    for file in files:
        img_path = os.path.join(directory, file)
        img = cv2.imread(img_path)
        
        if img is not None:
            cv2.imshow(f"Captura - {file}", img)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            if key == ord('q'):
                break

# Ejecutar test
if __name__ == "__main__":
    # Buscar archivos de video en el directorio actual
    video_files = [f for f in os.listdir('.') if f.endswith(('.mp4', '.avi', '.mov'))]
    
    print("=== ANÁLISIS DE DETECCIÓN DE PELOTA ===")
    print("1. Capturar nuevos frames")
    print("2. Mostrar capturas existentes")
    
    choice = input("\nSelecciona una opción (1 o 2): ")
    
    if choice == "1":
        if video_files:
            video_path = video_files[0]
            print(f"Procesando: {video_path}")
            capture_detection_frames(video_path)
        else:
            print("No se encontraron archivos de video")
    
    elif choice == "2":
        show_existing_captures()
    
    else:
        print("Opción no válida")
