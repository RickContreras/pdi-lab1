# importaciones
import cv2
import numpy as np
import os

# Variables globales
pausar_video = False
mostrar_pixel = False
mostrar_hsv = True
x, y = 0, 0
trayectoria = []

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
        
# Cargar el video
video_path = 'data/tiro_parabolico.mp4'
cap = cv2.VideoCapture(video_path)

# Validar si el video se abrió
if not cap.isOpened():
    print("Error: no se pudo abrir video")
else:
    print("video abierto correctamente")

# Crear una ventana y configurar la función de callback de mouse
cv2.namedWindow('Video')
cv2.setMouseCallback('Video', mouse_callback)

nuevo_alto = 480
nuevo_ancho = 680
frame_num = 0
coords = []
frame_actual = None

def nothing(x):
    pass

# Crear ventana de trackbars MÁS GRANDE
cv2.namedWindow("Controles", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Controles", 400, 400)

# TRACKBARS CON RANGOS MÁS AMPLIOS PARA CALIBRACIÓN
cv2.createTrackbar("H Min", "Controles", 0, 179, nothing)      # Empezar con rango completo
cv2.createTrackbar("H Max", "Controles", 179, 179, nothing)
cv2.createTrackbar("S Min", "Controles", 0, 255, nothing)      # Empezar con rango completo
cv2.createTrackbar("S Max", "Controles", 255, 255, nothing)
cv2.createTrackbar("V Min", "Controles", 0, 255, nothing)      # Empezar con rango completo
cv2.createTrackbar("V Max", "Controles", 255, 255, nothing)
cv2.createTrackbar("Kernel", "Controles", 5, 15, nothing)
cv2.createTrackbar("Area Min", "Controles", 50, 1000, nothing)  # Área mínima

cv2.moveWindow("Controles", 50, 50)
os.makedirs('results', exist_ok=True)

print("=== CALIBRACIÓN DE COLORES HSV ===")
print("1. PAUSA el video (ESPACIO)")
print("2. HAZ CLICK en la pelota para ver sus valores HSV")
print("3. AJUSTA los trackbars según los valores mostrados:")
print("   - H Min/Max: Ajusta el tono (color)")
print("   - S Min: Sube para eliminar colores desaturados")
print("   - V Min: Sube para eliminar áreas muy oscuras")
print("4. La máscara debe mostrar SOLO la pelota en NEGRO")
print("")
print("Controles:")
print("- ESPACIO: Pausar/reanudar")
print("- Click en pelota: Ver valores HSV")
print("- 'p': Mostrar valores sugeridos")
print("- ESC: Salir")

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
    
    # Obtener valores de trackbars
    h_min = cv2.getTrackbarPos("H Min", "Controles")
    h_max = cv2.getTrackbarPos("H Max", "Controles")
    s_min = cv2.getTrackbarPos("S Min", "Controles")
    s_max = cv2.getTrackbarPos("S Max", "Controles")
    v_min = cv2.getTrackbarPos("V Min", "Controles")
    v_max = cv2.getTrackbarPos("V Max", "Controles")
    kernel_size = max(1, cv2.getTrackbarPos("Kernel", "Controles"))
    area_min = cv2.getTrackbarPos("Area Min", "Controles")

    # Crear máscara
    bajo = np.array([h_min, s_min, v_min])
    alto = np.array([h_max, s_max, v_max])
    mascara_normal = cv2.inRange(hsv, bajo, alto)
    
    # Aplicar filtros morfológicos
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mascara_normal = cv2.morphologyEx(mascara_normal, cv2.MORPH_CLOSE, kernel)
    mascara_normal = cv2.morphologyEx(mascara_normal, cv2.MORPH_OPEN, kernel)
    
    # INVERTIR LA MÁSCARA
    mascara_invertida = cv2.bitwise_not(mascara_normal)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(mascara_normal, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_deteccion = frame3.copy()
    
    # MOSTRAR INFORMACIÓN DETALLADA DEL PIXEL SELECCIONADO
    if mostrar_pixel and 0 <= x < frame3.shape[1] and 0 <= y < frame3.shape[0]:
        hsv_pixel = hsv[y, x]
        bgr_pixel = frame3[y, x]
        
        # Mostrar información completa
        cv2.putText(frame_deteccion, f'BGR: {bgr_pixel}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(frame_deteccion, f'HSV: {hsv_pixel}', (10, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Sugerir rangos basados en el pixel seleccionado
        h_sugerido_min = max(0, hsv_pixel[0] - 10)
        h_sugerido_max = min(179, hsv_pixel[0] + 10)
        s_sugerido_min = max(50, hsv_pixel[1] - 50)
        v_sugerido_min = max(50, hsv_pixel[2] - 50)
        
        cv2.putText(frame_deteccion, f'Sugerido H: {h_sugerido_min}-{h_sugerido_max}', (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame_deteccion, f'Sugerido S Min: {s_sugerido_min}', (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame_deteccion, f'Sugerido V Min: {v_sugerido_min}', (10, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        cv2.circle(frame_deteccion, (x, y), 8, (255, 0, 0), 2)
        cv2.circle(hsv, (x, y), 8, (0, 255, 255), 2)
    
    # Mostrar valores actuales de trackbars
    cv2.putText(frame_deteccion, f'H:[{h_min},{h_max}] S:[{s_min},{s_max}] V:[{v_min},{v_max}]', 
               (10, nuevo_alto - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Detectar pelota y contar contornos
    contornos_validos = [c for c in contours if cv2.contourArea(c) >= area_min]
    
    cv2.putText(frame_deteccion, f'Contornos totales: {len(contours)} | Válidos: {len(contornos_validos)}', 
               (10, nuevo_alto - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Dibujar todos los contornos válidos
    for i, c in enumerate(contornos_validos):
        x_c, y_c, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        color = (0, 255, 0) if i == 0 else (0, 255, 255)
        cv2.rectangle(frame_deteccion, (x_c, y_c), (x_c+w, y_c+h), color, 2)
        cv2.putText(frame_deteccion, f'#{i} A:{int(area)}', (x_c, y_c-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    # Estado de pausa
    if pausar_video:
        cv2.putText(frame_deteccion, 'PAUSADO - Perfecto para calibrar!', 
                   (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # MOSTRAR VENTANAS
    cv2.imshow('Mascara Original', mascara_normal)      # Pelota blanca
    cv2.imshow('Mascara Invertida', mascara_invertida)  # Pelota negra
    cv2.imshow('HSV', hsv)                              # Para ver colores
    cv2.imshow('Video', frame_deteccion)                # Con detecciones
    
    key = cv2.waitKey(33) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # ESPACIO
        pausar_video = not pausar_video
        print("Pausado" if pausar_video else "Reanudado")
    elif key == ord('p'):  # Mostrar valores actuales
        print(f"Valores actuales HSV: H[{h_min},{h_max}] S[{s_min},{s_max}] V[{v_min},{v_max}]")
        if mostrar_pixel:
            hsv_pixel = hsv[y, x]
            print(f"Pixel seleccionado HSV: {hsv_pixel}")
    elif key == ord('r'):  # Reset valores
        cv2.setTrackbarPos("H Min", "Controles", 0)
        cv2.setTrackbarPos("H Max", "Controles", 179)
        cv2.setTrackbarPos("S Min", "Controles", 0)
        cv2.setTrackbarPos("S Max", "Controles", 255)
        cv2.setTrackbarPos("V Min", "Controles", 0)
        cv2.setTrackbarPos("V Max", "Controles", 255)
        print("Trackbars reiniciados a valores completos")

cap.release()
cv2.destroyAllWindows()

print("Calibración completada!")
