# Calibración automática usando el tamaño conocido de la pelota
import cv2
import numpy as np

def calibrar_escala_pelota(archivo_coords, tamaño_real_pelota_m=0.08):
    """
    Calibrar la escala píxeles-metros usando el tamaño conocido de la pelota
    """
    # Cargar coordenadas
    data = np.loadtxt(archivo_coords, delimiter=',', skiprows=1)
    frames = data[:, 0].astype(int)
    
    # Lista para almacenar tamaños de pelota en píxeles
    tamaños_px = []
    
    # Procesar video para medir la pelota en varios frames
    cap = cv2.VideoCapture('data/tiro_parabolico.mp4')
    
    for i, frame_num in enumerate(frames[:10]):  # Usar solo los primeros 10 frames para calibración
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
            
        # Convertir a HSV y aplicar la misma detección
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Usar los mismos rangos de color que en detecta_pelota.py
        lower_orange1 = np.array([5, 50, 50])
        upper_orange1 = np.array([35, 255, 255])
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
        
        lower_orange2 = np.array([20, 100, 100])
        upper_orange2 = np.array([30, 255, 255])
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
        
        mask = cv2.bitwise_or(mask1, mask2)
        
        # Filtros morfológicos
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
            if valid_contours:
                c = max(valid_contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(c)
                
                # Usar el promedio de ancho y alto como tamaño característico
                tamaño_promedio = (w + h) / 2
                tamaños_px.append(tamaño_promedio)
                
                print(f"Frame {frame_num}: Pelota {w}x{h} px, promedio: {tamaño_promedio:.1f} px")
    
    cap.release()
    
    if not tamaños_px:
        print("No se pudo detectar la pelota para calibración")
        return None
    
    # Calcular tamaño promedio de la pelota en píxeles
    tamaño_promedio_px = np.mean(tamaños_px)
    std_tamaño = np.std(tamaños_px)
    
    print(f"\nTamaño promedio de pelota: {tamaño_promedio_px:.1f} ± {std_tamaño:.1f} píxeles")
    print(f"Tamaño real de pelota: {tamaño_real_pelota_m} m")
    
    # Calcular escala: píxeles por metro
    escala_px_por_m = tamaño_promedio_px / tamaño_real_pelota_m
    
    print(f"Escala calculada: {escala_px_por_m:.1f} píxeles/metro")
    print(f"Resolución: {1/escala_px_por_m*1000:.2f} mm/píxel")
    
    return escala_px_por_m

if __name__ == '__main__':
    escala = calibrar_escala_pelota('results/coords.txt', 0.08)
    print(f"\nUsar esta escala en analisis_trayectoria.py: escala_px_por_m={escala:.1f}")