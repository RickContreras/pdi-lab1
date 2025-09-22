# -*- coding: utf-8 -*-
# Deteccion de pelota en video de tiro parabolico
import cv2
import numpy as np
import os

def convertir_pixeles_a_metros(x_px, y_px, escala_px_por_m, origen_y):
    """Convertir coordenadas de píxeles a metros"""
    x_m = x_px / escala_px_por_m
    y_m = (origen_y - y_px) / escala_px_por_m
    return x_m, y_m

def calcular_fisica_tiempo_real(posiciones_x, posiciones_y, tiempos, ventana_min=5):
    """Calcular parámetros físicos en tiempo real usando ventana deslizante suavizada"""
    if len(posiciones_x) < ventana_min:
        return None, None, None, None

    # Usar ventana más grande para suavizar más la transición entre frames
    ventana = min(8, len(posiciones_x))
    x_recent = np.array(posiciones_x[-ventana:])
    y_recent = np.array(posiciones_y[-ventana:])
    t_recent = np.array(tiempos[-ventana:])

    # Calcular velocidades usando diferencias finitas suavizadas
    if len(x_recent) >= 3:
        # Usar promedio ponderado de las últimas velocidades para suavizar
        velocidades_x = []
        velocidades_y = []

        for i in range(1, len(x_recent)):
            dt = t_recent[i] - t_recent[i-1]
            dt = max(dt, 1/30)  # Evitar dt muy pequeño
            vx_i = (x_recent[i] - x_recent[i-1]) / dt
            vy_i = (y_recent[i] - y_recent[i-1]) / dt
            velocidades_x.append(vx_i)
            velocidades_y.append(vy_i)

        # Promedio ponderado (más peso a velocidades recientes)
        if velocidades_x:
            pesos = np.linspace(0.5, 1.0, len(velocidades_x))
            vx = np.average(velocidades_x, weights=pesos)
            vy = np.average(velocidades_y, weights=pesos)
        else:
            vx = vy = 0
    else:
        vx = vy = 0

    # Calcular aceleraciones si hay suficientes puntos
    if len(posiciones_x) >= 3:
        # Usar ventana más pequeña para aceleración
        ventana_acel = min(3, len(posiciones_x))
        x_acel = np.array(posiciones_x[-ventana_acel:])
        y_acel = np.array(posiciones_y[-ventana_acel:])
        t_acel = np.array(tiempos[-ventana_acel:])

        # Aproximación de segunda derivada
        if len(x_acel) >= 3:
            dt1 = t_acel[-2] - t_acel[-3]
            dt2 = t_acel[-1] - t_acel[-2]
            dt_avg = (dt1 + dt2) / 2
            dt_avg = max(dt_avg, 1/30)

            ax = ((x_acel[-1] - x_acel[-2])/dt2 - (x_acel[-2] - x_acel[-3])/dt1) / dt_avg
            ay = ((y_acel[-1] - y_acel[-2])/dt2 - (y_acel[-2] - y_acel[-3])/dt1) / dt_avg
        else:
            ax = ay = 0
    else:
        ax = ay = 0

    return vx, vy, ax, ay

def dibujar_vectores_y_parametros(frame, cx, cy, vx, vy, ax, ay, pos_x_m, pos_y_m,
                                 frame_count, tiempo, trayectoria_puntos, escala_vector=50):
    """Dibujar vectores de velocidad y aceleración, trayectoria y mostrar parámetros"""

    # Dibujar trayectoria punto a punto
    if len(trayectoria_puntos) > 1:
        for i in range(1, len(trayectoria_puntos)):
            punto_anterior = trayectoria_puntos[i-1]
            punto_actual = trayectoria_puntos[i]

            # Gradiente de color para la trayectoria (más reciente = más brillante)
            alpha = min(1.0, i / len(trayectoria_puntos))
            color_intensidad = int(255 * alpha)
            color_trayectoria = (color_intensidad//2, color_intensidad, color_intensidad//2)

            cv2.line(frame, punto_anterior, punto_actual, color_trayectoria, 2)

            # Dibujar puntos pequeños en la trayectoria
            cv2.circle(frame, punto_actual, 2, color_trayectoria, -1)

    # Dibujar punto central de la pelota (más destacado)
    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 3)
    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)

    # Vectores unitarios para mejor visualización
    vector_length = 80  # Longitud fija para vectores unitarios (más grandes)

    # Vectores de velocidad descompuestos en X e Y
    if abs(vx) > 0.01 or abs(vy) > 0.01:
        vel_mag = np.sqrt(vx**2 + vy**2)

        # Vector velocidad total (azul claro)
        if vel_mag > 0:
            vx_unit = vx / vel_mag
            vy_unit = vy / vel_mag

            vel_end_x = int(cx + vx_unit * vector_length)
            vel_end_y = int(cy - vy_unit * vector_length)
            cv2.arrowedLine(frame, (cx, cy), (vel_end_x, vel_end_y), (255, 150, 0), 4, tipLength=0.15)

            # Etiqueta de velocidad total
            cv2.putText(frame, 'V: ' + str(round(vel_mag, 1)) + ' m/s', (vel_end_x + 5, vel_end_y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)

        # Componente X de velocidad (azul)
        if abs(vx) > 0.01:
            vx_magnitude = abs(vx)
            vx_unit = 1 if vx > 0 else -1
            vx_end_x = int(cx + vx_unit * vector_length * min(1.0, vx_magnitude / vel_mag))
            cv2.arrowedLine(frame, (cx, cy), (vx_end_x, cy), (255, 0, 0), 3, tipLength=0.15)

            # Etiqueta componente X
            offset_y = -30 if vy > 0 else 20
            cv2.putText(frame, 'Vx: ' + str(round(vx, 1)), (vx_end_x + 5, cy + offset_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 2)

        # Componente Y de velocidad (cian)
        if abs(vy) > 0.01:
            vy_magnitude = abs(vy)
            vy_unit = -1 if vy > 0 else 1  # Invertir Y para visualización
            vy_end_y = int(cy + vy_unit * vector_length * min(1.0, vy_magnitude / vel_mag))
            cv2.arrowedLine(frame, (cx, cy), (cx, vy_end_y), (255, 255, 0), 3, tipLength=0.15)

            # Etiqueta componente Y
            offset_x = 30 if vx > 0 else -80
            cv2.putText(frame, 'Vy: ' + str(round(vy, 1)), (cx + offset_x, vy_end_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 2)

    # Vector aceleración unitario (rojo) - más grande
    if abs(ax) > 0.1 or abs(ay) > 0.1:
        acel_mag = np.sqrt(ax**2 + ay**2)
        if acel_mag > 0:
            # Vector unitario de aceleración
            ax_unit = ax / acel_mag
            ay_unit = ay / acel_mag

            acel_end_x = int(cx + ax_unit * vector_length)  # Misma longitud que velocidad
            acel_end_y = int(cy - ay_unit * vector_length)
            cv2.arrowedLine(frame, (cx, cy), (acel_end_x, acel_end_y), (0, 0, 255), 4, tipLength=0.15)

            # Etiqueta de aceleración
            cv2.putText(frame, 'A: ' + str(round(acel_mag, 1)) + ' m/s²', (acel_end_x + 5, acel_end_y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Panel de información en la esquina superior izquierda
    panel_y = 30
    line_height = 25

    # Fondo semitransparente para el panel
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (400, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)

    # Información de frame y tiempo
    cv2.putText(frame, 'Frame: ' + str(frame_count), (10, panel_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    panel_y += line_height

    cv2.putText(frame, 'Tiempo: ' + str(round(tiempo, 2)) + ' s', (10, panel_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    panel_y += line_height

    # Posición
    cv2.putText(frame, 'Posicion: (' + str(round(pos_x_m, 2)) + ', ' + str(round(pos_y_m, 2)) + ') m', (10, panel_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    panel_y += line_height

    # Velocidad con componentes
    vel_mag = np.sqrt(vx**2 + vy**2)
    cv2.putText(frame, 'Velocidad: ' + str(round(vel_mag, 2)) + ' m/s', (10, panel_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    panel_y += line_height

    cv2.putText(frame, 'Vx: ' + str(round(vx, 2)) + ', Vy: ' + str(round(vy, 2)) + ' m/s', (10, panel_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 100, 100), 2)
    panel_y += line_height

    # Aceleración con componentes
    acel_mag = np.sqrt(ax**2 + ay**2)
    cv2.putText(frame, 'Aceleracion: ' + str(round(acel_mag, 2)) + ' m/s²', (10, panel_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    panel_y += line_height

    cv2.putText(frame, 'Ax: ' + str(round(ax, 2)) + ', Ay: ' + str(round(ay, 2)) + ' m/s²', (10, panel_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 255), 2)

def detectar_pelota_video(video_path, output_path='results/coords.txt', mostrar_video=False, guardar_frames=False,
                         mostrar_fisica=True, escala_px_por_m=564.4, origen_y=1034):
    """
    Detecta la pelota en cada frame del video y guarda las coordenadas
    Con opción de mostrar parámetros físicos en tiempo real
    """
    # Crear directorio de resultados si no existe
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Crear directorio para frames si se solicita guardarlos
    if guardar_frames:
        frames_dir = 'results/frames'
        os.makedirs(frames_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: No se pudo abrir el video " + video_path)
        return

    # Obtener propiedades del video
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Video: " + str(total_frames) + " frames a " + str(fps) + " fps")

    # Lista para almacenar coordenadas
    coordenadas = []
    frame_count = 0

    # Variables para cálculos físicos en tiempo real
    posiciones_x = []
    posiciones_y = []
    tiempos = []
    velocidades_x = []
    velocidades_y = []
    aceleraciones_x = []
    aceleraciones_y = []
    trayectoria_puntos = []  # Para almacenar puntos de la trayectoria

    # Configurar el codec y crear el writer para video de salida
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter('results/video_procesado.mp4', fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_original = frame.copy()

        # Convertir a HSV para mejor deteccion de color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Definir rangos de color para pelota naranja/amarilla
        # Rango 1: Naranja mas amplio
        lower_orange1 = np.array([5, 50, 50])
        upper_orange1 = np.array([35, 255, 255])
        mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)

        # Rango 2: Naranja mas especifico
        lower_orange2 = np.array([20, 100, 100])
        upper_orange2 = np.array([30, 255, 255])
        mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)

        # Combinar mascaras
        mask = cv2.bitwise_or(mask1, mask2)

        # Filtros morfologicos para limpiar la mascara
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Encontrar contornos
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pelota_detectada = False
        if contours:
            # Filtrar contornos por area minima
            valid_contours = [c for c in contours if cv2.contourArea(c) > 20]

            if valid_contours:
                # Tomar el contorno mas grande
                c = max(valid_contours, key=cv2.contourArea)

                # Calcular el centro del contorno
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])

                    # Guardar coordenadas
                    coordenadas.append([frame_count, cx, cy])
                    pelota_detectada = True

                    # Agregar punto a la trayectoria
                    trayectoria_puntos.append((cx, cy))
                    # Mantener solo los últimos 40 puntos para la trayectoria (más suave)
                    if len(trayectoria_puntos) > 40:
                        trayectoria_puntos.pop(0)

                    # Convertir a metros para cálculos físicos
                    pos_x_m, pos_y_m = convertir_pixeles_a_metros(cx, cy, escala_px_por_m, origen_y)
                    tiempo_actual = frame_count / fps

                    # Agregar a listas de física
                    posiciones_x.append(pos_x_m)
                    posiciones_y.append(pos_y_m)
                    tiempos.append(tiempo_actual)

                    # Calcular parámetros físicos en tiempo real
                    vx, vy, ax, ay = calcular_fisica_tiempo_real(posiciones_x, posiciones_y, tiempos)

                    # Agregar velocidades y aceleraciones a las listas
                    if vx is not None:
                        velocidades_x.append(vx)
                        velocidades_y.append(vy)
                        aceleraciones_x.append(ax)
                        aceleraciones_y.append(ay)
                    else:
                        velocidades_x.append(0)
                        velocidades_y.append(0)
                        aceleraciones_x.append(0)
                        aceleraciones_y.append(0)
                        vx = vy = ax = ay = 0

                    # Dibujar detección básica y contorno
                    cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)

                    # Mostrar información física si está habilitado
                    if mostrar_fisica and vx is not None:
                        dibujar_vectores_y_parametros(frame, cx, cy, vx, vy, ax, ay,
                                                    pos_x_m, pos_y_m, frame_count, tiempo_actual, trayectoria_puntos)
                    else:
                        # Mostrar información básica si no hay física
                        cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)
                        cv2.putText(frame, 'Frame: ' + str(frame_count), (10, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, 'Pos: (' + str(cx) + ', ' + str(cy) + ')', (10, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if not pelota_detectada:
            cv2.putText(frame, 'Frame: ' + str(frame_count) + ' - NO DETECTADA', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Escribir frame al video de salida
        out.write(frame)

        # Guardar frame como imagen si se solicita
        if guardar_frames:
            # Frame original con detecciones
            frame_original_filename = 'results/frames/frame_' + str(frame_count).zfill(4) + '_original.jpg'
            cv2.imwrite(frame_original_filename, frame)

            # Frame HSV
            frame_hsv_filename = 'results/frames/frame_' + str(frame_count).zfill(4) + '_hsv.jpg'
            cv2.imwrite(frame_hsv_filename, hsv)

            # Mascara (convertir a 3 canales para consistencia)
            mask_3channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            frame_mask_filename = 'results/frames/frame_' + str(frame_count).zfill(4) + '_mask.jpg'
            cv2.imwrite(frame_mask_filename, mask_3channel)

        # Mostrar video si se solicita
        if mostrar_video:
            cv2.imshow('Deteccion de Pelota', frame)
            cv2.imshow('Mascara', mask)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_count += 1

        # Mostrar progreso menos frecuentemente
        if frame_count % 100 == 0:
            print("Procesando frame " + str(frame_count) + "/" + str(total_frames))

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nProcesamiento completado: " + str(len(coordenadas)) + "/" + str(frame_count) + " detecciones (" + str(round(len(coordenadas)/frame_count*100, 1)) + "%)")

    # Guardar coordenadas en archivo
    if coordenadas:
        with open(output_path, 'w') as f:
            f.write('frame,x,y\n')
            for coord in coordenadas:
                f.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')

        # Guardar datos físicos completos si están disponibles
        if mostrar_fisica and len(posiciones_x) > 0:
            output_fisica = output_path.replace('.txt', '_fisica.csv')
            with open(output_fisica, 'w') as f:
                f.write('frame,x_px,y_px,x_m,y_m,tiempo,vx,vy,ax,ay\n')
                for i in range(len(coordenadas)):
                    coord = coordenadas[i]
                    if i < len(posiciones_x):
                        f.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + ',' +
                               str(round(posiciones_x[i], 4)) + ',' + str(round(posiciones_y[i], 4)) + ',' +
                               str(round(tiempos[i], 4)) + ',' + str(round(velocidades_x[i], 4)) + ',' +
                               str(round(velocidades_y[i], 4)) + ',' + str(round(aceleraciones_x[i], 4)) + ',' +
                               str(round(aceleraciones_y[i], 4)) + '\n')

        print("Archivos guardados: coords.txt, fisica.csv, video_procesado.mp4")

        # Estadísticas resumidas
        if mostrar_fisica and len(posiciones_x) > 0:
            if len(velocidades_x) > 0:
                vel_max = max([np.sqrt(vx**2 + vy**2) for vx, vy in zip(velocidades_x, velocidades_y)])
                print("Velocidad maxima: " + str(round(vel_max, 2)) + " m/s")
            if len(aceleraciones_y) > 3:
                ay_promedio = np.mean(aceleraciones_y[3:])
                print("Gravedad estimada: " + str(round(-ay_promedio, 2)) + " m/s²")
    else:
        print("No se detecto ninguna pelota en el video")

def main():
    video_path = 'data/tiro_parabolico.mp4'

    if not os.path.exists(video_path):
        print("Error: No se encontro el video " + video_path)
        return

    print("Iniciando deteccion de pelota con parametros fisicos...")
    print("Presiona 'q' para salir del video (si se muestra)")

    # Ejecutar deteccion con física habilitada (cambiar mostrar_video=True para ver el proceso)
    detectar_pelota_video(video_path, mostrar_video=False, guardar_frames=True,
                         mostrar_fisica=True, escala_px_por_m=564.4, origen_y=1034)

if __name__ == '__main__':
    main()