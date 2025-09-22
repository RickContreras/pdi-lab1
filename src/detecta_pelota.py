# -*- coding: utf-8 -*-
# Deteccion de pelota en video de tiro parabolico
import cv2
import numpy as np
import os

def detectar_pelota_video(video_path, output_path='results/coords.txt', mostrar_video=False, guardar_frames=False):
    """
    Detecta la pelota en cada frame del video y guarda las coordenadas
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

                    # Dibujar en el frame
                    cv2.circle(frame, (cx, cy), 10, (0, 255, 0), 2)
                    cv2.circle(frame, (cx, cy), 2, (0, 0, 255), -1)

                    # Dibujar el contorno
                    cv2.drawContours(frame, [c], -1, (255, 0, 0), 2)

                    # Mostrar coordenadas
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

        # Mostrar progreso cada 30 frames
        if frame_count % 30 == 0:
            print("Procesando frame " + str(frame_count) + "/" + str(total_frames) + " - Detecciones: " + str(len(coordenadas)))

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("\nProcesamiento completado:")
    print("Frames procesados: " + str(frame_count))
    print("Detecciones exitosas: " + str(len(coordenadas)))
    if frame_count > 0:
        print("Tasa de deteccion: " + str(round(len(coordenadas)/frame_count*100, 1)) + "%")

    # Guardar coordenadas en archivo
    if coordenadas:
        with open(output_path, 'w') as f:
            f.write('frame,x,y\n')
            for coord in coordenadas:
                f.write(str(coord[0]) + ',' + str(coord[1]) + ',' + str(coord[2]) + '\n')

        print("Coordenadas guardadas en: " + output_path)
        print("Video procesado guardado en: results/video_procesado.mp4")

        # Mostrar estadisticas basicas
        coords_array = np.array(coordenadas)
        print("\nEstadisticas de deteccion:")
        print("Rango X: " + str(coords_array[:, 1].min()) + " - " + str(coords_array[:, 1].max()) + " pixeles")
        print("Rango Y: " + str(coords_array[:, 2].min()) + " - " + str(coords_array[:, 2].max()) + " pixeles")
        print("Duracion del movimiento: " + str(round(len(coordenadas)/fps, 2)) + " segundos")
    else:
        print("No se detecto ninguna pelota en el video")

def main():
    video_path = 'data/tiro_parabolico.mp4'

    if not os.path.exists(video_path):
        print("Error: No se encontro el video " + video_path)
        return

    print("Iniciando deteccion de pelota...")
    print("Presiona 'q' para salir del video (si se muestra)")

    # Ejecutar deteccion (cambiar mostrar_video=True para ver el proceso)
    detectar_pelota_video(video_path, mostrar_video=False, guardar_frames=True)

if __name__ == '__main__':
    main()