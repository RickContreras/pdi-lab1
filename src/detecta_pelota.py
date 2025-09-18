# importaciones
import cv2
import numpy as np

# Ruta del video
VIDEO_PATH = 'data/tiro_parabolico.mp4'

def main():
	cap = cv2.VideoCapture(VIDEO_PATH)
	frame_num = 0
	coords = []
	while cap.isOpened():
		ret, frame = cap.read()
		if not ret:
			break

		# Convertir a HSV
		hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #¿Buscar frame corruptos?

        # De mascara el fondo azul

		# Rango de color para la pelota (expandido para mejor detección) 
		# Rango 1: naranja/amarillo
		lower_orange1 = np.array([5, 50, 50])
		upper_orange1 = np.array([35, 255, 255])
		mask1 = cv2.inRange(hsv, lower_orange1, upper_orange1)
		
		# Rango 2: amarillo más brillante
		lower_orange2 = np.array([20, 100, 100])
		upper_orange2 = np.array([30, 255, 255])
		mask2 = cv2.inRange(hsv, lower_orange2, upper_orange2)
		
		# Combinar máscaras
		mask = cv2.bitwise_or(mask1, mask2)
		
		# Aplicar filtros morfológicos para limpiar ruido
		kernel = np.ones((3,3), np.uint8)
		mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
		mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

		# Encontrar contornos
		contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		if contours:
			# Filtrar contornos por área mínima
			valid_contours = [c for c in contours if cv2.contourArea(c) > 20]
			if valid_contours:
				# Tomar el contorno más grande
				c = max(valid_contours, key=cv2.contourArea)
				x, y, w, h = cv2.boundingRect(c)
				
				# Filtrar por relación de aspecto (pelota debería ser circular)
				aspect_ratio = w / h
				if 0.5 <= aspect_ratio <= 2.0:
					cx, cy = x + w//2, y + h//2
					coords.append((frame_num, cx, cy))
					# Dibujar rectángulo y centro
					cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
					cv2.circle(frame, (cx, cy), 5, (0,0,255), -1)
					# Guardar frame con pelota detectada
					cv2.imwrite(f'results/pelota_frame_{frame_num:04d}.jpg', frame)
					print(f"Frame {frame_num}: Pelota detectada en ({cx}, {cy}) - Imagen guardada")

		frame_num += 1
		# Mostrar frame (no funciona en WSL sin servidor X)
		cv2.imshow('Pelota', frame)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

	# Guardar coordenadas
	np.savetxt('results/coords.txt', np.array(coords), fmt='%d', delimiter=',', header='frame,x,y')

if __name__ == '__main__':
	main()
