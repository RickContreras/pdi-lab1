# Análisis de trayectoria parabólica
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd

def cargar_coordenadas(archivo):
    """Cargar coordenadas desde el archivo de texto"""
    data = np.loadtxt(archivo, delimiter=',', skiprows=1)
    frames = data[:, 0]
    x = data[:, 1]
    y = data[:, 2]
    return frames, x, y

def convertir_pixeles_a_metros(x_px, y_px, escala_px_por_m=100, origen_y=1080):
    """
    Convertir coordenadas de píxeles a metros
    escala_px_por_m: píxeles por metro (ajustar según calibración)
    origen_y: coordenada Y del suelo en píxeles
    """
    x_m = x_px / escala_px_por_m
    y_m = (origen_y - y_px) / escala_px_por_m  # Invertir Y para que arriba sea positivo
    return x_m, y_m

def ajuste_parabolico(x, y):
    """Ajustar una parábola a los datos y=ax²+bx+c"""
    def parabola(x, a, b, c):
        return a * x**2 + b * x + c
    
    # Ajuste por mínimos cuadrados
    popt, pcov = optimize.curve_fit(parabola, x, y)
    return popt, pcov

def calcular_parametros_fisicos(x, y, fps=30):
    """Calcular parámetros físicos del movimiento"""
    # Convertir frames a tiempo
    t = np.arange(len(x)) / fps
    
    # Calcular velocidades usando diferencias finitas
    dt = 1/fps
    
    # Usar diferencias finitas centradas para mayor precisión
    vx = np.zeros_like(x)
    vy = np.zeros_like(y)
    
    # Diferencias hacia adelante para el primer punto
    vx[0] = (x[1] - x[0]) / dt
    vy[0] = (y[1] - y[0]) / dt
    
    # Diferencias centradas para puntos intermedios
    for i in range(1, len(x)-1):
        vx[i] = (x[i+1] - x[i-1]) / (2*dt)
        vy[i] = (y[i+1] - y[i-1]) / (2*dt)
    
    # Diferencias hacia atrás para el último punto
    vx[-1] = (x[-1] - x[-2]) / dt
    vy[-1] = (y[-1] - y[-2]) / dt
    
    # Velocidad inicial (promedio de los primeros valores para estabilidad)
    v0x = np.mean(vx[:3])
    v0y = np.mean(vy[:3])
    v0 = np.sqrt(v0x**2 + v0y**2)
    
    # Ángulo de lanzamiento
    angulo = np.arctan2(v0y, v0x) * 180 / np.pi
    
    # Calcular aceleración usando ajuste lineal de velocidad vs tiempo
    # Esto es más robusto que usar gradientes
    from scipy import stats
    
    # Ajuste lineal para velocidad en Y (la gravedad afecta solo a vy)
    slope_y, intercept_y, r_value_y, p_value_y, std_err_y = stats.linregress(t, vy)
    g_estimada = -slope_y  # La pendiente de vy vs t es -g
    
    # También calcular usando diferencias finitas como método alternativo
    ay = np.gradient(vy, dt)
    g_gradiente = -np.mean(ay[2:-2]) if len(ay) > 4 else -np.mean(ay)
    
    print(f"Gravedad por regresión lineal: {g_estimada:.2f} m/s²")
    print(f"Gravedad por gradiente: {g_gradiente:.2f} m/s²")
    print(f"R² del ajuste lineal: {r_value_y**2:.3f}")
    
    return {
        'v0x': v0x,
        'v0y': v0y,
        'v0': v0,
        'angulo': angulo,
        'gravedad': g_estimada,
        'gravedad_alt': g_gradiente,
        'r_squared': r_value_y**2,
        'tiempo': t,
        'velocidad_x': vx,
        'velocidad_y': vy
    }

def generar_graficos(x, y, parametros, ajuste_coef):
    """Generar gráficos de análisis"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Gráfico 1: Trayectoria
    ax1.plot(x, y, 'bo-', label='Datos detectados', markersize=4)
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = ajuste_coef[0] * x_fit**2 + ajuste_coef[1] * x_fit + ajuste_coef[2]
    ax1.plot(x_fit, y_fit, 'r-', label='Ajuste parabólico')
    ax1.set_xlabel('Posición X (m)')
    ax1.set_ylabel('Posición Y (m)')
    ax1.set_title('Trayectoria de la Pelota')
    ax1.legend()
    ax1.grid(True)
    
    # Gráfico 2: Posición vs Tiempo
    t = parametros['tiempo']
    ax2.plot(t, x, 'b-', label='X(t)')
    ax2.plot(t, y, 'r-', label='Y(t)')
    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Posición (m)')
    ax2.set_title('Posición vs Tiempo')
    ax2.legend()
    ax2.grid(True)
    
    # Gráfico 3: Velocidad vs Tiempo
    ax3.plot(t, parametros['velocidad_x'], 'b-', label='Vx(t)')
    ax3.plot(t, parametros['velocidad_y'], 'r-', label='Vy(t)')
    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Velocidad (m/s)')
    ax3.set_title('Velocidad vs Tiempo')
    ax3.legend()
    ax3.grid(True)
    
    # Gráfico 4: Información de parámetros
    ax4.text(0.1, 0.8, f'Velocidad inicial: {parametros["v0"]:.2f} m/s', transform=ax4.transAxes)
    ax4.text(0.1, 0.7, f'Ángulo de lanzamiento: {parametros["angulo"]:.1f}°', transform=ax4.transAxes)
    ax4.text(0.1, 0.6, f'Velocidad inicial X: {parametros["v0x"]:.2f} m/s', transform=ax4.transAxes)
    ax4.text(0.1, 0.5, f'Velocidad inicial Y: {parametros["v0y"]:.2f} m/s', transform=ax4.transAxes)
    ax4.text(0.1, 0.4, f'Gravedad estimada: {parametros["gravedad"]:.2f} m/s²', transform=ax4.transAxes)
    ax4.text(0.1, 0.3, f'Altura máxima: {max(y):.2f} m', transform=ax4.transAxes)
    ax4.text(0.1, 0.2, f'Alcance: {max(x) - min(x):.2f} m', transform=ax4.transAxes)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Parámetros del Movimiento')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/analisis_trayectoria.png', dpi=300, bbox_inches='tight')
    print("Gráficos guardados en results/analisis_trayectoria.png")

def main():
    # Cargar datos
    frames, x_px, y_px = cargar_coordenadas('results/coords.txt')
    
    # Convertir a metros usando la escala calibrada con el tamaño real de la pelota (8cm)
    escala_calibrada = 678.8  # píxeles/metro (calculada con calibrar_escala.py)
    x_m, y_m = convertir_pixeles_a_metros(x_px, y_px, escala_px_por_m=escala_calibrada, origen_y=1080)
    
    print(f"Datos cargados: {len(frames)} puntos")
    print(f"Escala calibrada: {escala_calibrada:.1f} píxeles/metro (basada en pelota de 8cm)")
    print(f"Rango X: {min(x_m):.3f} - {max(x_m):.3f} m")
    print(f"Rango Y: {min(y_m):.3f} - {max(y_m):.3f} m")
    
    # Ajuste parabólico
    coef, cov = ajuste_parabolico(x_m, y_m)
    print(f"Coeficientes parabólicos: a={coef[0]:.4f}, b={coef[1]:.4f}, c={coef[2]:.4f}")
    
    # Calcular parámetros físicos
    parametros = calcular_parametros_fisicos(x_m, y_m, fps=30)
    
    # Mostrar resultados
    print("\n=== ANÁLISIS DE TRAYECTORIA PARABÓLICA (CALIBRADO) ===")
    print(f"Velocidad inicial: {parametros['v0']:.2f} m/s")
    print(f"Ángulo de lanzamiento: {parametros['angulo']:.1f}°")
    print(f"Componente horizontal de velocidad: {parametros['v0x']:.2f} m/s")
    print(f"Componente vertical de velocidad: {parametros['v0y']:.2f} m/s")
    print(f"Gravedad estimada (regresión): {parametros['gravedad']:.2f} m/s²")
    print(f"Gravedad estimada (gradiente): {parametros['gravedad_alt']:.2f} m/s²")
    print(f"Confiabilidad del ajuste (R²): {parametros['r_squared']:.3f}")
    print(f"Altura máxima: {max(y_m):.3f} m")
    print(f"Alcance horizontal: {max(x_m) - min(x_m):.3f} m")
    print(f"Tiempo de vuelo: {len(frames)/30:.2f} s")
    
    # Análisis de validez
    print(f"\n=== ANÁLISIS DE VALIDEZ ===")
    if parametros['r_squared'] > 0.8:
        print("✓ Ajuste lineal de velocidad es bueno (R² > 0.8)")
    else:
        print("⚠ Ajuste lineal de velocidad es pobre (R² < 0.8)")
    
    if 5 < parametros['gravedad'] < 15:
        print("✓ Gravedad estimada está en rango razonable (5-15 m/s²)")
    else:
        print("⚠ Gravedad estimada fuera del rango esperado")
        print("  Posibles causas:")
        print("  - Solo se capturó parte de la trayectoria")
        print("  - Efectos de perspectiva de cámara")
        print("  - Resolución temporal insuficiente (30 fps)")
        print("  - Calibración de escala incorrecta")
    
    # Generar gráficos
    generar_graficos(x_m, y_m, parametros, coef)
    
    # Guardar resultados en archivo
    resultados = pd.DataFrame({
        'frame': frames,
        'x_px': x_px,
        'y_px': y_px,
        'x_m': x_m,
        'y_m': y_m,
        'tiempo': parametros['tiempo'],
        'vx': parametros['velocidad_x'],
        'vy': parametros['velocidad_y']
    })
    resultados.to_csv('results/analisis_completo_calibrado.csv', index=False)
    print("Datos completos calibrados guardados en results/analisis_completo_calibrado.csv")

if __name__ == '__main__':
    main()