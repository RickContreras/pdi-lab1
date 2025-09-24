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
    """Calcular parámetros físicos del movimiento con métodos mejorados"""
    from scipy import stats
    from scipy.ndimage import gaussian_filter1d

    # Convertir frames a tiempo
    t = np.arange(len(x)) / fps
    dt = 1/fps

    # Suavizar datos con filtro Gaussiano para reducir ruido
    x_smooth = gaussian_filter1d(x, sigma=1.0)
    y_smooth = gaussian_filter1d(y, sigma=1.0)

    # MÉTODO 1: Diferencias finitas centradas (mejorado)
    vx = np.gradient(x_smooth, dt)
    vy = np.gradient(y_smooth, dt)

    # Suavizar velocidades también
    vx_smooth = gaussian_filter1d(vx, sigma=0.5)
    vy_smooth = gaussian_filter1d(vy, sigma=0.5)

    # MÉTODO 2: Ajuste parabólico directo para gravedad
    # y = a*x² + b*x + c, donde para movimiento projectil: a = -g/(2*v0x²)
    try:
        # Ajustar parábola en coordenadas (x,y)
        coef_parab = np.polyfit(x, y, 2)
        a_parab = coef_parab[0]

        # Estimar v0x promedio para calcular g
        v0x_est = np.mean(vx_smooth[:5]) if len(vx_smooth) >= 5 else np.mean(vx_smooth)

        if abs(v0x_est) > 0.1:  # Evitar división por cero
            g_parabolico = -2 * a_parab * v0x_est**2
        else:
            g_parabolico = None
    except:
        g_parabolico = None

    # MÉTODO 3: Regresión lineal en vy vs t (original mejorado)
    if len(t) >= 3:
        slope_y, intercept_y, r_value_y, p_value_y, std_err_y = stats.linregress(t, vy_smooth)
        g_regresion = -slope_y
        r_squared = r_value_y**2
    else:
        g_regresion = None
        r_squared = 0

    # MÉTODO 4: Usar ecuaciones de cinemática
    # Para movimiento projectil: y = y0 + v0y*t - (1/2)*g*t²
    # Reorganizar: y = c + b*t + a*t² donde a = -g/2
    try:
        t_fit = t - t[0]  # Normalizar tiempo
        coef_tiempo = np.polyfit(t_fit, y, 2)
        g_cinematica = -2 * coef_tiempo[0]
    except:
        g_cinematica = None

    # MÉTODO 5: Análisis de curvatura local
    if len(y) >= 5:
        # Calcular segunda derivada (aceleración) localmente
        ay_local = np.gradient(vy_smooth, dt)
        # Tomar valores del medio para evitar efectos de borde
        mid_start = len(ay_local) // 3
        mid_end = 2 * len(ay_local) // 3
        g_curvatura = -np.mean(ay_local[mid_start:mid_end]) if mid_end > mid_start else -np.mean(ay_local)
    else:
        g_curvatura = None

    # Determinar mejor estimación de gravedad
    gravedad_estimaciones = []
    metodos = []

    if g_regresion is not None and abs(g_regresion) < 50:  # Filtrar valores extremos
        gravedad_estimaciones.append(g_regresion)
        metodos.append("Regresión lineal")

    if g_parabolico is not None and abs(g_parabolico) < 50:
        gravedad_estimaciones.append(g_parabolico)
        metodos.append("Ajuste parabólico")

    if g_cinematica is not None and abs(g_cinematica) < 50:
        gravedad_estimaciones.append(g_cinematica)
        metodos.append("Ecuaciones cinemáticas")

    if g_curvatura is not None and abs(g_curvatura) < 50:
        gravedad_estimaciones.append(g_curvatura)
        metodos.append("Análisis de curvatura")

    # Calcular parámetros de velocidad inicial
    v0x = np.mean(vx_smooth[:3]) if len(vx_smooth) >= 3 else vx_smooth[0]
    v0y = np.mean(vy_smooth[:3]) if len(vy_smooth) >= 3 else vy_smooth[0]
    v0 = np.sqrt(v0x**2 + v0y**2)
    angulo = np.arctan2(v0y, v0x) * 180 / np.pi

    # Seleccionar mejor estimación de gravedad
    if gravedad_estimaciones:
        # Usar la mediana para ser robusto ante outliers
        g_mejor = np.median(gravedad_estimaciones)
        g_std = np.std(gravedad_estimaciones)
    else:
        g_mejor = 9.81  # Valor teórico si no hay estimaciones válidas
        g_std = float('inf')

    # Mostrar resultados de métodos
    print(f"\n=== ANÁLISIS MULTI-MÉTODO DE GRAVEDAD ===")
    for i, (g_val, metodo) in enumerate(zip(gravedad_estimaciones, metodos)):
        print(f"{metodo}: {g_val:.2f} m/s²")

    if gravedad_estimaciones:
        print(f"Promedio: {np.mean(gravedad_estimaciones):.2f} ± {g_std:.2f} m/s²")
        print(f"Mediana (recomendada): {g_mejor:.2f} m/s²")

    print(f"R² del ajuste lineal: {r_squared:.3f}")

    return {
        'v0x': v0x,
        'v0y': v0y,
        'v0': v0,
        'angulo': angulo,
        'gravedad': g_mejor,
        'gravedad_std': g_std,
        'gravedad_metodos': dict(zip(metodos, gravedad_estimaciones)) if gravedad_estimaciones else {},
        'r_squared': r_squared,
        'tiempo': t,
        'velocidad_x': vx_smooth,
        'velocidad_y': vy_smooth,
        'datos_suavizados': {'x': x_smooth, 'y': y_smooth}
    }

def generar_graficos(x, y, parametros, ajuste_coef):
    """Generar gráficos de análisis con datos teóricos"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    t = parametros['tiempo']

    # Parámetros experimentales para comparación
    v0x_exp = parametros['v0x']
    v0y_exp = parametros['v0y']
    g_exp = parametros['gravedad']
    x0 = x[0]  # Posición inicial X
    y0 = y[0]  # Posición inicial Y

    # PARÁMETROS TEÓRICOS usando g = 9.8 m/s² estándar
    g_teorico = 9.8  # Gravedad teórica estándar

    # Usar los mismos valores iniciales experimentales pero con g teórica
    v0x_teorico = v0x_exp  # Velocidad inicial X (se conserva)
    v0y_teorico = v0y_exp  # Velocidad inicial Y (se conserva)
    x0_teorico = x0  # Posición inicial X (se conserva)
    y0_teorico = y0  # Posición inicial Y (se conserva)

    # Crear tiempo extendido para curvas teóricas suaves
    t_teorico = np.linspace(0, max(t)*1.2, 200)

    # ECUACIONES CINEMÁTICAS TEÓRICAS (g = 9.8 m/s²) con condiciones ajustadas
    x_teorico = x0_teorico + v0x_teorico * t_teorico
    y_teorico = y0_teorico + v0y_teorico * t_teorico - 0.5 * g_teorico * t_teorico**2
    vx_teorico = np.full_like(t_teorico, v0x_teorico)  # Velocidad X constante
    vy_teorico = v0y_teorico - g_teorico * t_teorico  # Velocidad Y = v0y - gt

    # Gráfico 1: Trayectoria Y vs X (EXPERIMENTAL vs TEÓRICO)
    ax1.plot(x, y, 'bo-', label='Datos experimentales', markersize=4, alpha=0.7)
    if 'datos_suavizados' in parametros:
        ax1.plot(parametros['datos_suavizados']['x'], parametros['datos_suavizados']['y'],
                'go-', label='Datos suavizados', markersize=3)

    # Ajuste parabólico experimental
    x_fit = np.linspace(min(x), max(x), 100)
    y_fit = ajuste_coef[0] * x_fit**2 + ajuste_coef[1] * x_fit + ajuste_coef[2]
    ax1.plot(x_fit, y_fit, 'r-', linewidth=2,
             label=f'Ajuste exp: y = {ajuste_coef[0]:.3f}x² + {ajuste_coef[1]:.3f}x + {ajuste_coef[2]:.3f}')

    # Trayectoria teórica
    mask_positivo = y_teorico >= 0
    ax1.plot(x_teorico[mask_positivo], y_teorico[mask_positivo], 'g--', linewidth=2,
             label=f'Teoría cinemática (g={g_teorico} m/s²)', alpha=0.8)

    ax1.set_xlabel('Posición X (m)')
    ax1.set_ylabel('Posición Y (m)')
    ax1.set_title('Trayectoria: Experimental vs Teórica')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Gráfico 2: Posición vs Tiempo (EXPERIMENTAL vs TEÓRICO)
    ax2.plot(t, x, 'bo-', linewidth=1.5, markersize=3, label='X experimental')
    ax2.plot(t, y, 'ro-', linewidth=1.5, markersize=3, label='Y experimental')

    # Posiciones teóricas
    mask_tiempo = y_teorico >= 0
    ax2.plot(t_teorico, x_teorico, 'b--', linewidth=2, alpha=0.7, label=f'X teórica = {x0_teorico:.2f} + {v0x_teorico:.2f}t')
    ax2.plot(t_teorico[mask_tiempo], y_teorico[mask_tiempo], 'r--', linewidth=2, alpha=0.7,
             label=f'Y teórica = {y0_teorico:.2f} + {v0y_teorico:.2f}t - ½({g_teorico})t²')

    ax2.set_xlabel('Tiempo (s)')
    ax2.set_ylabel('Posición (m)')
    ax2.set_title('Posición vs Tiempo: Experimental vs Teórica')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Gráfico 3: Velocidad vs Tiempo (EXPERIMENTAL vs TEÓRICO)
    ax3.plot(t, parametros['velocidad_x'], 'bo-', linewidth=1.5, markersize=3, label='Vx experimental')
    ax3.plot(t, parametros['velocidad_y'], 'ro-', linewidth=1.5, markersize=3, label='Vy experimental')

    # Velocidades teóricas
    ax3.plot(t_teorico, vx_teorico, 'b--', linewidth=2, alpha=0.7, label=f'Vx teórica = {v0x_teorico:.2f} m/s')
    ax3.plot(t_teorico, vy_teorico, 'r--', linewidth=2, alpha=0.7, label=f'Vy teórica = {v0y_teorico:.2f} - {g_teorico}t')

    ax3.set_xlabel('Tiempo (s)')
    ax3.set_ylabel('Velocidad (m/s)')
    ax3.set_title('Velocidad vs Tiempo: Experimental vs Teórica')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Gráfico 4: Información de parámetros y ecuaciones teóricas
    info_text = []
    info_text.append('=== PARÁMETROS EXPERIMENTALES ===')
    info_text.append(f'Velocidad inicial: {parametros["v0"]:.2f} m/s')
    info_text.append(f'Ángulo de lanzamiento: {parametros["angulo"]:.1f}°')
    info_text.append(f'Velocidad inicial X: {parametros["v0x"]:.2f} m/s')
    info_text.append(f'Velocidad inicial Y: {parametros["v0y"]:.2f} m/s')
    info_text.append(f'Gravedad estimada: {parametros["gravedad"]:.2f} m/s²')

    if 'gravedad_std' in parametros and parametros['gravedad_std'] != float('inf'):
        info_text.append(f'Incertidumbre: ±{parametros["gravedad_std"]:.2f} m/s²')

    info_text.append(f'Altura máxima: {max(y):.2f} m')
    info_text.append(f'Alcance: {max(x) - min(x):.2f} m')
    info_text.append('')

    info_text.append('=== ECUACIONES TEÓRICAS (g=9.8) ===')
    info_text.append(f'x(t) = {x0_teorico:.2f} + {v0x_teorico:.2f}t')
    info_text.append(f'y(t) = {y0_teorico:.2f} + {v0y_teorico:.2f}t - ½({g_teorico})t²')
    info_text.append(f'vₓ(t) = {v0x_teorico:.2f} m/s (constante)')
    info_text.append(f'vᵧ(t) = {v0y_teorico:.2f} - {g_teorico}t')
    info_text.append('')

    # Calcular predicciones teóricas con g = 9.8
    if v0y_teorico > 0:
        t_vuelo_teorico = 2 * v0y_teorico / g_teorico
        info_text.append('=== PREDICCIONES TEÓRICAS ===')
        info_text.append(f'Tiempo de vuelo teórico: {t_vuelo_teorico:.2f} s')
        x_alcance_teorico = x0_teorico + v0x_teorico * t_vuelo_teorico
        info_text.append(f'Alcance teórico: {x_alcance_teorico:.2f} m')
        y_max_teorica = y0_teorico + (v0y_teorico**2) / (2 * g_teorico)
        info_text.append(f'Altura máx teórica: {y_max_teorica:.2f} m')
        info_text.append('')

    # Comparación experimental vs teórica
    info_text.append('=== COMPARACIÓN EXP vs TEÓRICA ===')
    info_text.append(f'Gravedad exp: {g_exp:.2f} vs teórica: {g_teorico}')
    error_g = abs(g_exp - g_teorico)
    info_text.append(f'Error en gravedad: {error_g:.2f} m/s²')
    if v0y_teorico > 0:
        t_vuelo_exp = 2 * v0y_exp / g_exp if g_exp > 0 else 0
        if t_vuelo_exp > 0:
            info_text.append(f'T vuelo exp: {t_vuelo_exp:.2f}s vs teór: {t_vuelo_teorico:.2f}s')

    # Mostrar métodos usados
    if 'gravedad_metodos' in parametros and parametros['gravedad_metodos']:
        info_text.append('')
        info_text.append('=== MÉTODOS DE ANÁLISIS ===')
        for metodo, valor in parametros['gravedad_metodos'].items():
            info_text.append(f'{metodo}: {valor:.2f} m/s²')

    for i, text in enumerate(info_text):
        y_pos = 0.98 - i * 0.04
        if y_pos > 0:
            fontsize = 8 if text.startswith('===') else 7
            weight = 'bold' if text.startswith('===') else 'normal'
            ax4.text(0.02, y_pos, text, transform=ax4.transAxes,
                    fontsize=fontsize, verticalalignment='top', weight=weight)

    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_title('Parámetros y Ecuaciones Cinemáticas', fontweight='bold')
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('results/analisis_trayectoria.png', dpi=300, bbox_inches='tight')
    print("Gráficos guardados en results/analisis_trayectoria.png")

def main():
    # Cargar datos
    frames, x_px, y_px = cargar_coordenadas('results/coords.txt')

    # FILTRAR DATOS HASTA EL SUELO (antes del rebote)
    # Identificar el punto más bajo (suelo) - frame donde y_px es máximo
    idx_suelo = np.argmax(y_px)
    print(f"Suelo detectado en frame {frames[idx_suelo]:.0f} (y_px = {y_px[idx_suelo]:.0f})")

    # Tomar solo datos hasta el suelo (inclusive)
    frames_pre_rebote = frames[:idx_suelo+1]
    x_px_pre_rebote = x_px[:idx_suelo+1]
    y_px_pre_rebote = y_px[:idx_suelo+1]

    print(f"Datos originales: {len(frames)} puntos")
    print(f"Datos hasta suelo (sin rebote): {len(frames_pre_rebote)} puntos")

    # Convertir a metros usando la escala calibrada con altura total de 180cm
    # Rango Y: 17-1034 píxeles (1017 píxeles total) para 180cm de altura real
    escala_calibrada = 564.4  # píxeles/metro (1017 píxeles / 1.8 metros)
    origen_y_calibrado = 1034  # Punto más bajo detectado (suelo)

    # Convertir datos completos (para comparación)
    x_m_completo, y_m_completo = convertir_pixeles_a_metros(x_px, y_px, escala_px_por_m=escala_calibrada, origen_y=origen_y_calibrado)

    # Convertir datos hasta el suelo
    x_m, y_m = convertir_pixeles_a_metros(x_px_pre_rebote, y_px_pre_rebote, escala_px_por_m=escala_calibrada, origen_y=origen_y_calibrado)
    
    print(f"\n=== DATOS SIN REBOTE ===")
    print(f"Puntos analizados: {len(frames_pre_rebote)} puntos")
    print(f"Escala calibrada: {escala_calibrada:.1f} píxeles/metro (basada en altura total de 180cm)")
    print(f"Rango X: {min(x_m):.3f} - {max(x_m):.3f} m")
    print(f"Rango Y: {min(y_m):.3f} - {max(y_m):.3f} m")

    # Ajuste parabólico (solo datos hasta suelo)
    coef, cov = ajuste_parabolico(x_m, y_m)
    print(f"Coeficientes parabólicos: a={coef[0]:.4f}, b={coef[1]:.4f}, c={coef[2]:.4f}")

    # Calcular parámetros físicos (solo datos hasta suelo)
    parametros = calcular_parametros_fisicos(x_m, y_m, fps=30)

    # COMPARACIÓN: calcular también con datos completos para ver la diferencia
    print(f"\n=== COMPARACIÓN: DATOS COMPLETOS (CON REBOTE) ===")
    parametros_completo = calcular_parametros_fisicos(x_m_completo, y_m_completo, fps=30)

    # Mostrar resultados (sin rebote)
    print("\n=== ANÁLISIS DE TRAYECTORIA (SIN REBOTE) ===")
    print(f"Velocidad inicial: {parametros['v0']:.2f} m/s")
    print(f"Ángulo de lanzamiento: {parametros['angulo']:.1f}°")
    print(f"Componente horizontal de velocidad: {parametros['v0x']:.2f} m/s")
    print(f"Componente vertical de velocidad: {parametros['v0y']:.2f} m/s")
    print(f"Gravedad estimada (mediana): {parametros['gravedad']:.2f} m/s²")
    if 'gravedad_std' in parametros and parametros['gravedad_std'] != float('inf'):
        print(f"Incertidumbre en gravedad: ±{parametros['gravedad_std']:.2f} m/s²")
    print(f"Confiabilidad del ajuste (R²): {parametros['r_squared']:.3f}")
    print(f"Altura máxima: {max(y_m):.3f} m")
    print(f"Alcance horizontal: {max(x_m) - min(x_m):.3f} m")
    print(f"Tiempo hasta suelo: {len(frames_pre_rebote)/30:.2f} s")

    # Mostrar comparación
    print(f"\n=== COMPARACIÓN DE RESULTADOS ===")
    print(f"{'Parámetro':<25} {'Sin rebote':<15} {'Con rebote':<15} {'Mejora'}")
    print("-" * 70)
    print(f"{'Gravedad (m/s²)':<25} {parametros['gravedad']:<15.2f} {parametros_completo['gravedad']:<15.2f} {abs(parametros['gravedad'] - 9.81) < abs(parametros_completo['gravedad'] - 9.81)}")
    print(f"{'R² ajuste lineal':<25} {parametros['r_squared']:<15.3f} {parametros_completo['r_squared']:<15.3f} {parametros['r_squared'] > parametros_completo['r_squared']}")
    print(f"{'Incertidumbre (±m/s²)':<25} {parametros.get('gravedad_std', float('inf')):<15.2f} {parametros_completo.get('gravedad_std', float('inf')):<15.2f}")
    print(f"{'Puntos analizados':<25} {len(frames_pre_rebote):<15} {len(frames):<15}")

    # Análisis de cercanía a g=9.81 m/s²
    error_sin_rebote = abs(parametros['gravedad'] - 9.81)
    error_con_rebote = abs(parametros_completo['gravedad'] - 9.81)
    print(f"\nError respecto a g=9.81 m/s²:")
    print(f"Sin rebote: {error_sin_rebote:.2f} m/s²")
    print(f"Con rebote: {error_con_rebote:.2f} m/s²")
    if error_sin_rebote < error_con_rebote:
        print("✅ El análisis SIN rebote es más preciso")
    else:
        print("⚠️ El análisis CON rebote sigue siendo mejor")
    
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
    
    # Guardar resultados en archivo (solo datos sin rebote)
    resultados = pd.DataFrame({
        'frame': frames_pre_rebote,
        'x_px': x_px_pre_rebote,
        'y_px': y_px_pre_rebote,
        'x_m': x_m,
        'y_m': y_m,
        'tiempo': parametros['tiempo'],
        'vx': parametros['velocidad_x'],
        'vy': parametros['velocidad_y']
    })
    resultados.to_csv('results/analisis_sin_rebote_calibrado.csv', index=False)
    print("Datos sin rebote guardados en results/analisis_sin_rebote_calibrado.csv")

if __name__ == '__main__':
    main()