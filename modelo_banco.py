"""
Modelo de un dia de atencion en una agencia bancaria (8:00 a 17:00).
La tasa de llegadas cambia durante el dia: hay picos a media manana,
a la hora de almuerzo y antes del cierre.
"""
import numpy as np
import simulador as sim

APERTURA = 8          # hora de apertura
DURACION = 540        # minutos (8:00 - 17:00)
MU = 1 / 5            # servicio promedio de 5 minutos
TASA_MAX = 1.2        # cota superior de lambda(t) para el thinning
INF = np.inf

# picos: (minuto del dia, altura en clientes/min, ancho en min)
PICOS = [(90, 0.45, 35),    # 9:30
         (285, 0.65, 40),   # 12:45
         (495, 0.45, 35)]   # 16:15
BASE = 0.45                 # 27 clientes por hora fuera de los picos


def tasa_llegadas(t):
    """lambda(t) en clientes por minuto, t = minutos desde la apertura."""
    t = np.asarray(t, dtype=float)
    valor = np.full_like(t, BASE)
    for centro, altura, ancho in PICOS:
        valor += altura * np.exp(-(t - centro) ** 2 / (2 * ancho ** 2))
    return valor


def tasa_promedio():
    t = np.linspace(0, DURACION, 5401)
    return np.trapezoid(tasa_llegadas(t), t) / DURACION


# ventanas de refuerzo (minutos desde la apertura)
VENTANAS_PICO = [(45, 150),    # 8:45 - 10:30
                 (225, 345),   # 11:45 - 13:45
                 (435, INF)]   # 15:15 - cierre


def plan_turnos(fijos, refuerzo):
    """
    Horario de cada cajero. Los fijos trabajan todo el dia y los de refuerzo
    solo entran en las ventanas de hora pico.
    """
    return [[(0, INF)]] * fijos + [VENTANAS_PICO] * refuerzo


def horas_cajero(turnos):
    return sum(min(salida, DURACION) - entrada
               for horario in turnos for entrada, salida in horario) / 60


def simular_dia(rng, turnos):
    llegadas, _ = sim.llegadas_no_homogeneas(tasa_llegadas, TASA_MAX, DURACION, rng)
    servicios = sim.exponencial(MU, len(llegadas), rng)
    inicio, fin = sim.simular_cola_turnos(llegadas, servicios, turnos)
    return llegadas, inicio, fin


def resumen_dia(llegadas, inicio, fin):
    espera = inicio - llegadas
    return {
        "clientes": len(llegadas),
        "Wq": espera.mean(),
        "Wq_p95": np.percentile(espera, 95),
        "Wq_max": espera.max(),
        "P_mas_10": (espera > 10).mean(),
        "horas_espera": espera.sum() / 60,
        "cierre_real": fin.max(),
    }
