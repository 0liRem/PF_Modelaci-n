"""
Simulador de sistemas de colas (M/M/1, M/M/c, M/G/c y llegadas no homogeneas).
Todas las tasas estan en clientes por minuto y los tiempos en minutos.
"""
import heapq
import math
import numpy as np
from scipy import stats


def crear_generador(semilla):
    # Mersenne Twister, el mismo que vimos en el lab 2
    return np.random.Generator(np.random.MT19937(semilla))



# Generacion de variables aleatorias


def exponencial(tasa, n, rng):
    # transformada inversa: F^-1(u) = -ln(1-u)/tasa
    u = rng.random(n)
    return -np.log(1 - u) / tasa


def llegadas_poisson(tasa, T, rng):
    """Tiempos de llegada de un proceso de Poisson homogeneo en [0, T]."""
    n = int(tasa * T * 1.2) + 50
    t = np.cumsum(exponencial(tasa, n, rng))
    while t[-1] < T:
        extra = t[-1] + np.cumsum(exponencial(tasa, n, rng))
        t = np.concatenate([t, extra])
    return t[t <= T]


def llegadas_no_homogeneas(funcion_tasa, tasa_max, T, rng):
    """
    Proceso de Poisson no homogeneo por el metodo de thinning (Lewis-Shedler).
    Es aceptacion-rechazo: se generan llegadas con tasa_max y cada una se
    acepta con probabilidad lambda(t)/tasa_max.
    """
    candidatas = llegadas_poisson(tasa_max, T, rng)
    u = rng.random(len(candidatas))
    aceptadas = u <= funcion_tasa(candidatas) / tasa_max
    return candidatas[aceptadas], len(candidatas)


# ---------------------------------------------------------------------
# Simulacion
# ---------------------------------------------------------------------

def simular_cola(llegadas, servicios, c):
    """
    Cola FIFO con c servidores identicos. Cada cliente pasa con el primer
    servidor que se desocupe. Devuelve el inicio y fin de cada servicio.
    """
    n = len(llegadas)
    inicio = np.empty(n)
    fin = np.empty(n)
    libres = [0.0] * c  # momento en que se desocupa cada servidor
    for i in range(n):
        t_libre = heapq.heappop(libres)
        inicio[i] = max(llegadas[i], t_libre)
        fin[i] = inicio[i] + servicios[i]
        heapq.heappush(libres, fin[i])
    return inicio, fin


def primer_momento_disponible(t, horario):
    """Primer instante >= t en que el servidor esta dentro de alguno de sus turnos."""
    for entrada, salida in horario:
        if t < salida:
            return max(t, entrada)
    return math.inf


def simular_cola_turnos(llegadas, servicios, turnos, devolver_servidor=False):
    """
    Igual que simular_cola pero cada servidor tiene su horario, que es una lista
    de turnos [(entrada, salida), ...] ordenados en el tiempo.
    Un servidor no empieza a atender a nadie fuera de su turno,
    pero si termina al cliente que ya estaba atendiendo.
    """
    n = len(llegadas)
    inicio = np.empty(n)
    fin = np.empty(n)
    servidor = np.empty(n, dtype=int)
    libre = [0.0] * len(turnos)
    for i in range(n):
        mejor, k_mejor = math.inf, -1
        for k, horario in enumerate(turnos):
            t = primer_momento_disponible(max(llegadas[i], libre[k]), horario)
            if t < mejor:
                mejor, k_mejor = t, k
        if k_mejor == -1:
            # nadie lo puede atender (no deberia pasar si el ultimo turno cubre el cierre)
            mejor = max(llegadas[i], min(libre))
            k_mejor = int(np.argmin(libre))
        inicio[i] = mejor
        fin[i] = mejor + servicios[i]
        servidor[i] = k_mejor
        libre[k_mejor] = fin[i]
    if devolver_servidor:
        return inicio, fin, servidor
    return inicio, fin


def medidas(llegadas, inicio, fin, c, t0, t1):
    """
    Medidas de desempeno en la ventana [t0, t1].
    Wq, W: promedio de los clientes que llegaron en la ventana.
    Lq, L, utilizacion: promedios en el tiempo (integral / duracion).
    """
    dentro = (llegadas >= t0) & (llegadas < t1)
    espera = inicio[dentro] - llegadas[dentro]
    sistema = fin[dentro] - llegadas[dentro]

    # tiempo que cada cliente paso en cola / en el sistema dentro de la ventana
    a = np.clip(llegadas, t0, t1)
    b = np.clip(inicio, t0, t1)
    d = np.clip(fin, t0, t1)
    duracion = t1 - t0

    return {
        "Wq": espera.mean(),
        "W": sistema.mean(),
        "Lq": (b - a).sum() / duracion,
        "L": (d - a).sum() / duracion,
        "utilizacion": (d - b).sum() / (c * duracion),
        "P_espera": (espera > 1e-9).mean(),
        "lambda_obs": dentro.sum() / duracion,
    }


def replicar(funcion, R, semilla=2026):
    """Corre R replicas independientes y junta los resultados en arreglos."""
    resultados = []
    for r in range(R):
        rng = crear_generador(semilla + r)
        resultados.append(funcion(rng))
    salida = {}
    for k in resultados[0]:
        valores = [res[k] for res in resultados]
        # los escalares se guardan como arreglo, lo demas (listas de esperas, etc.) como lista
        salida[k] = np.array(valores) if np.isscalar(valores[0]) else valores
    return salida


def intervalo_confianza(x, nivel=0.95):
    n = len(x)
    media = x.mean()
    h = stats.t.ppf((1 + nivel) / 2, n - 1) * x.std(ddof=1) / math.sqrt(n)
    return media, h


# ---------------------------------------------------------------------
# Resultados teoricos
# ---------------------------------------------------------------------

def teoria_mmc(lam, mu, c):
    """Formulas de M/M/c (Erlang C). Para c = 1 da las de M/M/1."""
    a = lam / mu
    rho = a / c
    if rho >= 1:
        return {"rho": rho, "P0": 0, "P_espera": 1, "Lq": math.inf,
                "L": math.inf, "Wq": math.inf, "W": math.inf}
    suma = sum(a**k / math.factorial(k) for k in range(c))
    ultimo = a**c / (math.factorial(c) * (1 - rho))
    P0 = 1 / (suma + ultimo)
    P_espera = ultimo * P0          # formula de Erlang C
    Lq = P_espera * rho / (1 - rho)
    Wq = Lq / lam
    W = Wq + 1 / mu
    L = lam * W
    return {"rho": rho, "P0": P0, "P_espera": P_espera,
            "Lq": Lq, "L": L, "Wq": Wq, "W": W}


def teoria_mg1(lam, media_s, var_s):
    """Formula de Pollaczek-Khinchine para M/G/1."""
    rho = lam * media_s
    if rho >= 1:
        return {"rho": rho, "Lq": math.inf, "Wq": math.inf, "W": math.inf, "L": math.inf}
    Wq = lam * (var_s + media_s**2) / (2 * (1 - rho))
    W = Wq + media_s
    return {"rho": rho, "Lq": lam * Wq, "Wq": Wq, "W": W, "L": lam * W}
