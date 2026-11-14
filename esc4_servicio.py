"""
Escenario 4: tiempos de servicio no exponenciales (M/G/1 y M/G/c)
Se mantiene el mismo tiempo promedio de atencion pero se cambia su variabilidad.
Para M/G/1 existe la formula de Pollaczek-Khinchine.
Para M/G/c no hay formula exacta; se compara con la aproximacion de Allen-Cunneen.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import simulador as sim

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(CARPETA, "figuras")
RES = os.path.join(CARPETA, "resultados")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

T = 40000
calentamiento = 4000
R = 30


# generadores de tiempos de servicio con media m
def determinista(m, n, rng):
    return np.full(n, m)


def uniforme(m, n, rng):
    return rng.uniform(0.2 * m, 1.8 * m, n)


def gamma_k(k):
    # gamma con forma k y media m, coeficiente de variacion 1/sqrt(k)
    def gen(m, n, rng):
        return rng.gamma(k, m / k, n)
    return gen


def exponencial(m, n, rng):
    return sim.exponencial(1 / m, n, rng)


distribuciones = [
    ("Determinista", determinista, 0.0),
    ("Uniforme", uniforme, (1.6 ** 2 / 12)),        # CV^2 = (b-a)^2 / (12 m^2)
    ("Gamma (k=2)", gamma_k(2), 0.5),
    ("Exponencial", exponencial, 1.0),
    ("Gamma (k=0.5)", gamma_k(0.5), 2.0),
    ("Gamma (k=0.25)", gamma_k(0.25), 4.0),
]


def corrida(rng, lam, m, generador, c):
    llegadas = sim.llegadas_poisson(lam, T, rng)
    servicios = generador(m, len(llegadas), rng)
    inicio, fin = sim.simular_cola(llegadas, servicios, c)
    return sim.medidas(llegadas, inicio, fin, c, calentamiento, T)


# ---------------------------------------------------------------
# 1. M/G/1: lambda = 20/h, servicio promedio 2.5 min (rho = 0.833)
# ---------------------------------------------------------------
lam1, m1 = 20 / 60, 2.5
filas = []
for nombre, gen, cv2 in distribuciones:
    res = sim.replicar(lambda g: corrida(g, lam1, m1, gen, 1), R, semilla=700)
    media, h = sim.intervalo_confianza(res["Wq"])
    teo = sim.teoria_mg1(lam1, m1, cv2 * m1 ** 2)["Wq"]
    filas.append([nombre, cv2, teo, media, h, abs(media - teo) / teo * 100])
mg1 = pd.DataFrame(filas, columns=["Distribucion", "CV^2", "Wq P-K (min)", "Wq simulado (min)",
                                   "IC 95% (+/-)", "Error %"])
print("=== M/G/1, rho = 0.833 ===")
print(mg1.round(3).to_string(index=False))
mg1.round(4).to_csv(os.path.join(RES, "esc4_mg1.csv"), index=False)

# ---------------------------------------------------------------
# 2. M/G/5: lambda = 50/h, servicio promedio 5 min (como el escenario 2)
# Allen-Cunneen: Wq(M/G/c) ~ Wq(M/M/c) * (1 + CV^2) / 2
# ---------------------------------------------------------------
lam5, m5, c = 50 / 60, 5.0, 5
wq_mmc = sim.teoria_mmc(lam5, 1 / m5, c)["Wq"]
filas = []
for nombre, gen, cv2 in distribuciones:
    res = sim.replicar(lambda g: corrida(g, lam5, m5, gen, c), R, semilla=800)
    media, h = sim.intervalo_confianza(res["Wq"])
    aprox = wq_mmc * (1 + cv2) / 2
    filas.append([nombre, cv2, aprox, media, h, abs(media - aprox) / media * 100])
mgc = pd.DataFrame(filas, columns=["Distribucion", "CV^2", "Wq Allen-Cunneen (min)",
                                   "Wq simulado (min)", "IC 95% (+/-)", "Diferencia %"])
print("\n=== M/G/5, rho = 0.833 ===")
print(mgc.round(3).to_string(index=False))
mgc.round(4).to_csv(os.path.join(RES, "esc4_mgc.csv"), index=False)

# ---------------------------------------------------------------
# 3. Grafica Wq vs CV^2
# ---------------------------------------------------------------
x = np.linspace(0, 4.2, 100)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

ax[0].plot(x, [sim.teoria_mg1(lam1, m1, v * m1 ** 2)["Wq"] for v in x], "k-",
           label="Pollaczek-Khinchine")
ax[0].errorbar(mg1["CV^2"], mg1["Wq simulado (min)"], yerr=mg1["IC 95% (+/-)"],
               fmt="o", color="tab:red", capsize=4, label="Simulación (IC 95%)")
ax[0].set_title("M/G/1 ($\\rho$ = 0.83)")

ax[1].plot(x, wq_mmc * (1 + x) / 2, "k--", label="Aproximación Allen-Cunneen")
ax[1].errorbar(mgc["CV^2"], mgc["Wq simulado (min)"], yerr=mgc["IC 95% (+/-)"],
               fmt="o", color="tab:blue", capsize=4, label="Simulación (IC 95%)")
ax[1].set_title("M/G/5 ($\\rho$ = 0.83)")

for eje, tabla in [(ax[0], mg1), (ax[1], mgc)]:
    for _, fila in tabla.iterrows():
        eje.annotate(fila["Distribucion"], (fila["CV^2"], fila["Wq simulado (min)"]),
                     textcoords="offset points", xytext=(6, -12), fontsize=8)
    eje.set_xlabel("Variabilidad del servicio $CV^2$")
    eje.set_ylabel("$W_q$ (min)")
    eje.grid(alpha=0.3)
    eje.legend(loc="upper left")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc4_variabilidad.png"), dpi=150)
plt.close()
