"""
Escenario 1: M/M/1 (una sola caja)
Sirve para validar el simulador contra las formulas teoricas.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import simulador as sim

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(CARPETA, "figuras")
RES = os.path.join(CARPETA, "resultados")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

# parametros: 20 clientes por hora, atencion promedio de 2.5 min
lam = 20 / 60
mu = 1 / 2.5
T = 10000         # minutos simulados por replica
calentamiento = 1000
R = 30


def una_corrida(rng, lam=lam, mu=mu, T=T, calentamiento=calentamiento):
    llegadas = sim.llegadas_poisson(lam, T, rng)
    servicios = sim.exponencial(mu, len(llegadas), rng)
    inicio, fin = sim.simular_cola(llegadas, servicios, 1)
    return sim.medidas(llegadas, inicio, fin, 1, calentamiento, T)


# ---------------------------------------------------------------
# 1. Validacion de los numeros aleatorios generados
# ---------------------------------------------------------------
rng = sim.crear_generador(1)
llegadas = sim.llegadas_poisson(lam, 60000, rng)   # unas 20,000 llegadas
entre_llegadas = np.diff(llegadas)
servicios = sim.exponencial(mu, 20000, rng)

print("=== Pruebas de bondad de ajuste ===")
filas = []
for nombre, datos, tasa in [("Entre llegadas", entre_llegadas, lam),
                            ("Servicio", servicios, mu)]:
    ks = stats.kstest(datos, "expon", args=(0, 1 / tasa))
    # chi cuadrado con 10 clases equiprobables
    k = 10
    bordes = stats.expon.ppf(np.linspace(0, 1, k + 1), scale=1 / tasa)
    obs, _ = np.histogram(datos, bins=bordes)
    esp = np.full(k, len(datos) / k)
    chi = stats.chisquare(obs, esp)
    filas.append([nombre, len(datos), datos.mean(), 1 / tasa,
                  ks.statistic, ks.pvalue, chi.statistic, chi.pvalue])
pruebas = pd.DataFrame(filas, columns=["Variable", "n", "Media muestral", "Media teorica",
                                       "D (KS)", "p-valor KS", "Chi2", "p-valor Chi2"])
print(pruebas.round(4).to_string(index=False))
pruebas.round(4).to_csv(os.path.join(RES, "esc1_pruebas_ajuste.csv"), index=False)

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for eje, datos, tasa, titulo in [(ax[0], entre_llegadas, lam, "Tiempo entre llegadas"),
                                 (ax[1], servicios, mu, "Tiempo de servicio")]:
    eje.hist(datos, bins=50, density=True, alpha=0.6, color="tab:blue", label="Simulado")
    x = np.linspace(0, np.percentile(datos, 99.5), 300)
    eje.plot(x, tasa * np.exp(-tasa * x), "r", lw=2, label="Exponencial teórica")
    eje.set_title(titulo)
    eje.set_xlabel("minutos")
    eje.legend()
ax[0].set_ylabel("densidad")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc1_histogramas.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 2. Simulacion vs teoria para rho = 0.833
# ---------------------------------------------------------------
res = sim.replicar(una_corrida, R)
teo = sim.teoria_mmc(lam, mu, 1)
teo["utilizacion"] = teo["rho"]

filas = []
for medida in ["Wq", "W", "Lq", "L", "utilizacion", "P_espera"]:
    media, h = sim.intervalo_confianza(res[medida])
    error = abs(media - teo[medida]) / teo[medida] * 100
    filas.append([medida, teo[medida], media, h, error])
tabla = pd.DataFrame(filas, columns=["Medida", "Teorico", "Simulado", "IC 95% (+/-)", "Error %"])
print("\n=== M/M/1, lambda = 20/h, mu = 24/h, rho = %.3f ===" % teo["rho"])
print(tabla.round(4).to_string(index=False))
tabla.round(4).to_csv(os.path.join(RES, "esc1_mm1_validacion.csv"), index=False)

# ley de Little: L = lambda * W
L_little = res["lambda_obs"] * res["W"]
print("\nLey de Little: L simulado = %.4f, lambda*W = %.4f" % (res["L"].mean(), L_little.mean()))

# ---------------------------------------------------------------
# 3. Barrido de rho: que tanto se parece la simulacion a la teoria
# con rho alto la cola tarda mas en estabilizarse, por eso corridas mas largas
# ---------------------------------------------------------------
rhos = [0.3, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
filas = []
for rho in rhos:
    lam_r = rho * mu
    # cerca de rho = 1 se necesita mucho mas tiempo para llegar al estado estable
    T_r = 40000 if rho < 0.85 else 150000
    r = sim.replicar(lambda g: una_corrida(g, lam=lam_r, T=T_r, calentamiento=T_r // 10),
                     R, semilla=500)
    media, h = sim.intervalo_confianza(r["Wq"])
    wq_teo = sim.teoria_mmc(lam_r, mu, 1)["Wq"]
    filas.append([rho, lam_r * 60, wq_teo, media, h, abs(media - wq_teo) / wq_teo * 100])
barrido = pd.DataFrame(filas, columns=["rho", "lambda (cl/h)", "Wq teorico", "Wq simulado",
                                       "IC 95% (+/-)", "Error %"])
print("\n=== Barrido de rho ===")
print(barrido.round(3).to_string(index=False))
barrido.round(4).to_csv(os.path.join(RES, "esc1_barrido_rho.csv"), index=False)

x = np.linspace(0.25, 0.96, 200)
plt.figure(figsize=(7, 4.5))
plt.plot(x, x / (mu - x * mu), "k-", label="Teoría: $W_q = \\rho/(\\mu-\\lambda)$")
plt.errorbar(barrido["rho"], barrido["Wq simulado"], yerr=barrido["IC 95% (+/-)"],
             fmt="o", color="tab:red", capsize=4, label="Simulación (IC 95%)")
plt.xlabel("Intensidad de tráfico $\\rho$")
plt.ylabel("Tiempo promedio en cola $W_q$ (min)")
plt.title("M/M/1: simulación vs teoría")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc1_barrido_rho.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Evolucion de la cola en el tiempo (una replica, 8 horas)
# ---------------------------------------------------------------
rng = sim.crear_generador(7)
lleg = sim.llegadas_poisson(lam, 480, rng)
serv = sim.exponencial(mu, len(lleg), rng)
ini, fin = sim.simular_cola(lleg, serv, 1)

# numero en cola en cada instante: llegadas - inicios de servicio
eventos = np.concatenate([lleg, ini])
cambios = np.concatenate([np.ones(len(lleg)), -np.ones(len(ini))])
orden = np.argsort(eventos, kind="stable")
t_ev = eventos[orden]
en_cola = np.cumsum(cambios[orden])

plt.figure(figsize=(10, 3.8))
plt.step(t_ev / 60, en_cola, where="post", lw=1)
plt.axhline(teo["Lq"], color="r", ls="--", label="$L_q$ teórico = %.2f" % teo["Lq"])
plt.xlabel("Horas desde la apertura")
plt.ylabel("Clientes en cola")
plt.title("Evolución de la cola en una jornada de 8 horas (M/M/1, $\\rho$ = 0.83)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc1_cola_tiempo.png"), dpi=150)
plt.close()
