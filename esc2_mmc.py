"""
Escenario 2: M/M/c (varias cajas con una sola fila)
Llegan 50 clientes por hora y cada atencion dura en promedio 5 minutos.
Se compara el desempeno para distintos numeros de cajas c.
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

lam = 50 / 60
mu = 1 / 5
T = 20000
calentamiento = 2000
R = 30
cajas = [5, 6, 7, 8]


def una_corrida(rng, c):
    llegadas = sim.llegadas_poisson(lam, T, rng)
    servicios = sim.exponencial(mu, len(llegadas), rng)
    inicio, fin = sim.simular_cola(llegadas, servicios, c)
    m = sim.medidas(llegadas, inicio, fin, c, calentamiento, T)
    sel = llegadas >= calentamiento
    m["esperas"] = inicio[sel] - llegadas[sel]  # se usa solo para la grafica de distribucion
    return m


print("Carga ofrecida a = lambda/mu = %.3f, se necesitan al menos %d cajas"
      % (lam / mu, int(np.floor(lam / mu)) + 1))

# ---------------------------------------------------------------
# 1. Tabla simulacion vs teoria para cada c
# ---------------------------------------------------------------
filas = []
corridas = {}
for c in cajas:
    res = sim.replicar(lambda g: una_corrida(g, c), R, semilla=100)
    corridas[c] = res
    teo = sim.teoria_mmc(lam, mu, c)
    teo["utilizacion"] = teo["rho"]
    for medida in ["Wq", "Lq", "W", "utilizacion", "P_espera"]:
        media, h = sim.intervalo_confianza(res[medida])
        filas.append([c, medida, teo[medida], media, h,
                      abs(media - teo[medida]) / teo[medida] * 100])

tabla = pd.DataFrame(filas, columns=["c", "Medida", "Teorico", "Simulado", "IC 95% (+/-)", "Error %"])
print("\n=== M/M/c, lambda = 50/h, servicio promedio 5 min ===")
print(tabla.round(4).to_string(index=False))
tabla.round(4).to_csv(os.path.join(RES, "esc2_mmc_validacion.csv"), index=False)

# tabla resumen en formato ancho (para el informe)
resumen = []
for c in cajas:
    teo = sim.teoria_mmc(lam, mu, c)
    r = corridas[c]
    resumen.append([c, teo["rho"], teo["P_espera"], r["P_espera"].mean(),
                    teo["Wq"], r["Wq"].mean(), teo["Lq"], r["Lq"].mean()])
resumen = pd.DataFrame(resumen, columns=["c", "rho", "P(esperar) teo", "P(esperar) sim",
                                         "Wq teo (min)", "Wq sim (min)", "Lq teo", "Lq sim"])
print("\n=== Resumen ===")
print(resumen.round(3).to_string(index=False))
resumen.round(4).to_csv(os.path.join(RES, "esc2_resumen.csv"), index=False)

# ---------------------------------------------------------------
# 2. Graficas de Wq y utilizacion contra c
# ---------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
c_teo = np.arange(5, 11)
ax[0].plot(c_teo, [sim.teoria_mmc(lam, mu, c)["Wq"] for c in c_teo], "k-o", ms=4,
           label="Teoría (Erlang C)")
medias = [sim.intervalo_confianza(corridas[c]["Wq"]) for c in cajas]
ax[0].errorbar(cajas, [m for m, _ in medias], yerr=[h for _, h in medias],
               fmt="s", color="tab:red", capsize=4, label="Simulación (IC 95%)")
ax[0].set_xlabel("Número de cajas c")
ax[0].set_ylabel("$W_q$ (min)")
ax[0].set_title("Tiempo promedio en cola")
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].bar(cajas, [corridas[c]["utilizacion"].mean() * 100 for c in cajas], color="tab:blue")
for c in cajas:
    u = corridas[c]["utilizacion"].mean() * 100
    ax[1].text(c, u + 1, "%.1f%%" % u, ha="center")
ax[1].set_ylim(0, 100)
ax[1].set_xlabel("Número de cajas c")
ax[1].set_ylabel("Utilización (%)")
ax[1].set_title("Utilización de los cajeros")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc2_wq_utilizacion.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 3. Distribucion del tiempo de espera
# En M/M/c: P(Wq > t) = P_espera * exp(-(c*mu - lambda) t)
# ---------------------------------------------------------------
plt.figure(figsize=(7, 4.5))
t = np.linspace(0, 40, 300)
colores = ["tab:red", "tab:orange", "tab:green", "tab:blue"]
for c, col in zip(cajas, colores):
    esperas = np.concatenate([e for e in corridas[c]["esperas"]])
    esperas_ord = np.sort(esperas)
    prob_sim = 1 - np.searchsorted(esperas_ord, t, side="right") / len(esperas_ord)
    teo = sim.teoria_mmc(lam, mu, c)
    plt.plot(t, prob_sim, color=col, lw=2, alpha=0.6, label="c = %d simulado" % c)
    plt.plot(t, teo["P_espera"] * np.exp(-(c * mu - lam) * t), "--", color=col, lw=1)
plt.plot([], [], "k--", lw=1, label="teoría")
plt.yscale("log")
plt.ylim(1e-3, 1)
plt.xlabel("t (min)")
plt.ylabel("P($W_q$ > t)")
plt.title("Probabilidad de esperar más de t minutos")
plt.legend()
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc2_distribucion_espera.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Una fila compartida vs una fila por caja (c = 6)
# Con filas separadas cada caja es un M/M/1 con lambda/c
# ---------------------------------------------------------------
c = 6


def filas_separadas(rng):
    # cada cliente escoge una caja al azar
    llegadas = sim.llegadas_poisson(lam, T, rng)
    servicios = sim.exponencial(mu, len(llegadas), rng)
    caja = rng.integers(0, c, len(llegadas))
    esperas = []
    for k in range(c):
        idx = caja == k
        ini, _ = sim.simular_cola(llegadas[idx], servicios[idx], 1)
        sel = llegadas[idx] >= calentamiento
        esperas.append(ini[sel] - llegadas[idx][sel])
    esperas = np.concatenate(esperas)
    return {"Wq": esperas.mean(), "P_mas_10": (esperas > 10).mean()}


def fila_unica(rng):
    llegadas = sim.llegadas_poisson(lam, T, rng)
    servicios = sim.exponencial(mu, len(llegadas), rng)
    ini, _ = sim.simular_cola(llegadas, servicios, c)
    sel = llegadas >= calentamiento
    esperas = ini[sel] - llegadas[sel]
    return {"Wq": esperas.mean(), "P_mas_10": (esperas > 10).mean()}


sep = sim.replicar(filas_separadas, R, semilla=300)
uni = sim.replicar(fila_unica, R, semilla=300)
teo_sep = sim.teoria_mmc(lam / c, mu, 1)
teo_uni = sim.teoria_mmc(lam, mu, c)

comp = pd.DataFrame([
    ["Una fila por caja (6 x M/M/1)", teo_sep["Wq"], sep["Wq"].mean(), sep["P_mas_10"].mean() * 100],
    ["Fila unica (M/M/6)", teo_uni["Wq"], uni["Wq"].mean(), uni["P_mas_10"].mean() * 100],
], columns=["Sistema", "Wq teorico (min)", "Wq simulado (min)", "% que espera > 10 min"])
print("\n=== Fila unica vs filas separadas (c = 6) ===")
print(comp.round(3).to_string(index=False))
comp.round(4).to_csv(os.path.join(RES, "esc2_fila_unica_vs_separadas.csv"), index=False)

plt.figure(figsize=(6, 4))
barras = plt.bar(["Una fila\npor caja", "Fila única"], comp["Wq simulado (min)"],
                 color=["tab:gray", "tab:green"])
for b, v in zip(barras, comp["Wq simulado (min)"]):
    plt.text(b.get_x() + b.get_width() / 2, v + 0.2, "%.2f min" % v, ha="center")
plt.ylabel("$W_q$ promedio (min)")
plt.title("Mismo número de cajas (6), distinta organización")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc2_fila_unica_vs_separadas.png"), dpi=150)
plt.close()
