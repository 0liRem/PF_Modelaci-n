
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import simulador as sim
import modelo_banco as banco

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(CARPETA, "figuras")
RES = os.path.join(CARPETA, "resultados")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

COSTO_CAJERO = 45     # Q por hora
COSTO_ESPERA = 60     # Q por hora de cliente en cola
R = 30

# 1. Caso estacionario: lambda = 50/h, servicio de 5 min
# costo por hora = Cs * c + Cw * Lq

lam, mu = 50 / 60, 1 / 5
T, calentamiento = 20000, 2000


def corrida(rng, c):
    llegadas = sim.llegadas_poisson(lam, T, rng)
    servicios = sim.exponencial(mu, len(llegadas), rng)
    inicio, fin = sim.simular_cola(llegadas, servicios, c)
    return sim.medidas(llegadas, inicio, fin, c, calentamiento, T)


filas = []
for c in range(5, 11):
    res = sim.replicar(lambda g: corrida(g, c), R, semilla=900)
    costo_sim = COSTO_CAJERO * c + COSTO_ESPERA * res["Lq"]
    media, h = sim.intervalo_confianza(costo_sim)
    lq_teo = sim.teoria_mmc(lam, mu, c)["Lq"]
    filas.append([c, COSTO_CAJERO * c, COSTO_ESPERA * lq_teo, COSTO_CAJERO * c + COSTO_ESPERA * lq_teo,
                  media, h])
tabla = pd.DataFrame(filas, columns=["c", "Costo cajeros (Q/h)", "Costo espera teo (Q/h)",
                                     "Costo total teo (Q/h)", "Costo total sim (Q/h)", "IC 95% (+/-)"])
c_opt = int(tabla.loc[tabla["Costo total sim (Q/h)"].idxmin(), "c"])
print("=== Caso estacionario (lambda = 50/h) ===")
print(tabla.round(2).to_string(index=False))
print("Numero optimo de cajeros: c* = %d" % c_opt)
tabla.round(3).to_csv(os.path.join(RES, "esc5_costo_estacionario.csv"), index=False)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(tabla["c"], tabla["Costo cajeros (Q/h)"], "--", color="tab:blue", label="Costo de cajeros")
ax.plot(tabla["c"], tabla["Costo espera teo (Q/h)"], "--", color="tab:orange", label="Costo de espera")
ax.plot(tabla["c"], tabla["Costo total teo (Q/h)"], "k-", label="Costo total (teoría)")
ax.errorbar(tabla["c"], tabla["Costo total sim (Q/h)"], yerr=tabla["IC 95% (+/-)"], fmt="o",
            color="tab:red", capsize=4, label="Costo total (simulación)")
fila_opt = tabla[tabla["c"] == c_opt].iloc[0]
ax.annotate("óptimo: c* = %d\nQ%.0f / hora" % (c_opt, fila_opt["Costo total sim (Q/h)"]),
            (c_opt, fila_opt["Costo total sim (Q/h)"]), xytext=(c_opt + 0.6, fila_opt["Costo total sim (Q/h)"] + 80),
            arrowprops=dict(arrowstyle="->"))
ax.set_xlabel("Número de cajeros c")
ax.set_ylabel("Costo (Q por hora)")
ax.set_title("Costo total según el número de cajeros")
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc5_costo_estacionario.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 2. Sensibilidad: como cambia c* si cambia la relacion Cw / Cs
# ---------------------------------------------------------------
razones = np.logspace(-1, 2, 300)
c_optimos = []
for k in razones:
    costos = {c: c + k * sim.teoria_mmc(lam, mu, c)["Lq"] for c in range(5, 16)}
    c_optimos.append(min(costos, key=costos.get))

plt.figure(figsize=(7, 4))
plt.step(razones, c_optimos, where="post")
plt.axvline(COSTO_ESPERA / COSTO_CAJERO, color="r", ls="--", lw=1,
            label="Caso base: Cw/Cs = %.2f" % (COSTO_ESPERA / COSTO_CAJERO))
plt.xscale("log")
plt.xlabel("Relación costo de espera / costo de cajero")
plt.ylabel("Número óptimo de cajeros c*")
plt.title("Sensibilidad de la solución óptima")
plt.grid(alpha=0.3, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc5_sensibilidad.png"), dpi=150)
plt.close()

cambios = [(razones[i], c_optimos[i]) for i in range(1, len(razones)) if c_optimos[i] != c_optimos[i - 1]]
print("\nCambios de c* segun Cw/Cs:")
for k, c in cambios:
    print("  a partir de Cw/Cs = %.2f -> c* = %d" % (k, c))


# 3. Dia con horas pico: mejor combinacion de cajeros fijos y de refuerzo
# costo del dia = Cs * horas-cajero + Cw * horas totales de espera
# se usan los mismos dias simulados para todos los planes

R_dias = 50
fijos_op = [3, 4, 5, 6]
refuerzo_op = [0, 1, 2, 3]
costo = np.zeros((len(fijos_op), len(refuerzo_op)))
costos_por_dia = {}
filas = []
for i, f in enumerate(fijos_op):
    for j, rf in enumerate(refuerzo_op):
        turnos = banco.plan_turnos(f, rf)
        horas = banco.horas_cajero(turnos)
        res = []
        for r in range(R_dias):
            rng = sim.crear_generador(5000 + r)
            res.append(banco.resumen_dia(*banco.simular_dia(rng, turnos)))
        espera_h = np.array([x["horas_espera"] for x in res])
        wq = np.array([x["Wq"] for x in res])
        p10 = np.array([x["P_mas_10"] for x in res])
        costo_dia = COSTO_CAJERO * horas + COSTO_ESPERA * espera_h
        m, h = sim.intervalo_confianza(costo_dia)
        costo[i, j] = m
        costos_por_dia[(f, rf)] = costo_dia
        filas.append([f, rf, horas, wq.mean(), p10.mean() * 100,
                      COSTO_CAJERO * horas, COSTO_ESPERA * espera_h.mean(), m, h])

planes = pd.DataFrame(filas, columns=["Fijos", "Refuerzo", "Horas-cajero", "Wq (min)",
                                      "% espera > 10 min", "Costo cajeros (Q)", "Costo espera (Q)",
                                      "Costo total (Q/dia)", "IC 95% (+/-)"])
planes = planes.sort_values("Costo total (Q/dia)").reset_index(drop=True)
print("\n=== Planes para un dia con horas pico (ordenados por costo) ===")
print(planes.round(2).to_string(index=False))
planes.round(3).to_csv(os.path.join(RES, "esc5_planes_dia.csv"), index=False)

mejor = planes.iloc[0]
print("\nPlan optimo: %d fijos + %d de refuerzo, costo = Q%.0f por dia"
      % (mejor["Fijos"], mejor["Refuerzo"], mejor["Costo total (Q/dia)"]))

# como todos los planes se simularon con los mismos dias, se puede comparar
# el mejor contra los siguientes con una diferencia pareada
print("\nComparacion pareada contra el mejor plan:")
p1 = (int(mejor["Fijos"]), int(mejor["Refuerzo"]))
for k in range(1, 4):
    otro = planes.iloc[k]
    p2 = (int(otro["Fijos"]), int(otro["Refuerzo"]))
    dif = costos_por_dia[p2] - costos_por_dia[p1]
    m, h = sim.intervalo_confianza(dif)
    signif = "si" if m - h > 0 else "no"
    print("  %d+%d menos %d+%d: Q%.1f +/- %.1f  (diferencia significativa: %s)"
          % (p2[0], p2[1], p1[0], p1[1], m, h, signif))

fig, ax = plt.subplots(figsize=(7, 5))
im = ax.imshow(np.minimum(costo, 6000), cmap="viridis_r", origin="lower")
for i in range(len(fijos_op)):
    for j in range(len(refuerzo_op)):
        texto = "Q%.0f" % costo[i, j] if costo[i, j] < 10000 else "> Q10k"
        ax.text(j, i, texto, ha="center", va="center",
                color="white" if costo[i, j] > 3500 else "black", fontsize=9)
i_opt = fijos_op.index(int(mejor["Fijos"]))
j_opt = refuerzo_op.index(int(mejor["Refuerzo"]))
ax.add_patch(plt.Rectangle((j_opt - 0.5, i_opt - 0.5), 1, 1, fill=False, ec="red", lw=3))
ax.set_xticks(range(len(refuerzo_op)))
ax.set_xticklabels(refuerzo_op)
ax.set_yticks(range(len(fijos_op)))
ax.set_yticklabels(fijos_op)
ax.set_xlabel("Cajeros de refuerzo en horas pico")
ax.set_ylabel("Cajeros fijos")
ax.set_title("Costo total de un día (Q)")
plt.colorbar(im, ax=ax, label="Q por día")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc5_planes_dia.png"), dpi=150)
plt.close()
