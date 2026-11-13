"""
Escenario 3: llegadas variables durante el dia (horas pico)
Las llegadas siguen un proceso de Poisson no homogeneo lambda(t).
Aqui ya no hay formula exacta, por eso se simula.
Se comparan varios planes de personal.
"""
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

R = 50
D = banco.DURACION
marcas = np.arange(0, D + 1, 60)
etiquetas = ["%d:00" % (banco.APERTURA + m // 60) for m in marcas]


def eje_horas(ax):
    ax.set_xticks(marcas)
    ax.set_xticklabels(etiquetas)
    ax.set_xlim(0, D)


lam_prom = banco.tasa_promedio()
t = np.linspace(0, D, 1000)
print("lambda promedio = %.2f clientes/h, lambda maximo = %.2f clientes/h"
      % (lam_prom * 60, banco.tasa_llegadas(t).max() * 60))
print("clientes esperados por dia = %.1f" % (lam_prom * D))

# ---------------------------------------------------------------
# 1. Validacion del thinning: llegadas simuladas vs lambda(t)
# ---------------------------------------------------------------
ancho = 15
bordes = np.arange(0, D + ancho, ancho)
conteos = []
aceptadas, candidatas = 0, 0
for r in range(R):
    rng = sim.crear_generador(10 + r)
    lleg, n_cand = sim.llegadas_no_homogeneas(banco.tasa_llegadas, banco.TASA_MAX, D, rng)
    conteos.append(np.histogram(lleg, bordes)[0])
    aceptadas += len(lleg)
    candidatas += n_cand
conteos = np.array(conteos)

print("\nEficiencia del thinning: teorica = %.4f, observada = %.4f"
      % (lam_prom / banco.TASA_MAX, aceptadas / candidatas))

# numero esperado de llegadas en cada intervalo = integral de lambda(t)
esperado = []
for a, b in zip(bordes[:-1], bordes[1:]):
    x = np.linspace(a, b, 50)
    esperado.append(np.trapezoid(banco.tasa_llegadas(x), x))

fig, ax = plt.subplots(figsize=(10, 4))
centros = bordes[:-1] + ancho / 2
ax.bar(centros, conteos.mean(axis=0) * 60 / ancho, width=ancho * 0.9, alpha=0.5,
       label="Llegadas simuladas (promedio de %d días)" % R)
ax.plot(t, banco.tasa_llegadas(t) * 60, "r", lw=2, label="$\\lambda(t)$")
ax.axhline(lam_prom * 60, color="k", ls="--", lw=1, label="$\\lambda$ promedio = %.1f" % (lam_prom * 60))
ax.set_ylabel("Clientes por hora")
ax.set_title("Tasa de llegadas durante el día")
eje_horas(ax)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc3_tasa_llegadas.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 2. Comparacion de planes de personal
# ---------------------------------------------------------------
planes = {
    "5 fijos": banco.plan_turnos(5, 0),
    "6 fijos": banco.plan_turnos(6, 0),
    "4 fijos + 2 refuerzo": banco.plan_turnos(4, 2),
    "5 fijos + 1 refuerzo": banco.plan_turnos(5, 1),
}

filas = []
datos_planes = {}
for nombre, turnos in planes.items():
    dias = []
    res = []
    for r in range(R):
        rng = sim.crear_generador(1000 + r)   # mismos dias para todos los planes
        lleg, ini, fin = banco.simular_dia(rng, turnos)
        dias.append((lleg, ini, fin))
        res.append(banco.resumen_dia(lleg, ini, fin))
    datos_planes[nombre] = dias
    res = {k: np.array([x[k] for x in res]) for k in res[0]}
    wq, h = sim.intervalo_confianza(res["Wq"])
    filas.append([nombre, banco.horas_cajero(turnos), wq, h, res["Wq_p95"].mean(),
                  res["Wq_max"].mean(), res["P_mas_10"].mean() * 100,
                  (res["cierre_real"].mean() - D)])

tabla = pd.DataFrame(filas, columns=["Plan", "Horas-cajero", "Wq (min)", "IC 95% (+/-)",
                                     "Percentil 95 (min)", "Espera max (min)",
                                     "% espera > 10 min", "Min. despues del cierre"])
print("\n=== Planes de personal (%d dias simulados) ===" % R)
print(tabla.round(2).to_string(index=False))
tabla.round(3).to_csv(os.path.join(RES, "esc3_planes.csv"), index=False)

# que diria la formula de M/M/c si se usara lambda promedio
print("\nSi se usa M/M/c con lambda promedio:")
for c in [4, 5, 6]:
    print("  c = %d -> Wq teorico = %.2f min" % (c, sim.teoria_mmc(lam_prom, banco.MU, c)["Wq"]))

# ---------------------------------------------------------------
# 3. Espera promedio segun la hora de llegada
# ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 4.5))
for nombre, dias in datos_planes.items():
    lleg = np.concatenate([d[0] for d in dias])
    esp = np.concatenate([d[1] - d[0] for d in dias])
    idx = np.digitize(lleg, bordes) - 1
    prom = [esp[idx == k].mean() for k in range(len(bordes) - 1)]
    ax.plot(centros, prom, "-o", ms=3, label="%s (%.1f h-cajero)" % (nombre, banco.horas_cajero(planes[nombre])))
ax.set_ylabel("Espera promedio en cola (min)")
ax.set_title("Tiempo de espera según la hora de llegada")
eje_horas(ax)
ax.grid(alpha=0.3)
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc3_espera_por_hora.png"), dpi=150)
plt.close()

# ---------------------------------------------------------------
# 4. Largo de la cola en el tiempo vs lo que diria la teoria estacionaria
# ---------------------------------------------------------------
c = 5
dias = datos_planes["5 fijos"]
grilla = np.linspace(0, D, 1081)
cola = []
for lleg, ini, fin in dias:
    cola.append(np.searchsorted(lleg, grilla, side="right") - np.searchsorted(ini, grilla, side="right"))
cola = np.array(cola)

lq_estacionario = []
for x in grilla:
    lq = sim.teoria_mmc(banco.tasa_llegadas(x).item(), banco.MU, c)["Lq"]
    lq_estacionario.append(lq if np.isfinite(lq) else np.nan)

fig, ax = plt.subplots(figsize=(10, 4.5))
ax.fill_between(grilla, np.percentile(cola, 10, axis=0), np.percentile(cola, 90, axis=0),
                alpha=0.25, label="Simulación: 10% - 90% de los días")
ax.plot(grilla, cola.mean(axis=0), lw=2, label="Simulación: promedio")
ax.plot(grilla, lq_estacionario, "r--", lw=1.5, label="Fórmula M/M/5 con $\\lambda(t)$")
ax.set_ylabel("Clientes en cola")
ax.set_title("Largo de la cola durante el día con 5 cajas fijas")
ax.set_ylim(0, 20)
eje_horas(ax)
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "esc3_cola_dia.png"), dpi=150)
plt.close()
