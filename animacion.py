"""
Animacion de un dia en la agencia bancaria.
Arriba se ven las cajas (verde = atendiendo, gris = libre, blanco = fuera de turno)
y la fila de clientes. Abajo, el largo de la cola a lo largo del dia.

Uso:
    python animacion.py            -> guarda los GIF en la carpeta figuras
    python animacion.py mostrar    -> abre la animacion en una ventana
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle

import simulador as sim
import modelo_banco as banco

CARPETA = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(CARPETA, "figuras")
os.makedirs(FIG, exist_ok=True)

PASO = 2            # minutos simulados por cuadro
POR_FILA = 15       # clientes por fila en el dibujo
MAX_DIBUJO = 45     # maximo de clientes dibujados en la fila


def animar(fijos, refuerzo, semilla=42, archivo=None, captura=None):
    turnos = banco.plan_turnos(fijos, refuerzo)
    n_cajas = len(turnos)
    rng = sim.crear_generador(semilla)
    llegadas, _ = sim.llegadas_no_homogeneas(banco.tasa_llegadas, banco.TASA_MAX, banco.DURACION, rng)
    servicios = sim.exponencial(banco.MU, len(llegadas), rng)
    inicio, fin, servidor = sim.simular_cola_turnos(llegadas, servicios, turnos, devolver_servidor=True)

    t_final = max(banco.DURACION, fin.max())
    tiempos = np.arange(0, t_final + PASO, PASO)
    cola = np.searchsorted(llegadas, tiempos, side="right") - np.searchsorted(inicio, tiempos, side="right")

    fig = plt.figure(figsize=(11, 6.5))
    ax = fig.add_axes([0.05, 0.42, 0.9, 0.53])
    ax2 = fig.add_axes([0.08, 0.08, 0.86, 0.26])

    # dibujo de las cajas
    ax.set_xlim(-1, POR_FILA + 1)
    ax.set_ylim(-4.5, 3)
    ax.axis("off")
    cajas = []
    sep = POR_FILA / n_cajas
    for k in range(n_cajas):
        x = k * sep + sep / 2 - 0.6
        r = Rectangle((x, 1.2), 1.2, 1.0, ec="black", lw=1.5)
        ax.add_patch(r)
        etiqueta = "Caja %d" % (k + 1) if k < fijos else "Refuerzo %d" % (k - fijos + 1)
        ax.text(x + 0.6, 2.45, etiqueta, ha="center", fontsize=8)
        cajas.append(r)
    puntos = ax.scatter([], [], s=90, color="tab:orange", edgecolor="k", zorder=3)
    reloj = ax.text(-0.8, -4.3, "", fontsize=13, fontweight="bold")
    info = ax.text(POR_FILA + 0.8, -4.3, "", fontsize=10, ha="right")
    ax.text(-0.8, 0.5, "Fila", fontsize=10, style="italic")
    if refuerzo > 0:
        titulo = "%d cajeros fijos + %d de refuerzo en horas pico" % (fijos, refuerzo)
    else:
        titulo = "%d cajeros fijos todo el día" % fijos
    ax.set_title("Simulación de un día: " + titulo, fontsize=12)
    for j, (color, texto) in enumerate([("tab:green", "atendiendo"), ("lightgray", "libre"),
                                        ("white", "fuera de turno")]):
        ax.add_patch(Rectangle((4 + 3 * j, -3.7), 0.35, 0.35, fc=color, ec="black"))
        ax.text(4.5 + 3 * j, -3.65, texto, fontsize=8)

    # grafica del largo de la cola
    ax2.set_xlim(0, t_final)
    ax2.set_ylim(0, max(cola.max() + 2, 10))
    marcas = np.arange(0, t_final + 1, 60)
    ax2.set_xticks(marcas)
    ax2.set_xticklabels(["%d:00" % (banco.APERTURA + m // 60) for m in marcas])
    ax2.set_ylabel("Clientes en cola")
    ax2.grid(alpha=0.3)
    for entrada, salida in banco.VENTANAS_PICO:
        if refuerzo > 0:
            ax2.axvspan(entrada, min(salida, banco.DURACION), color="tab:green", alpha=0.08)
    linea, = ax2.plot([], [], color="tab:blue", lw=1.5)

    def actualizar(i):
        g = tiempos[i]
        # estado de cada caja
        atendiendo = servidor[(inicio <= g) & (fin > g)]
        for k, horario in enumerate(turnos):
            en_turno = any(entrada <= g < salida for entrada, salida in horario)
            if k in atendiendo:
                cajas[k].set_facecolor("tab:green")
            elif en_turno and g < banco.DURACION:
                cajas[k].set_facecolor("lightgray")
            else:
                cajas[k].set_facecolor("white")

        # clientes en la fila
        n = int(((llegadas <= g) & (inicio > g)).sum())
        m = min(n, MAX_DIBUJO)
        pos = np.arange(m)
        xs = pos % POR_FILA
        ys = -(pos // POR_FILA) * 1.2
        puntos.set_offsets(np.column_stack([xs, ys]) if m > 0 else np.empty((0, 2)))

        h = banco.APERTURA + int(g) // 60
        reloj.set_text("%02d:%02d" % (h, int(g) % 60))
        atendidos = int((fin <= g).sum())
        info.set_text("En cola: %d   |   Atendidos: %d   |   Llegadas: %d"
                      % (n, atendidos, int((llegadas <= g).sum())))
        linea.set_data(tiempos[:i + 1], cola[:i + 1])
        return cajas + [puntos, reloj, info, linea]

    if captura is not None:
        # imagen fija a la 1:00 pm para el informe
        actualizar(int(np.searchsorted(tiempos, 300)))
        fig.savefig(captura, dpi=150)

    anim = FuncAnimation(fig, actualizar, frames=len(tiempos), interval=60, blit=False)
    if archivo is None:
        plt.show()
    else:
        anim.save(archivo, writer=PillowWriter(fps=15))
        plt.close(fig)
        print("Guardado:", archivo)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "mostrar":
        animar(4, 2)
    else:
        animar(5, 0, archivo=os.path.join(FIG, "animacion_5_fijos.gif"),
               captura=os.path.join(FIG, "animacion_5_fijos_13h.png"))
        animar(4, 2, archivo=os.path.join(FIG, "animacion_4_fijos_2_refuerzo.gif"),
               captura=os.path.join(FIG, "animacion_4_fijos_2_refuerzo_13h.png"))
