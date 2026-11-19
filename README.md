# Simulación de un sistema de atención: teoría de colas M/M/c y optimización del número de cajeros

Proyecto final de **Modelación y Simulación**, Universidad del Valle de Guatemala (Opción 2, tema 17).

| Integrante | Carné |
|---|---|
| Olivier Viau | 23544 |
| Milton Polanco | 23471 |

## Descripción

Se simula la atención de clientes en una agencia bancaria (8:00 a 17:00) usando teoría de colas.
El simulador por eventos discretos se valida primero contra las fórmulas teóricas de M/M/1 y M/M/c,
y luego se usa en situaciones sin solución exacta: llegadas variables durante el día (horas pico,
proceso de Poisson no homogéneo con *thinning*), tiempos de servicio no exponenciales (M/G/1, M/G/c)
y la búsqueda del número de cajeros y plan de turnos que minimiza el costo total
(costo de cajeros + costo de espera de los clientes).

## Requisitos

- Python 3
- Librerías de `requirements.txt` (numpy, scipy, pandas, matplotlib, pillow)

```bash
pip install -r requirements.txt
```

## Uso

Correr todos los escenarios en orden (tarda unos minutos, la animación es lo más lento):

```bash
python correr_todo.py
```

También se puede correr cada escenario por separado, por ejemplo `python esc3_hora_pico.py`.
Para ver la animación de un día en una ventana:

```bash
python animacion.py mostrar
```

Las gráficas se guardan en `figuras/` y las tablas en `resultados/`.

## Estructura

| Archivo | Contenido |
|---|---|
| `simulador.py` | Generación de variables aleatorias, simulación de la cola (con y sin turnos), medidas de desempeño, réplicas, intervalos de confianza y fórmulas teóricas. |
| `modelo_banco.py` | Tasa de llegadas λ(t), ventanas de refuerzo y planes de turnos de la agencia. |
| `esc1_mm1.py` | Escenario 1: pruebas de ajuste, validación de M/M/1 y barrido de ρ. |
| `esc2_mmc.py` | Escenario 2: M/M/c, distribución de la espera y fila única contra filas separadas. |
| `esc3_hora_pico.py` | Escenario 3: día con horas pico y comparación de planes de personal. |
| `esc4_servicio.py` | Escenario 4: M/G/1 y M/G/5 con distintas distribuciones de servicio (Pollaczek-Khinchine, Allen-Cunneen). |
| `esc5_optimizacion.py` | Escenario 5: costo total, sensibilidad y búsqueda del mejor plan. |
| `animacion.py` | Animación de un día en la agencia (GIF en `figuras/`). |
| `correr_todo.py` | Ejecuta todos los escenarios y la animación. |
| `informe/` | Informe del proyecto (LaTeX y PDF). |
| `figuras/` | Gráficas y animaciones generadas. |
| `resultados/` | Tablas CSV generadas. |
| `Proyecto_Final_Modsim.pdf` | Enunciado del proyecto. |
| `Presentacion Simulación de colas M M c.pdf` | Presentación del proyecto. |
