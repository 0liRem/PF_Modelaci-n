
import subprocess
import sys
import time

scripts = ["esc1_mm1.py", "esc2_mmc.py", "esc3_hora_pico.py",
           "esc4_servicio.py", "esc5_optimizacion.py", "animacion.py"]

for s in scripts:
    print("\n" + "#" * 70 + "\n# " + s + "\n" + "#" * 70)
    t0 = time.time()
    subprocess.run([sys.executable, s], check=True)
    print("(%.1f s)" % (time.time() - t0))
