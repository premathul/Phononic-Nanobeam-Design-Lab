"""Deterministic staged driver; refuses to launch a full sweep after failed gates."""
from pathlib import Path
import subprocess,sys,argparse
ROOT=Path(__file__).resolve().parents[1]; PY=sys.executable
def run(script,*args): subprocess.run([PY,str(ROOT/"scripts"/script),*args],cwd=ROOT,check=True)
def main():
    run("inspect_excel.py"); run("visualize_geometry.py")
    run("build_geometry.py"); run("build_geometry.py","--reference","--name","reference_ge_beam")
    run("build_mesh.py"); run("build_mesh.py","--name","reference_ge_beam")
    run("run_frequency_sweep.py","--test")
    ap=argparse.ArgumentParser(); ap.add_argument("--full",action="store_true"); a=ap.parse_args()
    if a.full:
        run("run_frequency_sweep.py","--full"); run("analyze_transmission.py")
        print("PASS: full sweep and analysis completed")
    else:
        print("PASS: geometry, mesh, and five-frequency gate completed; use --full to continue")
if __name__ == "__main__": main()
