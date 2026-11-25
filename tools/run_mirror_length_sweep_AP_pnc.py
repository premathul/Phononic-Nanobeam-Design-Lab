"""Fixed 1 GHz periodic-mirror length sweep. Athul Prem | _AP_pnc."""
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.update(dict.fromkeys(("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS"), "1"))
root = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
out.mkdir(exist_ok=True)
sys.path[:0] = [str(root), str(root / "src")]
import numpy as np
from tools import finite_mirror_response_AP_pnc as finite
from tools import run_mirror_cell_phase_AP_pnc as cell
from phononic.port_modes_AP_pnc import assemble_port, solve_port_modes

folder = Path(sys.argv[3]).resolve()
receipt = finite._read(folder / cell.RECEIPT)
if receipt["status"] != "accepted" or receipt["inputs_sha256"] != finite.digest(receipt["inputs"]):
    raise ValueError("Accepted cell receipt or input digest missing")
archive = finite._verified(receipt["trace_arrays"], folder)
with np.load(archive, allow_pickle=False) as saved:
    if saved["inputs_sha256"].item() != receipt["inputs_sha256"] or saved["frequency_hz"].item() != 1e9:
        raise ValueError("Saved Schur matrices are not the 1 GHz frozen cell")
geometry, mesh = receipt["inputs"]["geometry"], receipt["inputs"]["mesh"]
args = geometry["thickness_nm"], mesh["central_target_nm"], mesh["thickness_layers"]
spec = cell._case_spec(*args)
if spec.inputs != {key: receipt["inputs"][key] for key in spec.inputs}:
    raise ValueError("Geometry/material controls changed")
system = cell.assemble_case(*args)
actual = finite._array_digest(system.mesh.p, system.mesh.t)
if actual != receipt["mesh"]["geometry_connectivity_sha256"]:
    raise ValueError(f"Rebuilt mesh identity differs: {actual} vs {receipt['mesh']['geometry_connectivity_sha256']}")
port = assemble_port(system, spec.material, max_dofs=mesh["max_face_dofs"])
modes = solve_port_modes(port, spec.inputs["frequency_hz"])

def amplitude(r, name):
    return np.asarray(r[name]["real"]) + 1j*np.asarray(r[name]["imag"])

def restricted_gram(r):
    roots = r["native_incident_root_ids"]
    t = amplitude(r,"transmission_amplitudes")[:,[roots.index(i) for i in focus_roots]]
    return t.conj().T @ t

def metrics(r):
    roots = r["native_incident_root_ids"]
    col = [roots.index(i) for i in focus_roots]
    t, ref = (amplitude(r,k) for k in ("transmission_amplitudes","reflection_amplitudes"))
    t, ref = t[:,col], ref[:,col]
    g = t.conj().T @ t
    return {
        "cells":r["cell_count"], "restricted_transmission_percent":(100*np.linalg.eigvalsh(g)).tolist(),
        "control_transmission_percent":100*r["transmission"][roots.index(control_root)],
        "restricted_flux_closure":float(np.linalg.norm(g + ref.conj().T @ ref - np.eye(2),2)),
        "maximum_original_residual":max(r["relative_residuals"].values()),
        "maximum_power_error":max(r["power_errors"].values()),
    }

def write(name, obj):
    with (out / name).open("x") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")

# Define a fixed local port span by the two lowest single-mode transmissions at eight
# cells; keep the same inputs throughout. This is a diagnostic, not prediction on
# the archived Windows root IDs. Among remaining modes use the most suppressed
# one-cell mode as a prespecified transmission control.
anchor = {n:finite.solve_finite_mirror_response(archive,modes,n) for n in (1,8)}
if any(x["status"] != "accepted" for x in anchor.values()):
    raise ValueError("Local endpoint gate failed")
r8 = anchor[8]
roots = r8["native_incident_root_ids"]
idx = np.argsort(r8["transmission"])
focus_roots = [roots[i] for i in idx[:2]]
control_root = min((r for r in roots if r not in focus_roots),
                   key=lambda r: anchor[1]["transmission"][roots.index(r)])
results, grams = {}, {}
for n in range(1,9):
    r = anchor[n] if n in anchor else finite.solve_finite_mirror_response(archive,modes,n)
    if r["status"] != "accepted":
        raise ValueError(f"Length {n} failed equation/power gates")
    if r["native_incident_root_ids"] != roots or r["native_port"] != r8["native_port"]:
        raise ValueError(f"Length {n} native port basis differs")
    m = metrics(r)
    if m["restricted_flux_closure"] > 1e-8:
        raise ValueError(f"Length {n} coherent closure failed")
    write(f"finite_{n}_local_AP_pnc.json", {"result":r,"metrics":m})
    results[n] = m
    grams[n] = restricted_gram(r)
    print(json.dumps(m, allow_nan=False),flush=True)
write("length_sweep_AP_pnc.json", {
    "scope":"One 1 GHz lossless discrete periodic mirror, coarse 225 nm central mesh, 100 nm film, two thickness layers; not the tapered device",
     "source_cell_sha256":receipt["trace_arrays"]["sha256"], "mesh_sha256":actual,
    "environment":{"python":sys.version.split()[0],"numpy":np.__version__,"scipy":__import__("scipy").__version__,"gmsh":__import__("gmsh").__version__,"platform":__import__("platform").platform()},
    "focus_rule":"Two least transmitting individual native inputs at eight cells, held fixed for all lengths. Local root IDs, not cross-platform labels.",
    "control_rule":"Among remaining inputs, least transmitting one-cell mode, held fixed.",
    "adjacent_decrease_pp":[{"from":n,"to":n+1,"bounds":(100*np.linalg.eigvalsh(grams[n]-grams[n+1])).tolist()} for n in range(1,8)],
    "one_to_eight_decrease_pp":(100*np.linalg.eigvalsh(grams[1]-grams[8])).tolist(),
    "root_ids":focus_roots, "control_root_id":control_root, "native_port":r8["native_port"], "results":[results[n] for n in sorted(results)]
})
