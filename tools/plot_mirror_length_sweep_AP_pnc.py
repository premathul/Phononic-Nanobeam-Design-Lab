"""Plot the finite mirror length comparison. Athul Prem | _AP_pnc."""
import argparse
import json
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt

p=argparse.ArgumentParser(description=__doc__)
p.add_argument("data",type=Path)
p.add_argument("figure",type=Path)
args=p.parse_args()
r=json.loads(args.data.read_text())["results"]
n=[x["cells"] for x in r]
lo,hi=zip(*(x["restricted_transmission_percent"] for x in r))
control=[x["control_transmission_percent"] for x in r]
mpl.rcParams["svg.hashsalt"]="mirror_length_AP_pnc"
fig,ax=plt.subplots(figsize=(5.4,3.1),layout="constrained")
ax.fill_between(n,lo,hi,color="#bcdbeb",label="All coherent mixes of the two selected inputs")
ax.plot(n,lo,"o-",color="#135169",ms=4,lw=1.3)
ax.plot(n,hi,"o-",color="#135169",ms=4,lw=1.3)
ax.plot(n,control,"s--",color="#9e401b",label="Control input",ms=4,lw=1.3)
ax.set(xlabel="Number of 550 nm mirror cells",ylabel="Transmitted power (%)",xticks=n,ylim=(50,102))
ax.grid(axis="y",alpha=.25)
ax.legend(loc="lower left",frameon=False,fontsize=8)
fig.savefig(args.figure,metadata={"Date":None})
args.figure.write_text("\n".join(line.rstrip() for line in args.figure.read_text().splitlines())+"\n")
