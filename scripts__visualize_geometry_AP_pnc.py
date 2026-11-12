from pathlib import Path
import sys
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import WORKBOOK, FIGURES_DIR, BEAM_WIDTH_NM
from scripts.common import read_holes, validate_holes, beam_extents_nm

COLORS = {"Mirror":"#4575b4", "Taper":"#fdae61", "Central cavity":"#d73027", "Mirror transition":"#74add1"}
def draw(df, path, xlim=None):
    xmin, xmax, ymin, ymax = beam_extents_nm(df)
    fig, ax = plt.subplots(figsize=(15, 3.8))
    ax.add_patch(Rectangle((xmin/1000, ymin/1000), (xmax-xmin)/1000, (ymax-ymin)/1000, facecolor="#f2f2f2", edgecolor="k"))
    for _, row in df.iterrows():
        x=row.position_nm/1000; r=row.diameter_nm/2000
        ax.add_patch(Circle((x,0), r, facecolor=COLORS.get(row.region,"#999999"), edgecolor="k", lw=.7))
        ax.text(x, .62, str(int(row.hole_number)), ha="center", va="bottom", fontsize=7)
    ax.set(xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
    # Explicit limits are required because Matplotlib patches do not always
    # contribute to autoscaling when added directly to an Axes.
    ax.set_xlim(*(xlim if xlim else (xmin/1000*1.03, xmax/1000*1.03)))
    ax.set_ylim(-BEAM_WIDTH_NM/2000*1.25, BEAM_WIDTH_NM/2000*1.25)
    ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)

def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    df,_,_=read_holes(WORKBOOK); df=validate_holes(df)
    draw(df, FIGURES_DIR/"pnc_geometry.png")
    draw(df, FIGURES_DIR/"pnc_geometry_center_zoom.png", (-3.0, 3.0))

if __name__ == "__main__": main()
