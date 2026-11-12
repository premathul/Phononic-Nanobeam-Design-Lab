"""Mesh inspection using the Gmsh API; useful when meshio is unavailable."""
from pathlib import Path
import csv, numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Circle, Rectangle

def inspect(mesh_path, out_image, csv_path='data/hole_geometry_derived.csv'):
    import gmsh
    holes=list(csv.DictReader(open(csv_path)))
    gmsh.initialize(); gmsh.open(str(mesh_path))
    tags, coords, _=gmsh.model.mesh.getNodes(); pts=np.asarray(coords).reshape(-1,3)[:,:2]; idx={int(t):i for i,t in enumerate(tags)}
    types, _, blocks=gmsh.model.mesh.getElements(2); tri=[]
    for typ,nodes in zip(types,blocks):
        prop=gmsh.model.mesh.getElementProperties(typ)
        if prop[0].startswith('Triangle') and prop[3]==3: tri.extend(np.asarray(nodes).reshape(-1,3))
    tri=np.asarray(tri,dtype=np.int64); ti=np.vectorize(idx.get)(tri); p=pts[ti]
    edges=np.concatenate([p[:,[0,1]],p[:,[1,2]],p[:,[2,0]]]); lengths=np.linalg.norm(edges[:,1]-edges[:,0],axis=1)
    cents=p.mean(axis=1); inside=0
    for h in holes:
        x=float(h['position_nm']); r=float(h['diameter_nm'])/2
        inside += int(np.count_nonzero((cents[:,0]-x)**2+cents[:,1]**2 < r*r-1e-6))
    seg=np.stack([p[:,:,0],p[:,:,1]],axis=2)/1000.0
    fig,ax=plt.subplots(figsize=(15,3.8)); ax.add_collection(LineCollection(seg,colors='#777777',linewidths=.15));
    ax.add_patch(Rectangle((-6.825,-.5),13.65,1,facecolor='none',edgecolor='k',lw=1))
    for h in holes: ax.add_patch(Circle((float(h['position_nm'])/1000,0),float(h['diameter_nm'])/2000,facecolor='none',edgecolor='#d73027',lw=.5))
    ax.set(xlim=(-8.4,8.4),ylim=(-.65,.65),xlabel='x (µm)',ylabel='y (µm)',aspect='equal'); ax.grid(alpha=.2); fig.tight_layout(); fig.savefig(out_image,dpi=220); plt.close(fig); gmsh.finalize()
    return dict(nodes=len(pts),triangles=len(tri),min_edge_nm=float(lengths.min()),max_edge_nm=float(lengths.max()),elements_across_250nm=250/float(np.median(lengths)),triangles_inside_holes=inside)

if __name__=='__main__':
    import sys
    print(inspect(sys.argv[1],sys.argv[2]))
