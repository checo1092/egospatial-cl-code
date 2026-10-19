"""Reproduce the weather/traffic mosaic from extracted release episodes."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--selection',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True)
width=7.09;gap=.035;pw=(width-3*gap)/4;ph=pw*.75;height=2*ph+gap
fig=plt.figure(figsize=(width,height))
for r in json.loads(a.selection.read_text())['frames']:
 path=a.dataset_root/r['rgb_path']
 if not path.exists():
  parts=Path(r['rgb_path']).parts;path=a.dataset_root/parts[0]/Path(*parts[2:])
 im=Image.open(path).convert('RGB');assert im.size==(800,600)
 ri,ci=r['row']-1,r['column']-1
 ax=fig.add_axes([ci*(pw+gap)/width,(height-(ri+1)*ph-ri*gap)/height,pw/width,ph/height]);ax.imshow(im,interpolation='none');ax.axis('off')
fig.savefig(a.output/'weather_traffic.pdf');fig.savefig(a.output/'weather_traffic.png',dpi=400);plt.close(fig)
