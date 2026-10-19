"""Draw the actual CLIP ViT pipeline using schematic planes and token stacks."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--image', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = Image.open(args.image).convert('RGB')
w,h=source.size
resized=source.resize((int(w*224/min(w,h)),int(h*224/min(w,h))),Image.Resampling.BICUBIC)
left=round((resized.width-224)/2)
top=round((resized.height-224)/2)
crop=resized.crop((left,top,left+224,top+224))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'pdf.fonttype':42})
fig,ax=plt.subplots(figsize=(16,5.05))
ax.set(xlim=(0,16),ylim=(0,5.05),aspect='equal')
ax.axis('off')
dark='#303030'

def label(x,y,text,size=12,bold=False):
    ax.text(x,y,text,ha='center',va='center',fontsize=size*1.35,weight='bold' if bold else 'normal',color=dark)

def arrow(x1,x2,y=2.5):
    ax.annotate('',xy=(x2,y),xytext=(x1,y),arrowprops={'arrowstyle':'-|>','lw':1.15,'color':dark},zorder=10)

def plane(x,y,w,h,color,shift=.18):
    pts=[(x,y),(x+w,y-.13),(x+w+shift,y+h-.13),(x+shift,y+h)]
    ax.add_patch(Polygon(pts,closed=True,facecolor=color,edgecolor='#686868',lw=.65,zorder=3))

ax.imshow(source,extent=(.15,2.25,1.72,3.295),zorder=4)
ax.add_patch(Rectangle((.15,1.72),2.1,1.575,fill=False,edgecolor='#757575',lw=.8))
label(1.2,4.25,'Saved RGB',bold=True)
label(1.2,1.05,'800 x 600')
label(1.2,.6,'Single saved frame',10)
arrow(2.32,2.95)

# The image shows the actual crop; grid lines identify the 32-pixel patches.
ax.imshow(crop,extent=(3.05,5.05,1.55,3.55),zorder=4)
for i in range(8):
    q=2*i/7
    ax.plot([3.05+q,3.05+q],[1.55,3.55],color='white',lw=.65,zorder=5)
    ax.plot([3.05,5.05],[1.55+q,1.55+q],color='white',lw=.65,zorder=5)
label(4.05,4.25,'Preprocessing',bold=True)
label(4.05,1.05,'224 x 224 crop')
label(4.05,.6,'Resize + center crop',10)
label(4.05,.25,'RGB normalization',10)
arrow(5.12,5.72)

for i in range(5):
    plane(5.83+i*.115,1.7+i*.11,.55,1.4,str(.8-i*.075),shift=.1)
label(6.36,4.25,'Patch\nembeddings',bold=True)
label(6.36,1.05,'49 patches + CLS')
label(6.36,.6,'32 x 32 pixels / patch',10)
label(6.36,.25,'50 tokens, width 768',10)
arrow(7.02,7.6)

for i in range(4):
    plane(7.8+i*.34,1.55+i*.09,1.05,1.9,str(.79-i*.13))
    ax.plot([8.02+i*.34,8.65+i*.34],[2.05+i*.09,1.97+i*.09],color='#999999',lw=.7,zorder=4)
label(8.95,4.25,'Frozen visual\nencoder',bold=True)
label(8.95,1.05,'12 blocks')
label(8.95,.6,'Self-attention + MLP',10)
label(8.95,.25,'ViT-B-32 / QuickGELU',10)
arrow(10.16,10.78)

# Representative coordinates illustrate the linear projection, not activations.
ys=[1.75,2.1,2.45,2.8,3.15]
for y in ys:
    for yy in ys:
        ax.plot([11.0,11.82],[y,yy],color='#b8b8b8',lw=.45,zorder=1)
for x,col in [(11.0,'#b7b7b7'),(11.82,'#737373')]:
    for y in ys:
        ax.add_patch(Circle((x,y),.095,facecolor=col,edgecolor='#505050',lw=.6,zorder=4))
label(11.41,4.25,'Class-token\nreadout',bold=True)
label(11.41,1.05,'768 to 512')
label(11.41,.6,'Norm + projection',10)
arrow(12.02,12.9)
label(12.49,3.08,'L2 norm',10)

for i,y in enumerate([1.65,1.92,2.19,2.46,2.73,3.0,3.27]):
    ax.add_patch(Rectangle((13.12,y),1.12,.19,facecolor=str(.37+.06*i),edgecolor='#6a6a6a',lw=.6))
label(14.55,2.57,r'$\cdots$',17)
label(13.8,4.25,'Image\nembedding',bold=True)
label(13.8,1.05,'512 dimensions')
label(13.8,.6,r'$\|z\|_2 = 1$',12)

fig.tight_layout(pad=.05)
fig.savefig(args.output/'clip_architecture.pdf',bbox_inches='tight',pad_inches=.025)
fig.savefig(args.output/'clip_architecture.png',dpi=220,bbox_inches='tight',pad_inches=.025)
fig.savefig(args.output/'clip_architecture.svg',bbox_inches='tight',pad_inches=.025)
plt.close(fig)
(args.output/'clip_architecture_provenance.json').write_text(json.dumps({'source_image':str(args.image),'source_sha256':hashlib.sha256(args.image.read_bytes()).hexdigest(),'source_size':source.size,'model':'ViT-B-32-quickgelu','patch_size':32,'image_size':224,'patch_tokens':49,'class_tokens':1,'width':768,'transformer_layers':12,'embedding_dim':512,'note':'Planes, token stacks and node coordinates are schematic, not measured activations. Image grid marks patches; normalization is applied after the displayed crop.'},indent=2))
