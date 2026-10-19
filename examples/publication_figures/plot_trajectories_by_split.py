"""Plot the two-domain split map from the public dataset manifest."""
import argparse,json
from pathlib import Path
import numpy as np
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
def pick(rows, domain, slot):
    return next(r for r in rows if r["domain"] == domain and r["slot"] == slot)

PUB_DOMAINS = [("clear_NPC00_none", "Clear \u2013 no NPCs"),
               ("clear_NPC31_both_high", "Clear \u2013 vehicles and pedestrians")]

PUB_ROW_LABELS = {"clear_NPC00_none": "Clear\nno NPCs",
                  "clear_NPC31_both_high": "Clear\nvehicles and\npedestrians"}

PUB_BLOCKS = [["Town01_Opt", "Town02_Opt", "Town04_Opt", "Town05_Opt"], ["Town06", "Town07", "Town10HD_Opt"]]

PUB_SPLIT_COLOR = {"train": "#0072B2", "val": "#E69F00", "test": "#CC79A7"}

PUB_SPLIT_LABEL = {"train": "Train", "val": "Validation", "test": "Test"}

def plot_publication_blocks(town_data, out_stem, blocks=PUB_BLOCKS, width_in=7.09, max_height_in=6.69,
                            color_by="route"):
    """Publication figure: each block is a row-group (rows = PUB_DOMAINS, columns = the block's towns).
    Every block spans the full width; column widths follow the town aspect ratio (no letterboxing) and rows in a
    block share one height. Town names above each block, scale bar under each column, one color key at the bottom.
    Sized for Scientific Data / Nature Portfolio: width 180 mm (7.09 in), height <= 170 mm (6.69 in): if the natural
    height is larger, all rows/columns are scaled by one common factor and centred.
    color_by = "route" (10 colours = slot order) or "split" (train / validation / test, Okabe-Ito).
    Writes <out_stem>.png (300 dpi) and <out_stem>.pdf (vector). Returns (W, H, [row_h per block])."""
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    cmap = plt.get_cmap("tab10")
    left, right, top, cgap, rgap = 0.42, 0.04, 0.24, 0.05, 0.05                   # inches
    bottom = 0.52 if color_by == "split" else 0.80                                  # scale bar + colour key
    bgap = 0.42                                                                     # between blocks (scale bar + titles)
    doms = [d for d, _ in PUB_DOMAINS]
    nd = len(doms)
    plan = []
    for blk in blocks:
        geo = []
        for town in blk:
            rows = town_data[town][0]
            ax_ = np.concatenate([r["x"] for r in rows if r["domain"] in doms])
            ay_ = np.concatenate([r["y"] for r in rows if r["domain"] in doms])
            pad = 0.03 * max(np.ptp(ax_), np.ptp(ay_))
            xl = (ax_.min() - pad, ax_.max() + pad); yl = (ay_.min() - pad, ay_.max() + pad)
            geo.append((xl, yl, (yl[1] - yl[0]) / (xl[1] - xl[0])))
        avail = width_in - left - right - cgap * (len(blk) - 1)
        row_h = avail / sum(1.0 / g[2] for g in geo)
        plan.append((geo, row_h, [row_h / g[2] for g in geo]))
    fixed = top + (nd - 1) * rgap * len(blocks) + bgap * (len(blocks) - 1) + bottom
    sc = min(1.0, (max_height_in - fixed) / sum(nd * p[1] for p in plan))           # common shrink to fit max height
    plan = [(g, rh * sc, [w * sc for w in ws]) for g, rh, ws in plan]
    blk_h = [nd * rh + (nd - 1) * rgap for _, rh, _ in plan]
    H = top + sum(blk_h) + bgap * (len(blocks) - 1) + bottom
    fig = plt.figure(figsize=(width_in, H))
    ytop = H - top
    for b, blk in enumerate(blocks):
        geo, row_h, widths = plan[b]
        x0 = left + ((width_in - left - right) - (sum(widths) + cgap * (len(blk) - 1))) / 2   # centre the block
        for i, town in enumerate(blk):
            rows, _, slots, _ = town_data[town]
            xl, yl, _a = geo[i]
            for j, dom in enumerate(doms):
                y0 = ytop - (j + 1) * row_h - j * rgap
                ax = fig.add_axes([x0 / width_in, y0 / H, widths[i] / width_in, row_h / H])
                for k, sp in sorted(enumerate(slots), key=lambda t: {"train": 0, "val": 1, "test": 2}[town_data[town][3][t[1]]]):
                    r = pick(rows, dom, sp)
                    spl = town_data[town][3][sp]
                    if color_by == "split":                                     # train underneath, val/test on top
                        col, lw, al, z = PUB_SPLIT_COLOR[spl], (1.0 if spl == "train" else 1.4), (0.75 if spl == "train" else 1.0), \
                            {"train": 2, "val": 3, "test": 4}[spl]
                    else:
                        col, lw, al, z = cmap(k), 1.0, 1.0, 2
                    ax.plot(r["x"], r["y"], color=col, lw=lw, alpha=al, solid_capstyle="round", zorder=z)
                    ax.plot(r["x"][0], r["y"][0], "o", color=col, ms=2.4, mec="white", mew=0.35, zorder=5)
                ax.set_xlim(*xl); ax.set_ylim(yl[1], yl[0])                     # y inverted (CARLA top view)
                ax.set_xticks([]); ax.set_yticks([])
                for sd in ax.spines.values():
                    sd.set_color("#999"); sd.set_linewidth(0.5)
                if j == 0:
                    ax.set_title(town.replace("_Opt", ""), fontsize=7, fontweight="bold", pad=3)
                if i == 0:
                    ax.set_ylabel(PUB_ROW_LABELS[dom], fontsize=6, labelpad=3, linespacing=1.15)
                if j == nd - 1:                                                 # scale bar under each column
                    span = xl[1] - xl[0]
                    nice = max(n for n in (25, 50, 100, 200, 250, 500) if n <= 0.5 * span)
                    xa = xl[0] + 0.03 * span
                    base = ax.get_xaxis_transform()
                    tr_line = mtransforms.offset_copy(base, fig=fig, y=-5, units="points")
                    tr_text = mtransforms.offset_copy(base, fig=fig, y=-8, units="points")
                    ax.plot([xa, xa + nice], [0, 0], color="#222", lw=0.9, transform=tr_line,
                            clip_on=False, solid_capstyle="butt")
                    ax.text(xa, 0, f"{nice} m", transform=tr_text, fontsize=6, color="#222", va="top", ha="left")
            x0 += widths[i] + cgap
        ytop -= blk_h[b] + bgap
    slots0, split0 = town_data[blocks[0][0]][2], town_data[blocks[0][0]][3]
    start = plt.Line2D([], [], marker="o", ls="none", markerfacecolor="#444", markeredgecolor="white",
                       markersize=4.5, label="Start")
    if color_by == "split":
        n_sp = {k: sum(1 for v in split0.values() if v == k) for k in PUB_SPLIT_COLOR}
        handles = [plt.Line2D([], [], color=PUB_SPLIT_COLOR[k], lw=1.8, label=f"{PUB_SPLIT_LABEL[k]} ({n_sp[k]})")
                   for k in PUB_SPLIT_COLOR] + [start]
        ncol, kw = len(handles), dict(labelspacing=0.45)
    else:
        routes = [plt.Line2D([], [], color=cmap(k), lw=1.8, label=f"Route {k + 1} ({split0[sp]})")
                  for k, sp in enumerate(slots0)]
        blank = plt.Line2D([], [], ls="none", label=" ")
        half = (len(routes) + 1) // 2
        cols = [[routes[c], routes[c + half]] for c in range(half)] + [[start, blank]]
        handles = [h for col in cols for h in col]
        ncol, kw = len(cols), dict(labelspacing=0.45)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.03 / H), ncol=ncol,
               frameon=False, fontsize=6.5, handlelength=1.6, handletextpad=0.5, columnspacing=1.5,
               borderaxespad=0, **kw)
    fig.savefig(f"{out_stem}.png", dpi=300)
    fig.savefig(f"{out_stem}.pdf")
    plt.close(fig)
    return width_in, H, [p[1] for p in plan]

def main():
 plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
 p=argparse.ArgumentParser();p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 a.output.mkdir(parents=True,exist_ok=True);town_data={}
 for town in [t for block in PUB_BLOCKS for t in block]:
  rows=[]
  for e in json.loads(a.manifest.read_text())['episodes']:
   domain=e['domain_name'].replace('__','_')
   if e['town']!=town or domain not in dict(PUB_DOMAINS):continue
   path=a.dataset_root/e['episode_uid']/'state.csv'
   if not path.exists():path=a.dataset_root/town/e['episode']/'state.csv'
   with path.open() as handle: positions=np.array([(float(r['ego_x']),float(r['ego_y'])) for r in csv.DictReader(handle)])
   rows.append(dict(domain=domain,slot=e['spatial_slot'],split=e['split'],x=positions[:,0],y=positions[:,1]))
  slots=list(dict.fromkeys(r['slot'] for r in rows));assert len(rows)==20 and len(slots)==10,(town,len(rows))
  town_data[town]=(rows,list(dict(PUB_DOMAINS)),slots,{r['slot']:r['split'] for r in rows})
 plot_publication_blocks(town_data,a.output/'trajectories_by_split',color_by='split')
if __name__=='__main__':main()
