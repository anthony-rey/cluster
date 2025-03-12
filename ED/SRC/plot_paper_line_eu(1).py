import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from scipy.optimize import curve_fit
import argparse
from __func__ import *
import seaborn as sns
import pandas as pd

plt.rcParams.update({
	'text.usetex': True,
	'font.family': 'serif',
	'font.size': 18,
	'legend.fontsize': 18,
	'legend.fancybox': True,
	'lines.solid_capstyle': "round",
	'savefig.dpi': 250,
	'savefig.bbox': 'tight',
	'savefig.format': 'pdf',
	"axes.linewidth": 0.8,
	"legend.edgecolor": "0.8",
	"patch.linewidth": 0.8,
	"xtick.major.size": 6,
	"xtick.major.width": 0.8,
	"xtick.minor.size": 3,
	"xtick.minor.width": 0.8,
	"ytick.major.size": 6,
	"ytick.major.width": 0.8,
	"ytick.minor.size": 3,
	"ytick.minor.width": 0.8
	})

colors = [[0.8, 0, 0.5],[0, 0.5, 0.8],[0.8, 0.5, 0]]

# mpl.rc('text.latex', preamble=r"\usepackage{mathpazo}")
# mpl.rc('text.latex', preamble=r"\usepackage{eulervm}")

dataFolder = "../data/line•EU(1)"
graphsFolder = "../graphs/line•EU(1)"
expression = fr"(?:pbc=)(?P<pbc>[0-9e\.-]+)(?:_arpack=)(?P<arpack>[0-9e\.-]+)(?:_N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"
folders = folderNames(dataFolder)

parser = argparse.ArgumentParser()

parser.add_argument("--save", help="save graph.s", action="store_true")
parser.add_argument("--show", help="show graph.s", action="store_true")

args = parser.parse_args()
			
nE = 10
nChoose = 18

useArpack = 1
PBCs = 1

Ns = []
diffsNs = []
		
for folder in folders:
	
	_, vals, dataFilename = loadFile(os.path.join(dataFolder, folder), expression)

	if np.unique([int(vals[i]['pbc']) for i in range(len(vals))])==PBCs and np.unique([int(vals[i]['arpack']) for i in range(len(vals))])==useArpack and np.unique([int(vals[i]['N']) for i in range(len(vals))])<nChoose:
		
		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))])[0]
		Ns.append(N)

		diff = np.zeros((len(lam), len(theta), nE-1))

		for i in range(len(dataFilename)):

			with open(os.path.join(dataFolder, folder, dataFilename[i]), 'rb') as file:
				engine = pickle.load(file)
		
			# ****************
			# print(dataFilename[i])
			# ****************

			E = np.array(engine.E)

			if useArpack:
				E = -E[::-1]

			gaps = E[1:] - E[:-1]

			diff[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = gaps[:nE-1]
				
		diffsNs.append(diff)

ind = np.argsort(Ns)

Ns = np.array(Ns)[ind]
diffsNs = np.array(diffsNs)[ind, :, :, :]

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[colors[6],colors[7],colors[0],colors[1],colors[2],colors[3],colors[5]], N=256)
plt.rcParams.update({'axes.prop_cycle': cycler(marker=([".", "|", "+"]*len(Ns))[:len(Ns)], color=cmap(np.linspace(0, 1, len(Ns))))})

if args.save:
	fig, ax = plt.subplots(1, 4, figsize=(15, 4))

for l in range(len(lam)):
	for i in range(4):

		for n in range(len(Ns)):
			p = ax[i].plot(theta, Ns[n]*diffsNs[n, l, :, i], alpha=1, linestyle=":", markeredgewidth=1.4, markersize=6, linewidth=0.6, zorder=100, label=fr"$2N = {Ns[n]}$")
			[p[s].set_marker(mpl.markers.MarkerStyle(p[s].get_marker(), capstyle="round")) for s in range(len(p))]

		ax[i].axvline(np.pi/4, color="k", linewidth=1, linestyle="--")
		ax[i].minorticks_on()
		ax[i].set_xlabel(r"$\theta$")
		if i==3:
			ax[i].set_title(fr"$2N(E^{{\,}}_{{{i+1}}} - E^{{\,}}_{{{i}}})$", loc="right", pad=10, fontsize=18)
		else:
			ax[i].set_title(fr"$2N(E^{{\,}}_{{{i+1}}} - E^{{\,}}_{{{i}}})$", pad=10, fontsize=18)

		# if i==0:

		# 	axin = ax[i].inset_axes([0.25, 0.15, 0.6, 0.5], xlim=(0.784, 0.787), ylim=(4.4, 4.7))
		# 	axin.set_zorder(110)

		# 	for n in range(len(Ns)):
		# 		axin.plot(theta, Ns[n]*diffsNs[n, l, :, i], marker='.', markersize=3, linewidth=0.5, zorder=0, label=fr"$N = {Ns[n]}$")
			
		# 	axin.set_xticks([0.784, np.pi/4, 0.787])
		# 	axin.set_xticklabels(["0.784", r"$\pi/4$", "0.787"])
		# 	axin.set_yticks([4.5, 4.7])

		# 	axin.axvline(np.pi/4, color="k", linewidth=0.5, linestyle="--", zorder=-10)

	ax[3].legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=1, ncol=1, labelspacing=0.8).set_zorder(200)

	ax[3].dataLim.y1 = 3e-5
	ax[3].autoscale_view()

	plt.draw()

	if args.save:
		saveGraph(fig, os.path.join(graphsFolder, f"pbc={PBCs}_arpack={useArpack}_isoLambda", f"lam={np.round(lam[l], 3)}"), f"gaps")