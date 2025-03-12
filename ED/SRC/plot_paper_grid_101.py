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

dataFolder = "../data/grid•101"
graphsFolder = "../graphs/grid•101"
expression = fr"(?:pbc=)(?P<pbc>[0-9e\.-]+)(?:_arpack=)(?P<arpack>[0-9e\.-]+)(?:_N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"
folders = folderNames(dataFolder)

parser = argparse.ArgumentParser()

parser.add_argument("--save", help="save graph.s", action="store_true")
parser.add_argument("--show", help="show graph.s", action="store_true")

args = parser.parse_args()

useArpack = 0
PBCs = 1
epsDegen = 1e-7
nChoose = 18

Ns = []
diffs = []
		
for folder in folders:
	
	_, vals, dataFilename = loadFile(os.path.join(dataFolder, folder), expression)

	if np.unique([int(vals[i]['pbc']) for i in range(len(vals))])==PBCs and np.unique([int(vals[i]['arpack']) for i in range(len(vals))])==useArpack and np.unique([int(vals[i]['N']) for i in range(len(vals))])<nChoose:
		
		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))])
		Ns.append(N[0])

		degen = np.zeros((len(lam), len(theta)))

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

			degen[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = len(np.where(E-E[0]<epsDegen)[0])

		if epsDegen==0.3 and PBCs==1:
	
			fig, ax = plt.subplots(figsize=(8, 6.4))
		
			data = pd.DataFrame(degen[::-1, :], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

			# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
			colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[colors[6],colors[8],colors[0],colors[2],colors[3]], N=np.max(degen))

			axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, vmin=0.5, vmax=np.max(degen)+0.5, rasterized=True)

			cb = fig.colorbar(ax.collections[0], label=r"$n \, : \, E^{\,}_{m} <  E^{\,}_{0} + \Delta \varepsilon, \quad 0 \leq m < n$")
			cb.ax.tick_params(size=0)
			cb.set_ticks(np.arange(1, np.max(degen)+1))
			tick_texts = cb.ax.set_yticklabels([fr"{int(i+1)}" for i in range(int(np.max(degen)))])

			plt.xlabel(r"$\theta$")
			plt.ylabel(r"$\lambda$")

			plt.xticks([0*len(lam), 0.5*len(lam), 1*len(lam)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
			plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$0$", r"$1/2$", r"$1$"])
			
			plt.xticks(rotation=0)
			plt.yticks(rotation=90)

			plt.annotate(r"Neel$^{\,}_{x}$", xy=(0.06, 0.14), xycoords='axes fraction')
			plt.annotate(r"$n = 2\, (2)$", xy=(0.1, 0.05), xycoords='axes fraction')
			plt.annotate(r"Neel$^{\mathrm{SPT}}_{y}$", xy=(0.06, 0.9), xycoords='axes fraction')
			plt.annotate(r"$n = 2\, (8)$", xy=(0.1, 0.81), xycoords='axes fraction')
			plt.annotate(r"Neel$^{\,}_{y}$", xy=(0.66, 0.14), xycoords='axes fraction')
			plt.annotate(r"$n = 2\, (2)$", xy=(0.7, 0.05), xycoords='axes fraction')
			plt.annotate(r"Neel$^{\mathrm{SPT}}_{x}$", xy=(0.66, 0.9), xycoords='axes fraction')
			plt.annotate(r"$n = 2\, (8)$", xy=(0.7, 0.81), xycoords='axes fraction')
			plt.annotate(r"FM$^{\,}_{z}$", xy=(0.4, 0.53), xycoords='axes fraction')
			plt.annotate(r"$n = 2\, (2)$", xy=(0.42, 0.44), xycoords='axes fraction')

			for _, spine in axs.spines.items(): 
			    spine.set_visible(True) 

			plt.title(fr"$\Delta \varepsilon = {epsDegen}$")

			plt.draw()

			if args.save:
				saveGraph(fig, os.path.join(graphsFolder, folder, "degeneracies"), f"eps={epsDegen}")

			if args.show:
				plt.show()

			plt.close()

		elif epsDegen==1e-7 and PBCs==1:

			fig, ax = plt.subplots(figsize=(8, 6.4))
		
			data = pd.DataFrame(degen[::-1, :], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

			# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
			colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[colors[8],colors[0]], N=np.max(degen))

			axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, vmin=0.5, vmax=np.max(degen)+0.5, rasterized=True)

			cb = fig.colorbar(ax.collections[0], label=r"$n \, : \, E^{\,}_{m} <  E^{\,}_{0} + \Delta \varepsilon, \quad 0 \leq m < n$")
			cb.ax.tick_params(size=0)
			cb.set_ticks(np.arange(1, np.max(degen)+1))
			tick_texts = cb.ax.set_yticklabels([fr"{int(i+1)}" for i in range(int(np.max(degen)))])

			plt.xlabel(r"$\theta$")
			plt.ylabel(r"$\lambda$")

			plt.xticks([0*len(lam), 0.5*len(lam), 1*len(lam)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
			plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$0$", r"$1/2$", r"$1$"])
			
			plt.xticks(rotation=0)
			plt.yticks(rotation=90)

			for _, spine in axs.spines.items(): 
			    spine.set_visible(True) 

			plt.title(fr"$\Delta \varepsilon = 10^{{{magnitude(epsDegen)}}}$")

			plt.draw()

			if args.save:
				saveGraph(fig, os.path.join(graphsFolder, folder, "degeneracies"), f"eps=10{magnitude(epsDegen)}")

			if args.show:
				plt.show()

			plt.close()
		
		elif PBCs==0:
	
			fig, ax = plt.subplots(figsize=(8, 6.4))
		
			data = pd.DataFrame(degen[::-1, :], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))

			# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
			colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

			cmap = mpl.colors.LinearSegmentedColormap.from_list("...", colors=[colors[6],colors[8],colors[0],colors[1],colors[2],colors[3],colors[4],colors[5]], N=np.max(degen))

			axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap, vmin=0.5, vmax=np.max(degen)+0.5, rasterized=True)

			cb = fig.colorbar(ax.collections[0], label=r"$n \, : \, E^{\,}_{m} <  E^{\,}_{0} + \Delta \varepsilon, \quad 0 \leq m < n$")
			cb.ax.tick_params(size=0)
			cb.set_ticks(np.arange(1, np.max(degen)+1))
			tick_texts = cb.ax.set_yticklabels([fr"{int(i+1)}" for i in range(int(np.max(degen)))])

			plt.xlabel(r"$\theta$")
			plt.ylabel(r"$\lambda$")

			plt.xticks([0*len(lam), 0.5*len(lam), 1*len(lam)], [r"$0$", r"$\pi/4$", r"$\pi/2$"])
			plt.yticks([1*len(lam), 0.5*len(lam), 0*len(lam)], [r"$0$", r"$1/2$", r"$1$"])
			
			plt.xticks(rotation=0)
			plt.yticks(rotation=90)

			plt.annotate(r"Neel$^{\,}_{x}$", xy=(0.06, 0.14), xycoords='axes fraction')
			plt.annotate(r"$n = 2\, (2)$", xy=(0.1, 0.05), xycoords='axes fraction')
			plt.annotate(r"Neel$^{\mathrm{SPT}}_{y}$", xy=(0.06, 0.9), xycoords='axes fraction')
			plt.annotate(r"$n = 8\, (2)$", xy=(0.1, 0.81), xycoords='axes fraction')
			plt.annotate(r"Neel$^{\,}_{y}$", xy=(0.66, 0.14), xycoords='axes fraction')
			plt.annotate(r"$n = 2\, (2)$", xy=(0.7, 0.05), xycoords='axes fraction')
			plt.annotate(r"Neel$^{\mathrm{SPT}}_{x}$", xy=(0.66, 0.9), xycoords='axes fraction')
			plt.annotate(r"$n = 8\, (2)$", xy=(0.7, 0.81), xycoords='axes fraction')
			plt.annotate(r"FM$^{\,}_{z}$", xy=(0.4, 0.6), xycoords='axes fraction')
			plt.annotate(r"$n = 2\, (2)$", xy=(0.42, 0.51), xycoords='axes fraction')

			for _, spine in axs.spines.items(): 
			    spine.set_visible(True) 

			plt.title(fr"$\Delta \varepsilon = {epsDegen}$")

			plt.draw()

			if args.save:
				saveGraph(fig, os.path.join(graphsFolder, folder, "degeneracies"), f"eps={epsDegen}")

			if args.show:
				plt.show()

			plt.close()
