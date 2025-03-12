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
	'font.size': 14,
	'legend.fontsize': 10,
	'legend.fancybox': True,
	'savefig.dpi': 250,
	'savefig.bbox': 'tight',
	'savefig.format': 'pdf',
	})

colors = [[0.8, 0, 0.5],[0, 0.5, 0.8],[0.8, 0.5, 0]]

# mpl.rc('text.latex', preamble=r"\usepackage{mathpazo}")
# mpl.rc('text.latex', preamble=r"\usepackage{eulervm}")

dataFolder = "../data/scaling/"
graphsFolder = "../graphs"
expression = fr"(?:pbc=)(?P<pbc>[0-9e\.-]+)(?:_arpack=)(?P<arpack>[0-9e\.-]+)(?:_N=)(?P<N>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.dat)"

parser = argparse.ArgumentParser()

parser.add_argument("--energies", action="store_true")
parser.add_argument("--gaps", action="store_true")
parser.add_argument("--degeneracies", action="store_true")
parser.add_argument("--lines", action="store_true")
parser.add_argument("--scaling", action="store_true")
parser.add_argument("--save", help="save graph.s", action="store_true")
parser.add_argument("--show", help="show graph.s", action="store_true")

args = parser.parse_args()
			
nE = 100
PBCs = 1
useArpack = 0
epsDegen = 0.3
saveMax = 100
nChoose = 10

Ns = []
diffs = []
		
folders = folderNames(dataFolder)

# folders = ["scaling•EU(1)/"]
for folder in folders:
	
	print(os.path.join(dataFolder, folder))
	_, vals, dataFilename = loadFile(os.path.join(dataFolder, folder), expression)

	# if np.unique([int(vals[i]['pbc']) for i in range(len(vals))])==PBCs and np.unique([int(vals[i]['arpack']) for i in range(len(vals))])==useArpack and np.unique([int(vals[i]['N']) for i in range(len(vals))])<nChoose:
	# if np.unique([int(vals[i]['pbc']) for i in range(len(vals))])==PBCs and np.unique([int(vals[i]['arpack']) for i in range(len(vals))])==useArpack and np.unique([int(vals[i]['N']) for i in range(len(vals))])==nChoose:
	if True:
		
		lam = np.sort(np.unique([vals[i]['lam'] for i in range(len(vals))]))
		theta = np.sort(np.unique([vals[i]['theta'] for i in range(len(vals))]))

		# 
		# lam = lam[lam==0.99]
		# theta = theta[theta<0.7859]
		# 

		N = np.unique([int(vals[i]['N']) for i in range(len(vals))])
		Ns.append(N[0])

		diff = np.zeros((len(lam), len(theta), saveMax-1))
		degen = np.zeros((len(lam), len(theta)))

		for i in range(len(dataFilename)):

			with open(os.path.join(dataFolder, folder, dataFilename[i]), 'rb') as file:
				engine = pickle.load(file)
		
			# ****************
			print(dataFilename[i])
			# ****************

			E = np.array(engine.E)

			if useArpack:
				E = -E[::-1]

			gaps = E[1:] - E[:-1]
			print((E[:60]-E[0])*8/(4.47979))

			# if args.localDegen:
			# 	if vals[i]['lam']>0.2 and vals[i]['lam']<0.5 and vals[i]['theta']>0.2 and vals[i]['theta']<0.8:
			# 		degen[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = len(np.where(E-E[0]<epsDegen)[0])
				
			degen[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta)] = len(np.where(E-E[0]<epsDegen)[0])
			diff[np.where(vals[i]['lam']==lam), np.where(vals[i]['theta']==theta), :] = gaps[:saveMax-1]
				
			if args.gaps:

				fig, ax = plt.subplots(figsize=(5, 5))

				ax.scatter(range(1, nE), gaps[:nE-1], marker='x', zorder=100)
				
				ax.minorticks_on()
				ax.grid(which='minor', linewidth=0.2)
				ax.grid(which='major', linewidth=0.6)
				# ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

				plt.xlabel("$i$")
				plt.ylabel(r"$E_i-E_{i-1}$")

				plt.draw()

				if args.save:
					saveGraph(fig, os.path.join(graphsFolder, folder, "gaps"), dataFilename[i])

				if args.show:
					plt.show()

				plt.close()

			if args.energies:

				fig, ax = plt.subplots(figsize=(5, 5))

				ax.scatter(range(len(E[:nE])), E[:nE], marker='x', color=colors[0], zorder=100)
				
				ax.minorticks_on()
				ax.grid(which='minor', linewidth=0.2)
				ax.grid(which='major', linewidth=0.6)
				# ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

				plt.xlabel("$i$")
				plt.ylabel("$E_i$")

				plt.draw()

				if args.save:
					saveGraph(fig, os.path.join(graphsFolder, folder, "energies"), dataFilename[i])

				if args.show:
					plt.show()

				plt.close()
		
		if args.degeneracies:

			from scipy.ndimage.filters import gaussian_filter
			
			# print(N)
			data = pd.DataFrame(degen[::-1, :], index=np.round(lam, 4)[::-1], columns=np.round(theta, 4))
			# df3_smooth = gaussian_filter(data, sigma=2)
		
			fig, ax = plt.subplots(figsize=(5, 4))

			# axs = sns.heatmap(data, square=True, cbar=False, cmap=cmap)
			axs = sns.heatmap(data, square=True, cbar=False, cmap=mpl.cm.get_cmap('pink_r', np.max(degen)), vmin=0.5, vmax=np.max(degen)+0.5)
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
			    spine.set_linewidth(0.8) 

			# plt.title(fr"$\varepsilon_{{\mathrm{{degeneracy}}}} = 10^{{{magnitude(epsDegen)}}}$")
			plt.title(fr"$\Delta \varepsilon = {epsDegen}$")

			plt.draw()

			if args.save:
				# saveGraph(fig, os.path.join(graphsFolder, folder, "degeneracies"), f"eps=10{magnitude(epsDegen)}")
				saveGraph(fig, os.path.join(graphsFolder, folder, "degeneracies"), f"eps={epsDegen}")

			if args.show:
				plt.show()

			plt.close()

		diffs.append(diff)

if args.lines or args.scaling:

	ind = np.argsort(Ns)

	Ns = np.array(Ns)[ind]
	diffs = np.array(diffs)[ind, :, :, :]

if args.lines:

	if args.save:
		fig, ax = plt.subplots(figsize=(5, 5))

	# plot iso-lambda lines

	for l in range(len(lam)):
		for i in range(len(diffs[0, 0, 0, :])):

			if args.show:
				fig, ax = plt.subplots(figsize=(5, 5))

			for n in range(len(Ns)):
				ax.plot(theta, Ns[n]*diffs[n, l, :, i], marker='.', zorder=100, label=fr"$N = {Ns[n]}$")

			ax.minorticks_on()
			ax.grid(which='minor', linewidth=0.2)
			ax.grid(which='major', linewidth=0.6)
			ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0, ncol=3).set_zorder(101)

			plt.xlabel(r"$\theta$")
			plt.ylabel(fr"$N(E_{{{i+1}}} - E_{{{i}}})$")

			# plt.title(fr"$\lambda = {np.round(lam[l], 3)}$")

			plt.draw()

			if args.save:
				saveGraph(fig, os.path.join(graphsFolder, f"pbc={PBCs}_arpack={useArpack}_isoLambda", f"lam={np.round(lam[l], 3)}"), f"gap_{i+1}-{i}")
				ax.cla()
		
			if args.show:
				plt.show()
				plt.close() 

	for i in range(len(diffs[0, 0, 0, :])):
		for l in range(len(lam)):

			if args.show:
				fig, ax = plt.subplots(figsize=(5, 5))

			for n in range(len(Ns)):
				ax.plot(theta, Ns[n]*diffs[n, l, :, i], marker='.', zorder=100, label=fr"$N = {Ns[n]}$")

			ax.minorticks_on()
			ax.grid(which='minor', linewidth=0.2)
			ax.grid(which='major', linewidth=0.6)
			ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0, ncol=3).set_zorder(101)

			plt.xlabel(r"$\theta$")
			plt.ylabel(fr"$N(E_{{{i+1}}} - E_{{{i}}})$")

			# plt.title(fr"$\theta = {np.round(theta[t], 3)}$")

			plt.draw()

			if args.save:
				saveGraph(fig, os.path.join(graphsFolder, f"pbc={PBCs}_arpack={useArpack}_isoLambda", f"gap_{i+1}-{i}"), f"lam={np.round(lam[l], 3)}")
				ax.cla()

			if args.show:
				plt.show()
				plt.close() 	

	# plot iso-theta lines

	for t in range(len(theta)):
		for i in range(len(diffs[0, 0, 0, :])):

			if args.show:
				fig, ax = plt.subplots(figsize=(5, 5))

			for n in range(len(Ns)):
				ax.plot(lam, Ns[n]*diffs[n, :, t, i], marker='.', zorder=100, label=fr"$N = {Ns[n]}$")

			ax.minorticks_on()
			ax.grid(which='minor', linewidth=0.2)
			ax.grid(which='major', linewidth=0.6)
			ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0, ncol=3).set_zorder(101)

			plt.xlabel(r"$\lambda$")
			plt.ylabel(fr"$N(E_{{{i+1}}} - E_{{{i}}})$")

			# plt.title(fr"$\theta = {np.round(theta[t], 3)}$")

			plt.draw()

			if args.save:
				saveGraph(fig, os.path.join(graphsFolder, f"pbc={PBCs}_arpack={useArpack}_isoTheta", f"theta={np.round(theta[t], 3)}"), f"gap_{i+1}-{i}")
				ax.cla()

			if args.show:
				plt.show()
				plt.close() 

	for i in range(len(diffs[0, 0, 0, :])):
		for t in range(len(theta)):

			if args.show:
				fig, ax = plt.subplots(figsize=(5, 5))

			for n in range(len(Ns)):
				ax.plot(lam, Ns[n]*diffs[n, :, t, i], marker='.', zorder=100, label=fr"$N = {Ns[n]}$")

			ax.minorticks_on()
			ax.grid(which='minor', linewidth=0.2)
			ax.grid(which='major', linewidth=0.6)
			ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0, ncol=3).set_zorder(101)

			plt.xlabel(r"$\lambda$")
			plt.ylabel(fr"$N(E_{{{i+1}}} - E_{{{i}}})$")

			# plt.title(fr"$\theta = {np.round(theta[t], 3)}$")

			plt.draw()

			if args.save:
				saveGraph(fig, os.path.join(graphsFolder, f"pbc={PBCs}_arpack={useArpack}_isoTheta", f"gap_{i+1}-{i}"), f"theta={np.round(theta[t], 3)}")
				ax.cla()

			if args.show:
				plt.show()
				plt.close() 

if args.scaling:

	ind = np.where(Ns%2==0)[0]
	# ind = ind[np.where(Ns>6)[0]]

	r2s = np.zeros((len(lam), len(theta), len(diffs[0, 0, 0, :]))) 

	if args.save:
		fig, ax = plt.subplots(figsize=(5, 5))

	for l in range(len(lam)):
		for t in range(len(theta)):

			# if l in range(50, 51):
				# if t in range(0, 1):
				# if t == l:

					for i in range(len(diffs[0, 0, 0, :])):
					# for i in range(1):
					
						if args.show:
							fig, ax = plt.subplots(figsize=(5, 5))
				
						def f(N, a, b):
							return a*1/N + b
							# return np.exp(-a*N) + b

						popt, pcov = curve_fit(f, Ns[ind], diffs[ind, l, t, i])
						perr = np.sqrt(np.diag(pcov))

						r2_ = r2(diffs[ind, l, t, i], f(Ns[ind], *popt))
						r2s[l, t, i] = r2_

						# print(lam[l], theta[t], popt, perr)
						ax.plot(1/Ns[ind], f(Ns[ind], *popt), color=colors[1], linestyle='--', zorder=99, label=fr"$\frac{{{popt[0]:.3f} \pm {perr[0]:.3f}}}{{N}} + ({popt[1]:.3f} \pm {perr[1]:.3f}),\ R^2 = {r2_:1.6f}$")

						ax.scatter(1/Ns[ind], diffs[ind, l, t, i], marker='x', color=colors[0], zorder=100)

						ax.minorticks_on()
						ax.grid(which='minor', linewidth=0.2)
						ax.grid(which='major', linewidth=0.6)
						ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0).set_zorder(101)

						plt.xlabel(r"$\frac{1}{N}$")
						plt.ylabel(fr"$E_{{{i+1}}} - E_{{{i}}}$")

						plt.draw()

						if args.save:
							saveGraph(fig, os.path.join(graphsFolder, f"pbc={PBCs}_arpack={useArpack}_scaling", f"gap_{i+1}-{i}"), f"lam={np.round(lam[l], 3)}_theta={np.round(theta[t], 3)}")
							# here versy important to release memory correclty since we will plot about 100 000 figures
							ax.cla()

						if args.show:
							plt.show()
							plt.close() 

	# if args.save:
	# 	fig, ax = plt.subplots(figsize=(5, 5))

	# for l in range(len(lam)):

	# 	if args.show:
	# 		fig, ax = plt.subplots(figsize=(5, 5))

	# 	for i in range(len(diffs[0, 0, 0, :])):
	# 		ax.plot(theta, r2s[l, :, i], marker='.', zorder=100, label=fr"gap ${i+1}-{i}$")

	# 	ax.minorticks_on()
	# 	ax.grid(which='minor', linewidth=0.2)
	# 	ax.grid(which='major', linewidth=0.6)
	# 	ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0, ncol=3).set_zorder(101)

	# 	plt.xlabel(r"$\theta$")
	# 	plt.ylabel(fr"$R^2$")

	# 	ax.set_ylim(0, 1)

	# 	# plt.title(fr"$\lambda = {np.round(lam[l], 3)}$")

	# 	plt.draw()

	# 	if args.save:
	# 		saveGraph(fig, os.path.join(graphsFolder, f"pbc={PBCs}_arpack={useArpack}_scaling", f"r2_isoLambda"), f"lam={np.round(lam[l], 3)}")
	# 		ax.cla()
	
	# 	if args.show:
	# 		plt.show()
	# 		plt.close() 

	# for t in range(len(theta)):

	# 	if args.show:
	# 		fig, ax = plt.subplots(figsize=(5, 5))

	# 	for i in range(len(diffs[0, 0, 0, :])):
	# 		ax.plot(lam, r2s[:, t, i], marker='.', zorder=100, label=fr"gap ${i+1}-{i}$")

	# 	ax.minorticks_on()
	# 	ax.grid(which='minor', linewidth=0.2)
	# 	ax.grid(which='major', linewidth=0.6)
	# 	ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0, ncol=3).set_zorder(101)

	# 	plt.xlabel(r"$\lambda$")
	# 	plt.ylabel(fr"$R^2$")

	# 	ax.set_ylim(0, 1)

	# 	plt.draw()

	# 	if args.save:
	# 		saveGraph(fig, os.path.join(graphsFolder, f"pbc={PBCs}_arpack={useArpack}_scaling", f"r2_isoTheta"), f"theta={np.round(theta[t], 3)}")
	# 		ax.cla()
	
	# 	if args.show:
	# 		plt.show()
	# 		plt.close() 


