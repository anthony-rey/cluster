import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler
from scipy.optimize import curve_fit
import numpy as np
from __func__ import *

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

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

graphsFolder = "../graphs"

# ------------- theta = pi/4 --- ED

x = np.array([4, 6, 8, 10, 12, 14, 16, 18, 20])
y = np.array([0.146, 0.168, 0.197, 0.213, 0.219, 0.223, 0.226, 0.228, 0.229])

# x = x[3:]
# y = y[3:]

def f(N, a, b):
	return a*1/N + b

fig, ax = plt.subplots(figsize=(4.5, 6))

popt, pcov = curve_fit(f, x, y)
perr = np.sqrt(np.diag(pcov))

print(popt, perr)

p, = ax.plot(1/x, y, marker='+', linewidth=0, color=colors[0], zorder=100, markersize=10, markeredgewidth=2)
p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

ax.plot(1/x, f(x, *popt), color=colors[6], linestyle='--', zorder=99, label=fr"$\frac{{{popt[0]:.2f} \pm {perr[0]:.2f}}}{{N}} + ({popt[1]:.3f} \pm {perr[1]:.3f})$")

plt.xlabel(r"$1/(2N)$")
plt.ylabel(r"$\lambda^{\,}_{*}$")

ax.minorticks_on()
# ax.grid(which='minor', linewidth=0.2)
# ax.grid(which='major', linewidth=0.6)
ax.legend(loc="lower center", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, bbox_to_anchor=(0.5, 1.01), borderaxespad=0, ncol=3).set_zorder(101)

plt.draw()

saveGraph(fig, os.path.join(graphsFolder, "ed"), "tricriticalU(1)")

# ------------- N = 128, chi = 128, OBCs

x = np.array([0.782, 0.775, 0.765, 0.755, 0.735, 0.725,
			0.715, 0.71, 0.69, 0.675, 0.655,
			0.64, 0.628, 0.615, 0.59, 0.57,
			0.55, 0.54, 0.51, 0.49, 0.465,
			0.45, 0.415, 0.385, 0.34, 0.3])
y = np.array([0.25, 0.26, 0.27, 0.28, 0.29, 0.3,
			0.31, 0.32, 0.33, 0.34, 0.35,
			0.36, 0.37, 0.38, 0.39, 0.4,
			0.41, 0.42, 0.43, 0.44, 0.45,
			0.46, 0.47, 0.48, 0.49, 0.5])

dx = np.array([0.002, 0.005, 0.005, 0.02, 0.02, 0.02,
			0.02, 0.02, 0.02, 0.02, 0.02,
			0.02, 0.02, 0.02, 0.02, 0.02,
			0.02, 0.02, 0.02, 0.02, 0.02,
			0.04, 0.02, 0.02, 0.02, 0.04])

fig, ax = plt.subplots(figsize=(8, 7.5))

plots, caps, errs = ax.errorbar(x, y, xerr=dx, color=colors[6], ecolor=colors[0], zorder=100, elinewidth=3, linewidth=3, barsabove=True)

for err in errs:
	err.set_capstyle("round")

ax.set_xticks([0.3, 0.4, 0.5, 0.6, 0.7, np.pi/4])
ax.set_xticklabels(["0.3", "0.4", "0.5", "0.6", "0.7", r"$\pi/4$"])

ax.minorticks_on()
# ax.grid(which='minor', linewidth=0.2, alpha=0.33)
# ax.grid(which='major', linewidth=0.6)

plt.annotate(r"Neel$^{\,}_{x}$", xy=(0.16, 0.24), xycoords='axes fraction')
plt.annotate(r"FM$^{\,}_{z}$", xy=(0.76, 0.84), xycoords='axes fraction')

plt.xlabel(r"$\theta$")
plt.ylabel(r"$\lambda$")

plt.draw()

saveGraph(fig, os.path.join(graphsFolder, "dmrg"), "transitionsDiamond_LL")

plt.close()

fig, ax = plt.subplots(figsize=(8, 7.5))

plots, caps, errs = ax.errorbar(x, y, xerr=dx, color=colors[7], ecolor=colors[1], elinewidth=2, linewidth=2, zorder=100, barsabove=False)
for err in errs:
	err.set_capstyle("round")
plots, caps, errs = ax.errorbar(np.pi/2-x, y, xerr=dx, color=colors[7], ecolor=colors[1], elinewidth=2, linewidth=2, zorder=100, barsabove=False)
for err in errs:
	err.set_capstyle("round")
plots, caps, errs = ax.errorbar(x, 1-y, xerr=dx, color=colors[7], ecolor=colors[1], elinewidth=2, linewidth=2, zorder=100, barsabove=False)
for err in errs:
	err.set_capstyle("round")
plots, caps, errs = ax.errorbar(np.pi/2-x, 1-y, xerr=dx, color=colors[7], ecolor=colors[1], elinewidth=2, linewidth=2, zorder=100, barsabove=False)
for err in errs:
	err.set_capstyle("round")

ax.minorticks_on()
# ax.grid(which='minor', linewidth=0.2, alpha=0.33)
# ax.grid(which='major', linewidth=0.6)

# for spine in ax.spines.items():
# 	spine[1].set_linewidth(1)

plt.annotate(r"Neel$^{\,}_{x}$", xy=(0.16, 0.24), xycoords='axes fraction')
plt.annotate(r"Neel$^{\,}_{y}$", xy=(0.71, 0.24), xycoords='axes fraction')
plt.annotate(r"Neel$^{\mathrm{SPT}}_{x}$", xy=(0.71, 0.74), xycoords='axes fraction')
plt.annotate(r"Neel$^{\mathrm{SPT}}_{y}$", xy=(0.16, 0.74), xycoords='axes fraction')
plt.annotate(r"FM$^{\,}_{z}$", xy=(0.465, 0.49), bbox={'fc': 'w', 'boxstyle': 'round', 'ec': 'w'}, xycoords='axes fraction')

ax.axvline(np.pi/4, linestyle="--", color="k", linewidth=1)
ax.axhline(0.5, linestyle="--", color="k", linewidth=1)

ax.set_xticks([0, 0.2, 0.4, 0.6, np.pi/4, 1, 1.2, 1.4, np.pi/2])
ax.set_xticklabels(["0", "0.2", "0.4", "0.6", r"$\pi/4$", "1", "1.2", "1.4", r"$\pi/2$"])
ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])

ax.set_xlim(0, np.pi/2)
ax.set_ylim(0, 1)

plt.xlabel(r"$\theta$")
plt.ylabel(r"$\lambda$")

plt.draw()

saveGraph(fig, os.path.join(graphsFolder, "dmrg"), "transitionsDiamond")

plt.close()