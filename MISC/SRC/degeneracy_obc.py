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
	'legend.fontsize': 14,
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

graphsFolder = "../GRAPHS"

# # ------------- theta=pi/4, lambda=0.95 --- ED JULIA

# N = np.array([6, 8, 10, 12, 14, 16, 18, 20, 22])
# E = np.array([
# 	[-2.9799584453403103, -2.9799584453403076, -2.979377921495498, -2.979223295207093],
# 	[-4.656326574607155, -4.656326351058425, -4.649127637735762, -4.649127637735734],
# 	[-6.339874925049567, -6.339874513160864, -6.335132706741071, -6.335132706741064],
# 	[-8.028486039383699, -8.028485807667778, -8.02485808643345, -8.024858086433447],
# 	[-9.719502288720156, -9.719502138896651, -9.716526901977131, -9.716526901977131],
# 	[-11.411876624059465, -11.411876560888256, -11.409354541886595, -11.409354541886568],
# 	[-13.105108871768596, -13.105108843472552, -13.102917520335401, -13.102917520335394],
# 	[-14.79891690760494, -14.798916893620264, -14.796977541414917, -14.796977541414856],
# 	[-16.493129970079764, -16.49312996263627, -16.491389474011655, -16.491389474011577]
# 	])

# N = N[:]
# E = E[:, :]

# E1E0 = E[:, 1]-E[:, 0]

# def f(L, a, b):
# 	return a*1/L + b

# fig, ax = plt.subplots(figsize=(8, 4))

# popt, pcov = curve_fit(f, N, E1E0)
# perr = np.sqrt(np.diag(pcov))

# print(popt, perr)

# p, = ax.plot(1/N, E1E0, marker='+', linewidth=0, color=colors[0], zorder=100, markersize=10, markeredgewidth=2)
# p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

# ax.plot(1/N, f(N, *popt), color=colors[6], linestyle='--', zorder=99, label=fr"$\frac{{{popt[0]:.2f} \pm {perr[0]:.2f}}}{{N}} + ({popt[1]:.3f} \pm {perr[1]:.3f})$")

# plt.xlabel(r"$1/(2N)$")
# plt.ylabel(r"$E^{\,}_{1}-E^{\,}_{0}$")

# ax.minorticks_on()
# # ax.grid(which='minor', linewidth=0.2)
# # ax.grid(which='major', linewidth=0.6)
# ax.legend(loc="lower center", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, bbox_to_anchor=(0.5, 1.01), borderaxespad=0, ncol=3).set_zorder(101)

# plt.draw()

# saveGraph(fig, os.path.join(graphsFolder, "ED"), "gap_julia")



N = np.array([6, 8, 10, 12, 14, 16, 18, 20, 22])
E = np.array([
	[-2.9799584453403103, -2.9799584453403076, -2.979377921495498, -2.979223295207093],
	[-4.656326574607155, -4.656326351058425, -4.649127637735762, -4.649127637735734],
	[-6.339874925049567, -6.339874513160864, -6.335132706741071, -6.335132706741064],
	[-8.028486039383699, -8.028485807667778, -8.02485808643345, -8.024858086433447],
	[-9.719502288720156, -9.719502138896651, -9.716526901977131, -9.716526901977131],
	[-11.411876624059465, -11.411876560888256, -11.409354541886595, -11.409354541886568],
	[-13.105108871768596, -13.105108843472552, -13.102917520335401, -13.102917520335394],
	[-14.79891690760494, -14.798916893620264, -14.796977541414917, -14.796977541414856],
	[-16.493129970079764, -16.49312996263627, -16.491389474011655, -16.491389474011577]
	])

N = N[4:]
E = E[4:, :]

E1E0 = E[:, 1]-E[:, 0]

fig, ax = plt.subplots(figsize=(6, 3))

def g(L, a, b):
	return a*L + b

popt, pcov = curve_fit(g, N, np.log(E1E0))
perr = np.sqrt(np.diag(pcov))
r2_ = r2(np.log(E1E0), g(N, *popt))

print(popt, perr)

p, = ax.plot(N, np.log(E1E0), marker='+', linewidth=0, color=colors[0], zorder=100, markersize=10, markeredgewidth=2)
p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

ax.plot(N, g(N, *popt), color=colors[6], linestyle='--', zorder=99, label=fr"$({popt[0]:.2f} \pm {perr[0]:.2f}) \cdot 2N - ({np.abs(popt[1]):.1f} \pm {perr[1]:.1f})$")

ax.annotate(fr"$R^2={r2_:.5f}$", xy=(0.73, 0.71), xycoords="axes fraction", fontsize=14)

plt.xlabel(r"$2N$")
plt.ylabel(r"$\ln \,(E^{\,}_{1}-E^{\,}_{0})$")

ax.minorticks_on()

ax.legend(loc="upper right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=3).set_zorder(101)

plt.draw()

saveGraph(fig, os.path.join(graphsFolder, "ED"), "gap_julia_log_lin")



N = np.array([6, 8, 10, 12, 14, 16, 18, 20, 22])
E = np.array([
	[-2.9799584453403103, -2.9799584453403076, -2.979377921495498, -2.979223295207093],
	[-4.656326574607155, -4.656326351058425, -4.649127637735762, -4.649127637735734],
	[-6.339874925049567, -6.339874513160864, -6.335132706741071, -6.335132706741064],
	[-8.028486039383699, -8.028485807667778, -8.02485808643345, -8.024858086433447],
	[-9.719502288720156, -9.719502138896651, -9.716526901977131, -9.716526901977131],
	[-11.411876624059465, -11.411876560888256, -11.409354541886595, -11.409354541886568],
	[-13.105108871768596, -13.105108843472552, -13.102917520335401, -13.102917520335394],
	[-14.79891690760494, -14.798916893620264, -14.796977541414917, -14.796977541414856],
	[-16.493129970079764, -16.49312996263627, -16.491389474011655, -16.491389474011577]
	])

N = N[4:]
E = E[4:, :]

E1E0 = E[:, 1]-E[:, 0]

fig, ax = plt.subplots(figsize=(6, 3))

popt, pcov = curve_fit(g, np.log(N), np.log(E1E0))
perr = np.sqrt(np.diag(pcov))
r2_ = r2(np.log(E1E0), g(np.log(N), *popt))

print(popt, perr)

p, = ax.plot(np.log(N), np.log(E1E0), marker='+', linewidth=0, color=colors[0], zorder=100, markersize=10, markeredgewidth=2)
p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

ax.plot(np.log(N), g(np.log(N), *popt), color=colors[6], linestyle='--', zorder=99, label=fr"$({popt[0]:.2f} \pm {perr[0]:.2f})\ln\,2N + ({popt[1]:.2f} \pm {perr[1]:.2f})$")

ax.annotate(fr"$R^2={r2_:.5f}$", xy=(0.73, 0.71), xycoords="axes fraction", fontsize=14)

plt.xlabel(r"$\ln\,2N$")
plt.ylabel(r"$\ln\,(E^{\,}_{1}-E^{\,}_{0})$")

ax.minorticks_on()

ax.legend(loc="upper right", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, ncol=3).set_zorder(101)

plt.draw()

saveGraph(fig, os.path.join(graphsFolder, "ED"), "gap_julia_log_log")

# # ------------- theta=pi/4, lambda=0.95 --- ED PYTHON

# N = np.array([6, 8, 10, 12, 14])
# E = np.array([
# 	[-2.9799584453403054, -2.9799584453403045, -2.979377921495495, -2.9792232952070896],
# 	[-4.656326574607152,  -4.6563263510584205, -4.649127637735741, -4.649127637735741],
# 	[-6.339874925049574, -6.339874513160874, -6.335132706741073, -6.335132706741073],
# 	[-8.028486039383708, -8.02848580766779, -8.024858086433454, -8.024858086433449],
# 	[-9.719502288720156, -9.719502138896638, -9.716526901977147, -9.716526901977142]
# 	])

# N = N[:]
# E = E[:, :]

# E1E0 = E[:, 1]-E[:, 0]

# def f(L, a, b):
# 	return a*1/L + b

# fig, ax = plt.subplots(figsize=(8, 4))

# popt, pcov = curve_fit(f, N, E1E0)
# perr = np.sqrt(np.diag(pcov))

# print(popt, perr)

# p, = ax.plot(1/N, E1E0, marker='+', linewidth=0, color=colors[0], zorder=100, markersize=10, markeredgewidth=2)
# p.set_marker(mpl.markers.MarkerStyle(p.get_marker(), capstyle="round"))

# ax.plot(1/N, f(N, *popt), color=colors[6], linestyle='--', zorder=99, label=fr"$\frac{{{popt[0]:.2f} \pm {perr[0]:.2f}}}{{N}} + ({popt[1]:.3f} \pm {perr[1]:.3f})$")

# plt.xlabel(r"$1/(2N)$")
# plt.ylabel(r"$E^{\,}_{1}-E^{\,}_{0}$")

# ax.minorticks_on()
# # ax.grid(which='minor', linewidth=0.2)
# # ax.grid(which='major', linewidth=0.6)
# ax.legend(loc="lower center", labelspacing=0.5, columnspacing=0.8, handletextpad=0.6, handlelength=1.6, bbox_to_anchor=(0.5, 1.01), borderaxespad=0, ncol=3).set_zorder(101)

# plt.draw()

# saveGraph(fig, os.path.join(graphsFolder, "ED"), "gap_python")
