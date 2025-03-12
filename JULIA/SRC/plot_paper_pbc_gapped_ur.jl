# ENV["MPLBACKEND"]="qt5agg"

using FileIO
using PyCall
using PyPlot

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.rc("font", size=18)
PyPlot.rc("axes", titlesize=18)
PyPlot.rc("legend", fontsize=18)
PyPlot.rc("legend", fancybox=true)
PyPlot.rc("savefig", dpi=250)
PyPlot.rc("savefig", bbox="tight")
PyPlot.rc("savefig", format="png")
PyPlot.rc("axes", linewidth=0.8)
PyPlot.rc("legend", edgecolor="0.8")
PyPlot.rc("patch", linewidth=0.8)
PyPlot.rc("xtick.major", size=6)
PyPlot.rc("xtick.major", width=0.8)
PyPlot.rc("xtick.minor", size=3)
PyPlot.rc("xtick.minor", width=0.8)
PyPlot.rc("ytick.major", size=6)
PyPlot.rc("ytick.major", width=0.8)
PyPlot.rc("ytick.minor", size=3)
PyPlot.rc("ytick.minor", width=0.8)

py"""
def createFolderIfMissing(path):
	import os

	if not os.path.exists(path):
		os.makedirs(path)
"""

py"""
def loadFile(folder, loadEngines=False):
	import os
	import re
	
	expression = r"(?:N=)(?P<N>[0-9e\.-]+)(?:_chi=)(?P<chi>[0-9e\.-]+)(?:_lam=)(?P<lam>[0-9e\.-]+)(?:_theta=)(?P<theta>[0-9e\.-]+)(?:\.jld2)"

	engines = []
	vals = []
	names = []

	for file in sorted(os.scandir(folder), key=lambda e: e.name):

		try:
			m = re.match(expression, file.name)
			d = m.groupdict()
		except:
			pass
		else:
			for key in d.keys():
				d[key] = float(d[key])

			vals.append(d)
			names.append(file.name)

			if loadEngines:
			
				with open(os.path.join(folder, file.name), 'rb') as file:
					engines.append(pickle.load(file))

	return engines, vals, names
"""

py"""
def folderNames(folder):
	import os

	return sorted([name+"/" for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])
"""

numpy = pyimport("numpy")
matplotlib = pyimport("matplotlib")
pandas = pyimport("pandas")
scipy = pyimport("scipy")
cycler = pyimport("cycler")
seaborn = pyimport("seaborn")

PBCs = true

if PBCs
	graphFolder = "../GRAPHS/PBC/GAPPED_UR/"
	dataFolder = "../DATA/PBC/GAPPED_UR/"
else
	graphFolder = "../GRAPHS/OBC/GAPPED_UR/"
	dataFolder = "../DATA/OBC/GAPPED_UR/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_spins = false
plot_scaling_en = true

nE = 4

savePlot = true

Ns = []
diffsNs = []

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))
	push!(Ns, numpy.unique([Int(vals[i]["N"]) for i=1:length(vals)])[1])

	Es = numpy.zeros((length(lam), length(theta), nE))
	diffs = numpy.zeros((length(lam), length(theta), nE-1))

	for i=1:length(dataFilenames)

		N = Int(vals[i]["N"])
		chi = Int(vals[i]["chi"])

		data = 0
		data = load(dataFolder * folder * computeFolder * dataFilenames[i])
		
		k = length(data["energies"])

		print(k)

		println("··· ", dataFilenames[i])

		# ------------------------------
		if true
		# ------------------------------

			if plot_scaling_en

				E = numpy.sort(data["energies"])
	
				Es[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = numpy.sort([E[e] for e=1:nE])
				diffs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = [E[e+1]-E[e] for e=1:nE-1]
			end


			if plot_spins

				for j=1:k

					fig, ax = plt.subplots(figsize=(4, 4))

					mX = data["mags"][j, 1, :]
					mY = data["mags"][j, 2, :]
					mZ = data["mags"][j, 3, :]

					ax.plot(range(1, length(mX)), mX, color=colors[1], label=latexstring(L"$\langle \widehat{X}^{\,}_{j} \rangle$"), zorder=100)
					ax.plot(range(1, length(mY)), mY, color=colors[4], label=latexstring(L"$\langle \widehat{Y}^{\,}_{j} \rangle$"), zorder=100)
					ax.plot(range(1, length(mZ)), mZ, color=colors[7], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j} \rangle$"), zorder=100)
					
					ax.minorticks_on()
					# ax.grid(which="minor", linewidth=0.2)
					# ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper left", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(101)

					plt.xlabel(latexstring(L"$j$"))

					plt.draw()
			
					if savePlot
						py"createFolderIfMissing"(graphFolder * folder * "spins/")
						fig.savefig(graphFolder * folder * "spins/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".pdf")
						ax.cla()
					end

					plt.close()

					fig, ax = plt.subplots(figsize=(4, 4))

					mXSPT = data["opSPT"][j, 1, :]
					mYSPT = data["opSPT"][j, 2, :]

					ax.plot(range(1, length(mXS)+1), mXSPT, color=colors[1], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \widehat{X}^{\,}_{j} \widehat{Z}^{\,}_{j+1} \rangle$"), zorder=100)
					ax.plot(range(1, length(mYS)+1), mYSPT, color=colors[4], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \widehat{Y}^{\,}_{j} \widehat{Z}^{\,}_{j+1} \rangle$"), zorder=100)
					
					ax.minorticks_on()
					# ax.grid(which="minor", linewidth=0.2)
					# ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper left", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(101)

					plt.xlabel(latexstring(L"$j$"))

					plt.draw()
			
					if savePlot
						py"createFolderIfMissing"(graphFolder * folder * "spinsSPT/")
						fig.savefig(graphFolder * folder * "spinsSPT/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".pdf")
						ax.cla()
					end

					plt.close()
				end
			end
		end
	end

	if plot_scaling_en
		push!(diffsNs, diffs)
	end

end

indSort = numpy.argsort(Ns).+1

Ns = numpy.array(Ns)[indSort]
diffsNs = numpy.array(diffsNs)[indSort, :, :, :]

slopes = numpy.zeros((length(lam), length(theta), nE-1))
errs = numpy.zeros((length(lam), length(theta), nE-1))

fig_gaps, ax_gaps = plt.subplots(1, 2, figsize=(12, 3.5))
plt.subplots_adjust(wspace=0.3)

fig_spins_corner, ax_spins_corner = plt.subplots(2, 2, figsize=(12, 4))
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.2)
fig_spins_phase, ax_spins_phase = plt.subplots(2, 2, figsize=(12, 4))
plt.subplots_adjust(wspace=0.3)
plt.subplots_adjust(hspace=0.2)

if plot_scaling_en

	indSort = numpy.argsort(Ns).+1

	Ns = numpy.array(Ns)[indSort]
	diffsNs = numpy.array(diffsNs)[indSort, :, :, :]

	maxs = []
	global os = []
	s = 0
	for t=length(theta):-1:1
		l = t

		linestyles = ["-", ":", "--"]
		cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[1],colors[4],colors[6]], N=256)
		PyPlot.rc("axes", prop_cycle=cycler.cycler(marker=[".", "+", "x"], markersize=2 .*[10, 10, 7.5], markeredgewidth=2 .*[1, 1.5, 1.5], color=cmap(numpy.linspace(0, 1, nE-1))))

		for k=1:nE-1

			py"""
			def f(N, a, b):
				import numpy 

				return a/N + b
			"""

			py"""
			def r2(y, fit):
				import numpy

				ss_res = numpy.sum((y-fit)**2)
				ss_tot = numpy.sum((y-numpy.mean(y))**2)

				return (1 - ss_res/ss_tot)
			"""

			popt, pcov = scipy.optimize.curve_fit(py"f", Ns, diffsNs[:, l, t, k])
			perr = numpy.sqrt(numpy.diag(pcov))

			slopes[lam.==lam[l], theta.==theta[t], k] .= popt[1]
			errs[lam.==lam[l], theta.==theta[t], k] .= perr[1]
		end
			
		global ps = []
		global s += 1

		for k=1:nE-1

			py"""
			def f(N, a, b):
				import numpy 

				return a/N + b
			"""

			py"""
			def r2(y, fit):
				import numpy

				ss_res = numpy.sum((y-fit)**2)
				ss_tot = numpy.sum((y-numpy.mean(y))**2)

				return (1 - ss_res/ss_tot)
			"""

			popt, pcov = scipy.optimize.curve_fit(py"f", Ns, diffsNs[:, l, t, k])
			perr = numpy.sqrt(numpy.diag(pcov))

			p, = ax_gaps[s].plot(1 ./Ns, diffsNs[:, l, t, k], linewidth=0, zorder=-90+k, label=latexstring(L"$E^{\,}_{%$(k)}-E^{\,}_{%$(k-1)}$"))
			p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
			push!(ps, p)

			o, = ax_gaps[s].plot(1 ./Ns, py"f"(Ns, popt...), linestyle=linestyles[k], marker="none", color=p.get_color(), linewidth=2, zorder=-100+k, markersize=0, markeredgewidth=0, label=latexstring(L"$v^{\,}_{%$(k),%$(k-1)}(%$(lam[l])) = %$(round(popt[1]; digits=2)) \pm %$(round(perr[1]; digits=2))$"))
			push!(os, o)

			ax_gaps[s].set_xlabel(latexstring(L"$1/(2N)$"))
			ax_gaps[s].minorticks_on()

			ax_gaps[s].set_yticks([0, 1, 2, 3, 4])

			ax_gaps[s].dataLim.y0 = -0.1
			ax_gaps[s].dataLim.y1 = numpy.max(diffsNs)+0.1
			ax_gaps[s].autoscale_view()

			# if s>1
			# 	ax_gaps[s].set(yticklabels=[])
			# end
		end
	end
end

ax_gaps[1].set_title(latexstring(L"(a) \ PBC, \ $\theta=\pi/2,\ \lambda=1$"), pad=8)
ax_gaps[2].set_title(latexstring(L"(b) \ PBC, \ $\theta=19\pi/20,\ \lambda=9/10$"), pad=8)

ax_gaps[1].legend(handles=ps, loc="lower center", borderpad=0.6, markerscale=0.75, bbox_to_anchor=(1.1, 1.15), borderaxespad=0, ncol=3, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(200)

# ax_gaps[1].annotate("(a)", xy=(0.87, 0.45), xycoords="axes fraction", fontsize=16)
# ax_gaps[2].annotate("(b)", xy=(0.87, 0.45), xycoords="axes fraction", fontsize=16)

plt.draw()

if savePlot
	py"createFolderIfMissing"(graphFolder)
	fig_gaps.savefig(graphFolder * "gaps.pdf")
	fig_gaps.clf()
end

plt.close()

ps = []
data = load(dataFolder * "N=32_chi=128/ANALYZED/N=32_chi=128_lam=1.0_theta=1.570796326794896.jld2")

println(data["energies"])

mX = data["opSPT"][1, 1, :]
mY = data["opSPT"][1, 2, :]
mZ = data["opSPT"][1, 3, :]

println(numpy.mean([mX[l]*(-1)^l for l=1:length(mX)]))

p, = ax_spins_corner[1].plot(range(1, length(mX)), mX, markersize=3, lw=0.8, marker="x", color=colors[1], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{X}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax_spins_corner[1].plot(range(1, length(mY)), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{Y}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax_spins_corner[1].plot(range(1, length(mZ)), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{Z}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)

mX = data["opSPT"][6, 1, :]
mY = data["opSPT"][6, 2, :]
mZ = data["opSPT"][6, 3, :]

println(numpy.mean([mX[l]*(-1)^l for l=1:length(mX)]))

p, = ax_spins_corner[3].plot(range(1, length(mX)), mX, markersize=3, lw=0.8, marker="x", color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_corner[3].plot(range(1, length(mY)), mY,  markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_corner[3].plot(range(1, length(mZ)), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

mX = data["mags"][1, 1, :]
mY = data["mags"][1, 2, :]
mZ = data["mags"][1, 3, :]

p, = ax_spins_corner[2].plot(range(1, length(mX)), mX, markersize=4, markeredgewidth=2, lw=0.8, marker=2, color=colors[1], zorder=100, label=latexstring(L"$\langle \widehat{X}^{\,}_{j} \rangle^{\,}_{\Psi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax_spins_corner[2].plot(range(1, length(mY)), mY,  markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[4], zorder=100, label=latexstring(L"$\langle \widehat{Y}^{\,}_{j} \rangle^{\,}_{\Psi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = ax_spins_corner[2].plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, marker="x", color=colors[7], zorder=100, label=latexstring(L"$\langle \widehat{Z}^{\,}_{j} \rangle^{\,}_{\Psi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)

mX = data["mags"][6, 1, :]
mY = data["mags"][6, 2, :]
mZ = data["mags"][6, 3, :]

p, = ax_spins_corner[4].plot(range(1, length(mX)), mX, markersize=4, markeredgewidth=2, lw=0.8, marker=2, color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_corner[4].plot(range(1, length(mY)), mY,  markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_corner[4].plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, marker="x", color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

ax_spins_corner[1].set(xticklabels=[])
ax_spins_corner[3].set(xticklabels=[])

ax_spins_corner[1].minorticks_on()
ax_spins_corner[2].minorticks_on()
ax_spins_corner[3].minorticks_on()
ax_spins_corner[4].minorticks_on()

ax_spins_corner[2].set_xlabel(L"$j$")
ax_spins_corner[4].set_xlabel(L"$j$")

ax_spins_corner[1].set_xticks([0, 10, 20, 32])
ax_spins_corner[2].set_xticks([0, 10, 20, 32])
ax_spins_corner[3].set_xticks([0, 10, 20, 32])
ax_spins_corner[4].set_xticks([0, 10, 20, 32])

ax_spins_corner[1].set_xlim(0, 33)
ax_spins_corner[2].set_xlim(0, 33)
ax_spins_corner[3].set_xlim(0, 33)
ax_spins_corner[4].set_xlim(0, 33)

ax_spins_corner[1].set_ylim(-1.15, 1.15)
ax_spins_corner[2].set_ylim(-1.15, 1.15)
ax_spins_corner[3].set_ylim(-1.15, 1.15)
ax_spins_corner[4].set_ylim(-1.15, 1.15)

ax_spins_corner[1].set_title(latexstring(L"(a) \ $E^{\,}_{0}$, \ PBC, \ $\theta=\pi/2,\ \lambda=1$"), pad=8)
ax_spins_corner[3].set_title(latexstring(L"(b) \ $E^{\,}_{2}$, \ PBC, \ $\theta=\pi/2,\ \lambda=1$"), pad=8)

# axt.annotate("(c)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# ax__.annotate("(d)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# axl.annotate("(e)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# ax__l.annotate("(f)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

data = load(dataFolder * "N=32_chi=128/ANALYZED/N=32_chi=128_lam=0.9_theta=1.413716694115407.jld2")

println(data["energies"])

mX = data["opSPT"][1, 1, :]
mY = data["opSPT"][1, 2, :]
mZ = data["opSPT"][1, 3, :]

println(numpy.mean([mX[l]*(-1)^l for l=1:length(mX)]))

p, = ax_spins_phase[1].plot(range(1, length(mX)), mX, markersize=3, lw=0.8, marker="x", color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_phase[1].plot(range(1, length(mY)), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_phase[1].plot(range(1, length(mZ)), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

mX = data["opSPT"][3, 1, :]
mY = data["opSPT"][3, 2, :]
mZ = data["opSPT"][3, 3, :]

println(numpy.mean([mX[l]*(-1)^l for l=1:length(mX)]))

p, = ax_spins_phase[3].plot(range(1, length(mX)), mX, markersize=3, lw=0.8, marker="x", color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_phase[3].plot(range(1, length(mY)), mY,  markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_phase[3].plot(range(1, length(mZ)), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

mX = data["mags"][1, 1, :]
mY = data["mags"][1, 2, :]
mZ = data["mags"][1, 3, :]

p, = ax_spins_phase[2].plot(range(1, length(mX)), mX, markersize=4, markeredgewidth=2, lw=0.8, marker=2, color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_phase[2].plot(range(1, length(mY)), mY,  markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_phase[2].plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, marker="x", color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

mX = data["mags"][3, 1, :]
mY = data["mags"][3, 2, :]
mZ = data["mags"][3, 3, :]

p, = ax_spins_phase[4].plot(range(1, length(mX)), mX, markersize=4, markeredgewidth=2, lw=0.8, marker=2, color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_phase[4].plot(range(1, length(mY)), mY,  markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_spins_phase[4].plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, marker="x", color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

ax_spins_phase[1].set(xticklabels=[])
ax_spins_phase[3].set(xticklabels=[])

ax_spins_phase[1].minorticks_on()
ax_spins_phase[2].minorticks_on()
ax_spins_phase[3].minorticks_on()
ax_spins_phase[4].minorticks_on()

ax_spins_phase[2].set_xlabel(L"$j$")
ax_spins_phase[4].set_xlabel(L"$j$")

ax_spins_phase[1].set_xticks([0, 10, 20, 32])
ax_spins_phase[2].set_xticks([0, 10, 20, 32])
ax_spins_phase[3].set_xticks([0, 10, 20, 32])
ax_spins_phase[4].set_xticks([0, 10, 20, 32])

ax_spins_phase[1].set_xlim(0, 33)
ax_spins_phase[2].set_xlim(0, 33)
ax_spins_phase[3].set_xlim(0, 33)
ax_spins_phase[4].set_xlim(0, 33)

ax_spins_phase[1].set_ylim(-1.15, 1.15)
ax_spins_phase[2].set_ylim(-1.15, 1.15)
ax_spins_phase[3].set_ylim(-1.15, 1.15)
ax_spins_phase[4].set_ylim(-1.15, 1.15)

ax_spins_phase[1].set_title(latexstring(L"(c) \ $E^{\,}_{0}$, \ PBC, \ $\theta=19\pi/20,\ \lambda=9/10$"), pad=8)
ax_spins_phase[3].set_title(latexstring(L"(d) \ $E^{\,}_{2}$, \ PBC, \ $\theta=19\pi/20,\ \lambda=9/10$"), pad=8)

# axt_.annotate("(g)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# ax_.annotate("(h)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# axl_.annotate("(i)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# ax_l.annotate("(j)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

ax_spins_corner[1].legend(handles=ps[[1, 4, 2, 5, 3, 6]], loc="lower center", markerscale=1.5, bbox_to_anchor=(1.15, 1.3), borderaxespad=0, ncol=3, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(200)

plt.draw()

if savePlot
	py"createFolderIfMissing"(graphFolder)
	fig_spins_corner.savefig(graphFolder * "spins_corner.pdf")
	fig_spins_corner.clf()
	fig_spins_phase.savefig(graphFolder * "spins_phase.pdf")
	fig_spins_phase.clf()
end

plt.close()
