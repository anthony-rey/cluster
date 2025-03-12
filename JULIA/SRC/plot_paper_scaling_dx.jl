# ENV["MPLBACKEND"]="qt5agg"

using FileIO
using PyCall
using PyPlot

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.rc("font", size=18)
PyPlot.rc("axes", titlesize=9)
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

PBCs = false

if PBCs
	graphFolder = "../graphs/pbc/scaling•Dx/"
	dataFolder = "../data/pbc/scaling•Dx/"
else
	graphFolder = "../graphs/obc/scaling•Dx/"
	dataFolder = "../data/obc/scaling•Dx/"
end

computeFolder = "analyzed/"
folders = py"folderNames"(dataFolder)

plot_scaling_en = true

nE = 4

savePlot = true

Ns = []
diffsNs = []

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

		println("··· ", dataFilenames[i])

		# ------------------------------
		if true
		# ------------------------------

			if plot_scaling_en

				E = numpy.sort(data["energies"])
	
				Es[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = numpy.sort([E[e] for e=1:nE])
				diffs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = [E[e+1]-E[e] for e=1:nE-1]

				# E = data["bulkEnergies"]
	
				# Es[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = numpy.sort([numpy.min(E[e]) for e=1:nE])
				# diffs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = [numpy.min(E[e+1])-numpy.min(E[e]) for e=1:nE-1]
				
			end
		end
	end

	if plot_scaling_en
		push!(diffsNs, diffs)
	end

end

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

indSort = numpy.argsort(Ns).+1

Ns = numpy.array(Ns)[indSort]
diffsNs = numpy.array(diffsNs)[indSort, :, :, :]

if plot_scaling_en

	indSort = numpy.argsort(Ns).+1

	Ns = numpy.array(Ns)[indSort]
	diffsNs = numpy.array(diffsNs)[indSort, :, :, :]

	linestyles = ["-", "-", "-"]
	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[1],colors[2],colors[4]], N=256)
	PyPlot.rc("axes", prop_cycle=cycler.cycler(marker=[".", "x", "+"], markersize=1.5 .*[10, 7.5, 10], markeredgewidth=1.5 .*[1, 1.5, 1.5], color=cmap(numpy.linspace(0, 1, nE-1))))

	for l=1:length(lam)
		for t=1:length(theta)
			
			fig, ax = plt.subplots(figsize=(4, 4))

			for k=1:nE-1

				p, = ax.plot(1 ./Ns[1:5:end], diffsNs[1:5:end, l, t, k], linewidth=0, zorder=100, label=latexstring(L"$E^{\,}_{%$(k)}-E^{\,}_{%$(k-1)}$"))
				p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
				
			end
				
			for k=1:nE-1

				py"""
				def f(N, a, b):
					import numpy 

					return a/N + b
				"""

				popt, pcov = scipy.optimize.curve_fit(py"f", Ns[1:5:end], diffsNs[1:5:end, l, t, k])
				perr = numpy.sqrt(numpy.diag(pcov))

				o, = ax.plot(1 ./Ns, py"f"(Ns, popt...), linestyle=linestyles[k], marker="none", linewidth=1.5, zorder=90+k, markersize=0, markeredgewidth=0)

			end

			ax.minorticks_on()
			# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
			# ax.grid(which="major", linewidth=0.6)

			plt.xlabel(latexstring(L"$1/2N$"))

			ax.legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=0.5, ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(200)

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * "lam=$(lam[l])_theta=$(theta[t])/")
				fig.savefig(graphFolder * "lam=$(lam[l])_theta=$(theta[t])/" * "gaps.pdf")
				fig.clf()
			end

			fig, ax = plt.subplots(figsize=(6, 4))

			p, = ax.plot(1 ./Ns[1:end], diffsNs[1:end, l, t, 1], color=colors[2], markersize=15, linewidth=0, markeredgewidth=1, zorder=100)
			p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			for n in Ns[1:5:end]
				plt.axvline(1/n, color=colors[5], ls="--", lw=1.5)
			end

			ax.minorticks_on()
			# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
			# ax.grid(which="major", linewidth=0.6)

			plt.xlabel(latexstring(L"$1/2N$"))
			plt.ylabel(latexstring(L"$E^{\,}_{1}-E^{\,}_{0}$"))

			ax.set_xlim(0.00, 0.033)

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * "lam=$(lam[l])_theta=$(theta[t])/")
				fig.savefig(graphFolder * "lam=$(lam[l])_theta=$(theta[t])/" * "E1-E0.pdf")
				fig.clf()
			end

			plt.close()

		end
	end
end
