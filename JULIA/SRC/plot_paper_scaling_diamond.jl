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

PBCs = false

if PBCs
	graphFolder = "../GRAPHS/PBC/SCALING_DIAMOND/"
	dataFolder = "../DATA/PBC/SCALING_DIAMOND/"
else
	graphFolder = "../GRAPHS/OBC/SCALING_DIAMOND/"
	dataFolder = "../DATA/OBC/SCALING_DIAMOND/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_scaling_en = true

nE = 6

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

indSort = numpy.argsort(Ns).+1

Ns = numpy.array(Ns)[indSort]
diffsNs = numpy.array(diffsNs)[indSort, :, :, :]
	
# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

linestyles = ["-", "-", ":", "-", "--", ":", "-", "--", ":"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[7],colors[1],colors[4],colors[3],colors[6]], N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(marker=repeat([".", "x", L"$\setminus$", ".", L"$/$"], outer=5)[1:nE-1], markersize=repeat(1.5 .*[10, 7.5, 10, 10, 10], outer=5)[1:nE-1], markeredgewidth=repeat(1.5 .*[1, 1.5, 1.25, 1, 1.25], outer=5)[1:nE-1], color=cmap(numpy.linspace(0, 1, nE-1))))

if plot_scaling_en

	indSort = numpy.argsort(Ns).+1

	Ns = numpy.array(Ns)[indSort]
	diffsNs = numpy.array(diffsNs)[indSort, :, :, :]

	for l=1:length(lam)
		for t=1:length(theta)
			
			fig, ax = plt.subplots(1, 2, figsize=(6, 3))

			for k=1:nE-1

				py"""
				def f(N, a, b):
					import numpy 

					return a/N + b
				"""

				println(diffsNs[1:5:end, l, t, k])

				popt, pcov = scipy.optimize.curve_fit(py"f", Ns[1:5:end], diffsNs[1:5:end, l, t, k])
				perr = numpy.sqrt(numpy.diag(pcov))

				println(popt)

				p, = ax[1].plot(1 ./Ns[1:5:end], diffsNs[1:5:end, l, t, k], linewidth=0, zorder=111, label=latexstring(L"$E^{\,}_{%$(k)}-E^{\,}_{%$(k-1)}$"))
				p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

				o, = ax[1].plot(1 ./Ns[1:5:end], py"f"(Ns[1:5:end], popt...), color=p.get_color(), linestyle=linestyles[k], marker="none", linewidth=1.5, zorder=90+k, markersize=0, markeredgewidth=0)

				if k==3
					o.set_zorder(110)
				end
			end

			ax[1].minorticks_on()
			# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
			# ax.grid(which="major", linewidth=0.6)

			plt.xlabel(latexstring(L"$1/(2N)$"))
		
			ax[1].dataLim.y1 = numpy.max(diffsNs)
			ax[1].autoscale_view()
		
			# ax.title(latexstring(L""))

			for k=1:nE-1

				py"""
				def f(N, a, b):
					import numpy 

					return a/N + b
				"""

				println(diffsNs[2:5:end, l, t, k])

				popt, pcov = scipy.optimize.curve_fit(py"f", Ns[2:5:end], diffsNs[2:5:end, l, t, k])
				perr = numpy.sqrt(numpy.diag(pcov))

				println(popt)

				p, = ax[2].plot(1 ./Ns[2:5:end], diffsNs[2:5:end, l, t, k], linewidth=0, zorder=111, label=latexstring(L"$E^{\,}_{%$(k)}-E^{\,}_{%$(k-1)}$"))
				p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

				o, = ax[2].plot(1 ./Ns[2:5:end], py"f"(Ns[2:5:end], popt...), color=p.get_color(), linestyle=linestyles[k], marker="none", linewidth=1.5, zorder=90+k, markersize=0, markeredgewidth=0)

				if k==3
					o.set_zorder(110)
				end
			end

			ax[2].minorticks_on()
			# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
			# ax.grid(which="major", linewidth=0.6)

			ax[1].set_xlabel(latexstring(L"$1/(2N)$"))
			ax[2].set_xlabel(latexstring(L"$1/(2N)$"))

			# ax.title(latexstring(L""))

			ax[2].dataLim.y1 = numpy.max(diffsNs)
			ax[2].autoscale_view()

			ax[1].set_xlim(0.00, 0.033)
			ax[2].set_xlim(0.00, 0.033)

			ax[2].set(yticklabels=[])

			ax[2].legend(loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=0.5, ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(200)

			ax[1].annotate("(b)", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=16)
			ax[2].annotate("(c)", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=16)

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * "lam=$(lam[l])_theta=$(theta[t])/")
				fig.savefig(graphFolder * "lam=$(lam[l])_theta=$(theta[t])/" * "gaps.pdf")
				fig.clf()
			end

			plt.close()

			fig, ax = plt.subplots(figsize=(6, 3))

			p, = ax.plot(1 ./Ns[1:end], diffsNs[1:end, l, t, 4], color=colors[2], markersize=15, linewidth=0, markeredgewidth=1, zorder=100)
			p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			for n in Ns[1:5:end]
				plt.axvline(1/n, color=colors[5], ls="--", lw=1.5)
			end

			ax.minorticks_on()
			# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
			# ax.grid(which="major", linewidth=0.6)

			plt.xlabel(latexstring(L"$1/(2N)$"))
			plt.ylabel(latexstring(L"$E^{\,}_{4}-E^{\,}_{3}$"))

			ax.set_xlim(0.00, 0.033)

			ax.annotate("(a)", xy=(0.02, 0.9), xycoords="axes fraction", fontsize=16)

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * "lam=$(lam[l])_theta=$(theta[t])/")
				fig.savefig(graphFolder * "lam=$(lam[l])_theta=$(theta[t])/" * "E4-E3.pdf")
				fig.clf()
			end

			plt.close()

		end
	end
end
