# ENV["MPLBACKEND"]="qt5agg"

using FileIO
using PyCall
using PyPlot

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.rc("font", size=18)
PyPlot.rc("legend", fontsize=14)
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
seaborn = pyimport("seaborn")
scipy = pyimport("scipy")
cycler = pyimport("cycler")

PBCs = true

if PBCs
	graphFolder = "../GRAPHS/PBC/LINE_C_E+DU(1)/"
	dataFolder = "../DATA/PBC/LINE_C_E+DU(1)/"
else
	graphFolder = "../GRAPHS/OBC/LINE_C_E+DU(1)/"
	dataFolder = "../DATA/OBC/LINE_C_E+DU(1)/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_entropies = true
plot_lines_c = false
plot_lines_c_combined = true

win = 10

showPlot = false
savePlot = true

Ns = []
chis = []
for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	N = Int(numpy.unique([vals[i]["N"] for i=1:length(vals)])[1])
	chi = Int(numpy.unique([vals[i]["chi"] for i=1:length(vals)])[1])
	push!(Ns, N)
	push!(chis, chi)
end

indWin = numpy.where(Ns.>win)[1].+1
Ns = Ns[indWin]
chis = chis[indWin]
folders = folders[indWin]

ChiNs = numpy.asarray([Ns, chis])
sort = numpy.lexsort(ChiNs, axis=0).+1

Ns = Ns[sort]
chis = chis[sort]
folders = folders[sort]

chis_ = numpy.unique(chis)

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

fig, ax = plt.subplots(figsize=(5.1, 4))

for x=1:length(chis_)

	indN = numpy.where(chis.==chis_[x])[1].+1
	println(indN)

	NsN = Ns[indN]
	foldersN = folders[indN]

	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[6],colors[4],colors[3],colors[1]], N=256)
	PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=["--", "--", "-", "-"], marker=[".", "x", ".", "x"], markersize=[10, 7, 10, 7], markeredgewidth=[1.5, 1.5, 1.5, 1.5], color=cmap(numpy.linspace(0, 1, 4))))

	fig_combined, ax_combined = plt.subplots(figsize=(4.1, 4))

	for n=1:length(NsN)
		
		folder = foldersN[n]

		engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

		global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
		global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))
		
		cs = numpy.zeros((length(lam), length(theta), 2))

		for i=1:length(dataFilenames)

			N = Int(vals[i]["N"])
			chi = Int(vals[i]["chi"])

			data = 0
			data = load(dataFolder * folder * computeFolder * dataFilenames[i])
			
			k = length(data["energies"])

			println("··· ", dataFilenames[i])

			if plot_entropies | plot_lines_c | plot_lines_c_combined
				
				Ss = data["entropies"]

				sites = range(1, size(Ss, 2))

				win = Int(N/2+(N/2)%2)

				sites = sites[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]
				S = Ss[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]

				py"""
				def f(l, c, const):
					import numpy

					N = int(l[len(l)//2]*2)

					return c/3 * numpy.log(N/numpy.pi * numpy.sin(numpy.pi*l/N)) + const
				"""

				popt, pcov = scipy.optimize.curve_fit(py"f", collect(sites), S)
				perr = numpy.sqrt(numpy.diag(pcov))

				cs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= popt[1]
				cs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= perr[1]
				
				label=latexstring(L"$S = \frac{(%$(round(popt[1]; digits=5)) \pm 2\cdot 10^{-5})}{3} \ln\left[\frac{2N}{\pi}\sin\frac{\pi l}{2N} \right] + \mathrm{ const}$")

				ax.scatter(sites, S, marker="x", s=50, linewidths=2, color=colors[1], zorder=100)
				ax.plot(sites, py"f"(collect(sites), popt...), markersize=0, linewidth=2, linestyle=":", color=colors[7], label=label, zorder=99)

				ax.minorticks_on()
				# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
				# ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="lower center", labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)

				ax.set_xlabel(latexstring(L"$l$"))
				ax.set_ylabel(latexstring(L"$S$"))

				plt.draw()

				if savePlot & plot_entropies
					py"createFolderIfMissing"(graphFolder * folder * "ENTROPIES/")
					fig.savefig(graphFolder * folder * "ENTROPIES/" * chop(dataFilenames[i], tail=5) * ".pdf")
					ax.cla()
				end

				if showPlot
					plt.show()
				end

				ax.cla()
			end
		end

		if plot_lines_c

			for t=1:length(theta)

				ax.errorbar(lam, cs[:, t, 1], cs[:, t, 2], ecolor=colors[5], elinewidth=1, color=colors[2], marker=".", linewidth=0.5, markersize=5, zorder=105)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)

				plt.xlabel(latexstring(L"$\lambda$"))

				plt.draw()

				if savePlot
					py"createFolderIfMissing"(graphFolder * folder)
					fig.savefig(graphFolder * folder * "win=$(win)_theta=$(theta[t]).pdf")
					ax.cla()
				end
			end
		end

		if plot_lines_c_combined
			
			for t=1:length(theta)

				p, = ax_combined.plot(lam, cs[:, t, 1], linewidth=1, markeredgewidth=2, zorder=105, label=latexstring(L"$2N = %$(Ns[n])$"))
				p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			end
		end
	end

	ax_combined.minorticks_on()
	# ax_combined.grid(which="minor", linewidth=0.2, alpha=0.33)
	# ax_combined.grid(which="major", linewidth=0.6)
	ax_combined.legend(loc="upper left", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)

	ax_combined.set_xlabel(latexstring(L"$\lambda$"))
	ax_combined.set_ylabel(latexstring(L"$\mathsf{c}$"))

	plt.draw()

	if savePlot
		py"createFolderIfMissing"(graphFolder)
		fig_combined.savefig(graphFolder * "win=$(win)_chi=$(chis_[x])" * ".pdf")
		fig_combined.clf()
	end

	if showPlot
		plt.show()
	end

	plt.close()
end