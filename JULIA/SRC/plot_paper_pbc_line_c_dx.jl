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
	graphFolder = "../GRAPHS/PBC/LINE_C_DX/"
	dataFolder = "../DATA/PBC/LINE_C_DX/"
else
	graphFolder = "../GRAPHS/OBC/LINE_C_DX/"
	dataFolder = "../DATA/OBC/LINE_C_DX/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_entropies = false
plot_lines_c = true

win = 20

showPlot = false
savePlot = true

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

fig, ax = plt.subplots(figsize=(4, 4))

# folders = ["N=64_chi=256/"]
for folder in folders

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

		if plot_entropies | plot_lines_c
			
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

			label=latexstring(L"$S = \frac{(%$(round(popt[1]; digits=3)) \pm %$(round(perr[1]; digits=3)))}{3} \ln\left[\frac{N}{\pi}\sin\frac{\pi \ell}{N} \right] + (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3)))$")

			ax.scatter(sites, S, marker="x", color=colors[2], zorder=100)
			ax.plot(sites, py"f"(collect(sites), popt...), linestyle=":", color=colors[5], label=label, zorder=100)

			ax.minorticks_on()
			ax.grid(which="minor", linewidth=0.2, alpha=0.33)
			ax.grid(which="major", linewidth=0.6)
			ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)

			plt.xlabel(latexstring(L"$\ell$"))
			plt.ylabel(latexstring(L"$S$"))

			plt.draw()

			if savePlot & plot_entropies
				py"createFolderIfMissing"(graphFolder * folder * "entropies/")
				fig.savefig(graphFolder * folder * "entropies/" * chop(dataFilenames[i], tail=5) * ".pdf")
				ax.cla()
			end

			if showPlot
				plt.show()
			end

			ax.cla()
			plt.close()
		end
	end

	if plot_lines_c

		for l=1:length(lam)

			ax.errorbar(theta, cs[l, :, 1], cs[l, :, 2], ecolor=colors[5], elinewidth=0.5, color=colors[1], marker=".", linewidth=0.5, markersize=10, zorder=105)

			ax.minorticks_on()
			# ax.grid(which="minor", linewidth=0.2)
			# ax.grid(which="major", linewidth=0.6)
			
			ax.axvline(0.57, color="k", linewidth=1, linestyle="--")

			ax.set_xlabel(latexstring(L"$\theta$"))
			ax.set_ylabel(latexstring(L"$\mathsf{c}$"))

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder)
				fig.savefig(graphFolder * folder * "win=$(win)_lam=$(lam[l]).pdf")
				ax.cla()
			end
		end
	end
end