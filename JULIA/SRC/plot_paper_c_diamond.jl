# ENV["MPLBACKEND"]="qt5agg"

using FileIO
using PyCall
using PyPlot

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.rc("font", size=18)
PyPlot.rc("axes", titlesize=18)
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

PBCs = false

if PBCs
	graphFolder = "../graphs/pbc/c•Diamond/"
	dataFolder = "../data/pbc/c•Diamond/"
else
	graphFolder = "../graphs/obc/c•Diamond/"
	dataFolder = "../data/obc/c•Diamond/"
end

computeFolder = "analyzed/"
folders = py"folderNames"(dataFolder)

plot_entropies_combined = true

showPlot = false
savePlot = true

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

ps = []
a = 0

Ns = []
for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	N = Int(numpy.unique([vals[i]["N"]  for i=1:length(vals)])[1])
	push!(Ns, N)

end

sort = numpy.argsort(Ns).+1
Ns = Ns[sort]
folders = folders[sort]

cs = []
cerrs = []

fig, ax = plt.subplots(figsize=(8, 4))

linestyles = ["-", "-", "-", "-"]
cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[1],colors[3],colors[4],colors[6]], N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(marker=["x", ".", "x", "."], markersize=[4, 6, 4, 6], markeredgewidth=[1.25, 1.25, 1.25, 1.25], color=cmap(numpy.linspace(0, 1, length(Ns)))))

for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))

	for i=1:length(dataFilenames)
		

		N = Int(vals[i]["N"])
		global chi = Int(vals[i]["chi"])

		data = 0
		data = load(dataFolder * folder * computeFolder * dataFilenames[i])
		
		k = length(data["energies"])

		println("··· ", dataFilenames[i])

		if plot_entropies_combined
			global a += 1
			
			Ss = data["entropies"]

			win = Int(4*N/5-4*(N%5)/5-10)

			sites = range(1, size(Ss, 2))

			sites = sites[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]
			S = Ss[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]

			py"""
			def f(l, c, const):
				import numpy

				N = int(l[len(l)//2]*2)

				return c/3 * numpy.log(N/numpy.pi * numpy.sin(numpy.pi*l/N)) + const
			"""

			py"""
			def g(x, a, b):
				import numpy

				return a*x + b
			"""

			# popt, pcov = scipy.optimize.curve_fit(py"f", collect(sites), S)
			# perr = numpy.sqrt(numpy.diag(pcov))

			x = 1/6 * numpy.log(2*N/numpy.pi .* numpy.sin(numpy.pi .*sites ./N))

			popt, pcov = scipy.optimize.curve_fit(py"g", x, S)
			perr = numpy.sqrt(numpy.diag(pcov))

			println(popt, perr)

			# label=latexstring(L"$\mathsf{c}= %$(round(popt[1]; digits=5)) \pm %$(round(perr[1]; digits=5))$ for $2N=%$(N)$")

			p, = ax.plot(x, S, linewidth=0.5, label=latexstring(L"$2N=%$(N)$"), zorder=100)
			p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
			push!(ps, p)

			push!(cs, popt[1])
			push!(cerrs, perr[1])

			ax.minorticks_on()
			# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
			# ax.grid(which="major", linewidth=0.6)

			plt.xlabel(latexstring(L"$\frac{1}{6} \ln \left[ \frac{4N}{\pi} \sin\frac{\pi l}{2N} \right]$"))
			plt.ylabel(latexstring(L"$S(l)$"))

			plt.draw()

		end
	end
end

ax.set_ylim(0.7, 2)

ax.set_xticks([0.6, 0.8, 1])
ax.set_yticks([1, 1.5, 2])

axin = ax.inset_axes([0.15, 0.5, 0.3, 0.45])
axin.set_zorder(102)
		
plots, caps, errs = axin.errorbar(Ns, cs, cerrs, ecolor=colors[1], elinewidth=1.5, marker="x", markersize=5, markeredgewidth=1.5, linewidth=0.5, color=colors[7], zorder=110)
plots.set_marker(matplotlib.markers.MarkerStyle(plots.get_marker(), capstyle="round"))
# p, = axin.plot(Ns, 1 .- cs, marker="+", markersize=7, markeredgewidth=1.5, linewidth=0.5, color=colors[7], zorder=110)
# p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

for err in errs
	err.set_capstyle("round")
end

axin.set_xticks([0, 500, 1000])

axin.set_xlim(0, 1200)
axin.set_ylim(0.7, 1)

axin.set_xlabel(L"$2N$")
axin.set_ylabel(L"$\mathsf{c}$")

axin.minorticks_on()

# axin.set_yscale("log")

ax.legend(handles=ps, loc="upper right", labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, ncol=2).set_zorder(110)

if savePlot
	py"createFolderIfMissing"(graphFolder)
	fig.savefig(graphFolder * "chi=$(chi)" * ".pdf")
	ax.cla()
end

if showPlot
	plt.show()
end

plt.close()