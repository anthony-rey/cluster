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
scipy = pyimport("scipy")
cycler = pyimport("cycler")
seaborn = pyimport("seaborn")

PBCs = true

if PBCs
	graphFolder = "../GRAPHS/PBC/OP_EU(1)/"
	dataFolder = "../DATA/PBC/OP_EU(1)/"
else
	graphFolder = "../GRAPHS/OBC/OP_EU(1)/"
	dataFolder = "../DATA/OBC/OP_EU(1)/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_mags_chi = true

showPlot = false
savePlot = true

Ns = []
chis = []
for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))

	N = Int(numpy.unique([vals[i]["N"] for i=1:length(vals)])[1])
	chi = Int(numpy.unique([vals[i]["chi"] for i=1:length(vals)])[1])
	push!(Ns, N)
	push!(chis, chi)
end

ChiNs = numpy.asarray([chis, Ns])
sort = numpy.lexsort(ChiNs, axis=0).+1

Ns = Ns[sort]
chis = chis[sort]
folders = folders[sort]

Ns_ = numpy.unique(Ns)

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[1],colors[4],colors[6]], N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=["-", "--", ":"], marker=[".", L"$/$", L"$\setminus$"], markersize=[10, 10, 10], markeredgewidth=[1.5, 1.5, 1.5], color=cmap(numpy.linspace(0, 1, 3))))

fig, ax = plt.subplots(figsize=(6, 3))

for n=1:length(Ns_)

	indN = numpy.where(Ns.==Ns_[n])[1].+1
	println(indN)

	chisN = chis[indN]
	foldersN = folders[indN]

	magsX = numpy.zeros((length(chisN), length(lam), length(theta), 6))

	for x=1:length(chisN)
		
		folder = foldersN[x]

		engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

		for i=1:length(dataFilenames)

			N = Int(vals[i]["N"])
			chi = Int(vals[i]["chi"])

			data = 0
			data = load(dataFolder * folder * computeFolder * dataFilenames[i])
			
			k = length(data["energies"])

			println("··· ", dataFilenames[i])

			if plot_mags_chi
			
				mX = data["mags"][1, 1, :]
				mY = data["mags"][1, 2, :]
				mZ = data["mags"][1, 3, :]

				magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mX)
				magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean([mX[l]*(-1)^l for l=1:length(mX)])
				magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(mY)
				magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 4] .= numpy.mean([mY[l]*(-1)^l for l=1:length(mY)])
				magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 5] .= numpy.mean(mZ)
				magsX[x, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 6] .= numpy.mean([mZ[l]*(-1)^l for l=1:length(mZ)])
			end
		end
	end

	if plot_mags_chi
		
		for l=1:length(lam)
			for t=1:length(theta)

				p, = ax.plot(chisN, numpy.abs(magsX[:, l, t, 2])+numpy.abs(magsX[:, l, t, 4]), linewidth=1, markeredgewidth=2, zorder=105, label=latexstring(L"$2N=%$(Ns_[n])$"))
				p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			end
		end
	end
end

ax.set_xlim(0, 135)

ax.minorticks_on()
ax.legend(loc="upper right", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)

ax.set_xlabel(latexstring(L"$\chi$"))
ax.set_ylabel(latexstring(L"$|m^{x}_{\mathrm{sta}}|+|m^{y}_{\mathrm{sta}}|$"))

plt.draw()

if savePlot
	py"createFolderIfMissing"(graphFolder)
	fig.savefig(graphFolder * "op" * ".pdf")
	fig.clf()
end

if showPlot
	plt.show()
end

plt.close()