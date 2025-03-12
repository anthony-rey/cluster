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
scipy = pyimport("scipy")
cycler = pyimport("cycler")
seaborn = pyimport("seaborn")

PBCs = false

if PBCs
	graphFolder = "../GRAPHS/PBC/LINE_OP_EU(1)/"
	dataFolder = "../DATA/PBC/LINE_OP_EU(1)/"
else
	graphFolder = "../GRAPHS/OBC/LINE_OP_EU(1)/"
	dataFolder = "../DATA/OBC/LINE_OP_EU(1)/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_lines_mag = true

nE = 1

showPlot = false
savePlot = true

Ns = []
for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	N = Int(numpy.unique([vals[i]["N"]  for i=1:length(vals)])[1])
	push!(Ns, N)

end

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[1],colors[2],colors[4],colors[5],colors[7],colors[8]], N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=["-", "--", "-", "--", "-", "--"], marker=[".", ".", 2, 2, 3, 3], markersize=[6, 6, 6, 6, 6, 6], color=cmap(numpy.linspace(0, 1, 2*length(Ns)))))

fig, ax = plt.subplots(figsize=(8, 3))
fig_, ax_ = plt.subplots(figsize=(8, 3))

sort = numpy.argsort(Ns).+1
folders = folders[sort]

for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))
	
	N = Int(numpy.unique([vals[i]["N"]  for i=1:length(vals)])[1])
	chi = Int(numpy.unique([vals[i]["chi"]  for i=1:length(vals)])[1])

	mags = numpy.zeros((nE, length(lam), length(theta), 6))

	for i=1:length(dataFilenames)

		data = 0
		data = load(dataFolder * folder * computeFolder * dataFilenames[i])
		
		k = length(data["energies"])

		println("··· ", dataFilenames[i])
			
		mX = data["mags"][1, 1, :]
		mY = data["mags"][1, 2, :]
		mZ = data["mags"][1, 3, :]

		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mX)
		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean([mX[l]*(-1)^l for l=1:length(mX)])
		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(mY)
		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 4] .= numpy.mean([mY[l]*(-1)^l for l=1:length(mY)])
		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 5] .= numpy.mean(mZ)
		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 6] .= numpy.mean([mZ[l]*(-1)^l for l=1:length(mZ)])

	end

			py"""
			import numpy 

			def f(s, a, b, c, d):
				return a*numpy.heaviside(s-b, 1)*numpy.abs(s-b)**(c) + d
			def g(s, a, b):
				return a*s + b
			"""

			# if N == 256
			# 	println(theta[100])
			# 	r_ = theta[85:100].-pi/4
		
			# 	poptoz, pcov = scipy.optimize.curve_fit(py"g", collect(numpy.log(r_)), numpy.log(numpy.abs(mags[1, 1, 85:100, 4])), [0.1, 0])
			# 	perroz = numpy.sqrt(numpy.diag(pcov))
			# 	# chi2 = scipy.stats.chisquare(numpy.abs(mags[1:18, t, 1]), py"f"(collect(r_), poptoz...), length(r_)-5)
			# 	# poz = chi2[2]
			# 	println(poptoz, perroz)
			# 	ax_.plot(r_, numpy.exp(py"g"(collect(numpy.log(r_)), poptoz...)), ls="-", lw=2, color="0", zorder=101)		

			# 	ax_.plot(theta[85:100].-pi/4, numpy.abs(mags[1, 1, 85:100, 4]), markeredgewidth=1, lw=1, zorder=105, label=latexstring(L"$|m^{x}_{\mathrm{sta}}|$ : $2N=%$(N)$"))
			# end


	p1, = ax.plot(theta, numpy.abs(mags[1, 1, :, 2]), markeredgewidth=1, lw=1, zorder=105, label=latexstring(L"$|m^{x}_{\mathrm{sta}}|$ : $2N=%$(N)$"))
	p1.set_marker(matplotlib.markers.MarkerStyle(p1.get_marker(), capstyle="round"))

	p2, = ax.plot(theta, numpy.abs(mags[1, 1, :, 4]), markeredgewidth=1, lw=1, zorder=105, label=latexstring(L"$|m^{y}_{\mathrm{sta}}|$ : $2N=%$(N)$"))
	p2.set_marker(matplotlib.markers.MarkerStyle(p2.get_marker(), capstyle="round"))

	if N==128
		p1.set_zorder(107)
		p2.set_zorder(107)
	end

end

ax_.set_xscale("log")
ax_.set_yscale("log")

ax.axvline(pi/4, color="k", linewidth=1, linestyle=":")

ax.minorticks_on()
# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
# ax.grid(which="major", linewidth=0.6)
ax.legend(loc="center left", ncol=1, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)

plt.xlabel(latexstring(L"$\theta$"))

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