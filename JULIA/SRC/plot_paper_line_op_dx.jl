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
	graphFolder = "../GRAPHS/PBC/LINE_OP_DX/"
	dataFolder = "../DATA/PBC/LINE_OP_DX/"
else
	graphFolder = "../GRAPHS/OBC/LINE_OP_DX/"
	dataFolder = "../DATA/OBC/LINE_OP_DX/"
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
		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean([mX[l]*(-1)^l for l=Int(N/2-N/4):Int(N/2+N/4)])
		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(mY)
		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 4] .= numpy.mean([mY[l]*(-1)^l for l=Int(N/2-N/4):Int(N/2+N/4)])
		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 5] .= numpy.mean(mZ)
		mags[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 6] .= numpy.mean([mZ[l]*(-1)^l for l=Int(N/2-N/4):Int(N/2+N/4)])

	end

	py"""
	import numpy 

	def f(s, a, b, c, d):
		# return a*numpy.heaviside(b-s, 1)*numpy.abs(s-b)**(c) + d
		return a*numpy.heaviside(s-b, 1)*numpy.abs(s-b)**(c) + d
	def g(s, a, b, c):
		return a*numpy.log(numpy.abs(s-b)) + c
	# def g(s, a, b):
	# 	return a*numpy.log(numpy.abs(s)) + b
	"""

	py"""
	def r2(y, fit):
		import numpy

		ss_res = numpy.sum((y-fit)**2)
		ss_tot = numpy.sum((y-numpy.mean(y))**2)

		return (1 - ss_res/ss_tot)
	"""

	# if N==256
	# 	println(theta[[54:60; 80:102]])
	# 	r_ = theta[[54:60; 80:102]]

	# 	poptf, pcov = scipy.optimize.curve_fit(py"f", collect(r_), numpy.abs(mags[1, 1, [54:60; 80:102], 2]), [1, 0.58, 0.15, 0])
	# 	perrf = numpy.sqrt(numpy.diag(pcov))
	# 	# chi2 = scipy.stats.chisquare(numpy.abs(mags[1:end, t, 1]), py"f"(collect(r_), poptf...), length(r_)-5)
	# 	# poz = chi2[2]
	# 	println(poptf, perrf)
	# 	# ax.plot(r_, py"f"(collect(r_), poptf...), ls="-", lw=1, color="0", zorder=200)				

	# 	# axins = ax.inset_axes([0.5, 0.27, 0.4, 0.4])
	# 	# axins.set_zorder(100)

	# 	r_ = theta[53:60]
	# 	# r_ = theta[53:60].-0.65

	# 	poptg, pcov = scipy.optimize.curve_fit(py"g", collect(r_), numpy.log(numpy.abs(mags[1, 1, 53:60, 2])), [0.5, 0.575, 1])
	# 	# poptg, pcov = scipy.optimize.curve_fit(py"g", collect(r_), numpy.log(numpy.abs(mags[1, 1, 53:60, 2])), [0.5, 1])
	# 	perrg = numpy.sqrt(numpy.diag(pcov))
	# 	# chi2 = scipy.stats.chisquare(numpy.abs(mags[1:end, t, 1]), py"f"(collect(r_), poptg...), length(r_)-5)
	# 	# poz = chi2[2]
	# 	println(poptg, perrg)

	# 	r2_ = py"r2"(numpy.log(numpy.abs(mags[1, 1, 53:60, 2])), py"g"(collect(r_), poptg...))
	# 	println(r2_)

	# 	# axins.plot(numpy.log(numpy.abs(r_.-poptg[2])), py"g"(collect(r_), poptg...), ls="-", lw=2, color="k", zorder=101, label=latexstring(L"$\beta^{\,}_{\mathrm{tri}}\,\ln |\lambda-\lambda^{\,}_{2N,\mathrm{tri}}|+\mathrm{const}$"))
	# 	# # axins.plot(numpy.log(numpy.abs(r_)), py"g"(collect(r_), poptg...), ls="-", lw=2, color="k", zorder=101, label=latexstring(L"$\beta^{\,}_{\mathrm{tri}}\,\ln |\lambda-\lambda^{\,}_{2N,\mathrm{tri}}|+\mathrm{const}$"))

	# 	# p, = axins.plot(numpy.log(numpy.abs(theta[10:69].-poptg[2])), numpy.log(numpy.abs(mags[1, 1, 10:69, 2])), linewidth=0.1, marker="x", color=colors[4], markersize=6, markeredgewidth=1.5, zorder=100)	
	# 	# # p, = axins.plot(numpy.log(numpy.abs(theta[10:69].-0.65)), numpy.log(numpy.abs(mags[1, 1, 10:69, 2])), linewidth=0.1, marker="x", color=colors[4], markersize=6, markeredgewidth=1.5, zorder=100)	
	# 	# p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
	# end

	# if N==256
	# 	println(theta[[40:50; 70:80]])
	# 	r_ = theta[[40:50; 70:80]]

	# 	poptf, pcov = scipy.optimize.curve_fit(py"f", collect(r_), numpy.abs(mags[1, 1, [40:50; 70:80], 5]), [1, 0.576, 0.15, 0])
	# 	perrf = numpy.sqrt(numpy.diag(pcov))
	# 	# chi2 = scipy.stats.chisquare(numpy.abs(mags[1:end, t, 1]), py"f"(collect(r_), poptf...), length(r_)-5)
	# 	# poz = chi2[2]
	# 	println(poptf, perrf)
	# 	ax.plot(r_, py"f"(collect(r_), poptf...), ls="-", lw=1, color="0", zorder=200)				

	# 	axins = ax.inset_axes([0.5, 0.27, 0.4, 0.4])
	# 	axins.set_zorder(100)

	# 	r_ = theta[80:90]
	# 	# r_ = theta[80:90].-0.65

	# 	poptg, pcov = scipy.optimize.curve_fit(py"g", collect(r_), numpy.log(numpy.abs(mags[1, 1, 80:90, 5])), [0.5, 0.575, 1])
	# 	# poptg, pcov = scipy.optimize.curve_fit(py"g", collect(r_), numpy.log(numpy.abs(mags[1, 1, 80:90, 2])), [0.5, 1])
	# 	perrg = numpy.sqrt(numpy.diag(pcov))
	# 	# chi2 = scipy.stats.chisquare(numpy.abs(mags[1:end, t, 1]), py"f"(collect(r_), poptg...), length(r_)-5)
	# 	# poz = chi2[2]
	# 	println(poptg, perrg)

	# 	r2_ = py"r2"(numpy.log(numpy.abs(mags[1, 1, 80:90, 5])), py"g"(collect(r_), poptg...))
	# 	println(r2_)

	# 	axins.plot(numpy.log(numpy.abs(r_.-poptg[2])), py"g"(collect(r_), poptg...), ls="-", lw=2, color="k", zorder=101, label=latexstring(L"$\beta^{\,}_{\mathrm{tri}}\,\ln |\lambda-\lambda^{\,}_{2N,\mathrm{tri}}|+\mathrm{const}$"))
	# 	# axins.plot(numpy.log(numpy.abs(r_)), py"g"(collect(r_), poptg...), ls="-", lw=2, color="k", zorder=101, label=latexstring(L"$\beta^{\,}_{\mathrm{tri}}\,\ln |\lambda-\lambda^{\,}_{2N,\mathrm{tri}}|+\mathrm{const}$"))

	# 	p, = axins.plot(numpy.log(numpy.abs(theta[70:end].-poptg[2])), numpy.log(numpy.abs(mags[1, 1, 70:end, 5])), linewidth=0.1, marker="x", color=colors[4], markersize=6, markeredgewidth=1.5, zorder=100)	
	# 	# p, = axins.plot(numpy.log(numpy.abs(theta[10:69].-0.65)), numpy.log(numpy.abs(mags[1, 1, 10:69, 2])), linewidth=0.1, marker="x", color=colors[4], markersize=6, markeredgewidth=1.5, zorder=100)	
	# 	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
	# end

	p1, = ax.plot(theta, numpy.abs(mags[1, 1, :, 2]), markeredgewidth=1, lw=1, zorder=105, label=latexstring(L"$|m^{x}_{\mathrm{sta}}|_{	,}$ $2N=%$(N)$"))
	p1.set_marker(matplotlib.markers.MarkerStyle(p1.get_marker(), capstyle="round"))

	p2, = ax.plot(theta, numpy.abs(mags[1, 1, :, 5]), markeredgewidth=1, lw=1, zorder=105, label=latexstring(L"$|m^{z}_{\mathrm{uni}}|_{	,}$ $2N=%$(N)$"))
	p2.set_marker(matplotlib.markers.MarkerStyle(p2.get_marker(), capstyle="round"))

	if N==64
		p1.set_zorder(107)
		p2.set_zorder(107)
	end

end

ax.minorticks_on()
# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
# ax.grid(which="major", linewidth=0.6)
# ax.legend(loc="center left", ncol=1, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)
# ax.legend(loc="lower center", ncol=3, labelspacing=0.4, columnspacing=1.2, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(0.5, 1.03)).set_zorder(110)
ax.legend(loc="center left", ncol=1, labelspacing=0.4, columnspacing=1.2, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(0.02, 0.48)).set_zorder(110)

# ax.annotate("(e)", xy=(0.95, 0.88), xycoords="axes fraction", fontsize=16)

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