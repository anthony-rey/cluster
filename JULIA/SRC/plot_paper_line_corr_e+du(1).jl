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
	graphFolder = "../GRAPHS/PBC/LINE_CORR_E+DU(1)/"
	dataFolder = "../DATA/PBC/LINE_CORR_E+DU(1)/"
else
	graphFolder = "../GRAPHS/OBC/LINE_CORR_E+DU(1)/"
	dataFolder = "../DATA/OBC/LINE_CORR_E+DU(1)/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_correlations = true
plot_lines_correlations = true

global r = 96
start_ = 1
end_ = 1
nE = 1

showPlot = false
savePlot = true

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))
	
	corrs_alg = numpy.zeros((length(lam), length(theta), 2))

	for i=1:length(dataFilenames)

		N = Int(vals[i]["N"])
		chi = Int(vals[i]["chi"])

		data = 0
		data = load(dataFolder * folder * computeFolder * dataFilenames[i])
		
		k = length(data["energies"])
		k = nE

		println("··· ", dataFilenames[i])

		# ------------------------------
		if true
		# ------------------------------

			if plot_correlations

				for j=1:k

					fig, ax = plt.subplots(figsize=(8, 3))
					
					corrX = data["corrs"][j, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
					corrY = data["corrs"][j, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
					corrZ = data["corrs"][j, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

					p, = plt.plot(range(1, r), corrX, marker="3", color=colors[1], ls="-", lw=0.5, markersize=7, markeredgewidth=1.5, label=latexstring(L"$C^{x}$"), zorder=100)
					p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
					p, = plt.plot(range(1, r), corrY, marker="4", color=colors[4], ls="-", lw=0.5, markersize=7, markeredgewidth=1.5, label=latexstring(L"$C^{y}$"), zorder=101)
					p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
					p, = plt.plot(range(1, r), corrZ, marker="1", color=colors[7], ls="-", lw=0.5, markersize=7, markeredgewidth=1.5, label=latexstring(L"$C^{z}$"), zorder=102)
					p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

					sites = range(1, r)[start_:end+1-end_]

					corrX = corrX[start_:end+1-end_]
					corrY = corrY[start_:end+1-end_]
					corrZ = corrZ[start_:end+1-end_]
					
					py"""
					import numpy 

					def f(l, a, b, c):
						return a*(l**(-b)) + c
					def g(l, a, b, c):
						return a*numpy.exp(-l/b) + c
					def oz(l, a, b, c, d):
						return a*numpy.cos(2*numpy.pi*b*l + c) + d
					"""

					try
						poptfx, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrX))
						perrfx = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrX), py"f"(collect(sites), poptfx...), length(sites)-4)
						pfx = chi2[2]

						corrs_alg[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= poptfx[2]
						corrs_alg[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= perrfx[2]

						plt.plot(sites, py"f"(collect(sites), poptfx...), ls="--", lw=1, color=colors[1], zorder=103, label=latexstring(L"$|C^{x}| \sim s^{- %$(round(poptfx[2]; digits=3)) \pm %$(round(perrfx[2]; digits=3))}, \ p = %$(Int(round(pfx[1]; digits=2)))$"))
					catch
						poptfx = 0
						perrfx = 0
						pfx = 0
					end

					try
						poptgx, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrX))
						perrgx = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrX), py"g"(collect(sites), poptgx...), length(sites)-4)
						pgx = chi2[2]		

						plt.plot(sites, py"g"(collect(sites), poptgx...), ls=":", lw=1, color=colors[1], zorder=103, label=latexstring(L"$|C^{x}| \sim e^{- s/(%$(round(poptgx[2]; digits=1)) \pm %$(round(perrgx[2]; digits=1)))}, \ p = %$(round(pgx[1]; digits=2))$"))
					catch
						poptgx = 0
						perrgx = 0
						pgx = 0
					end
					
					ax.set_xlim(0, r)
					ax.set_ylim(-0.2, 0.4)
					
					ax.minorticks_on()
					# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
					# ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper right", ncol=2, bbox_to_anchor=(1, 1), borderaxespad=0.3, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(101)

					plt.xlabel(latexstring(L"$s$"))

					plt.draw()

					if savePlot
						py"createFolderIfMissing"(graphFolder * folder * "CORRELATIONS/")
						fig.savefig(graphFolder * folder * "CORRELATIONS/" * chop(dataFilenames[i], tail=5) * ".pdf")
						fig.clf()
					end

					if showPlot
						plt.show()
					end

					plt.close()
				end
			end
		end
	end

	if plot_lines_correlations

		fig_line, ax_line = plt.subplots(figsize=(3.4, 3))

		for t=1:length(theta)

			plots, caps, errs = ax_line.errorbar(lam, corrs_alg[:, t, 1], yerr=corrs_alg[:, t, 2], elinewidth=2, color=colors[7], ecolor=colors[1], linewidth=1.5, barsabove=true, zorder=105)
			for err in errs
				err.set_capstyle("round")
			end
			
			ax_line.minorticks_on()
			# ax_line.grid(which="minor", linewidth=0.2, alpha=0.33)
			# ax_line.grid(which="major", linewidth=0.6)
			
			ax_line.set_xlabel(latexstring(L"$\lambda$"))
			ax_line.set_ylabel(latexstring(L"$\eta$"))
			
			ax_line.set_ylim(0.3, 0.6)

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder)
				fig_line.savefig(graphFolder * folder * "line_eta_xy" * ".pdf")
				fig_line.clf()
			end

			if showPlot
				plt.show()
			end

			plt.close()
		end
	end
end