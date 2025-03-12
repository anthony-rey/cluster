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
	graphFolder = "../GRAPHS/PBC/SCALING_E+DU(1)/"
	dataFolder = "../DATA/PBC/SCALING_E+DU(1)/"
else
	graphFolder = "../GRAPHS/OBC/SCALING_E+DU(1)/"
	dataFolder = "../DATA/OBC/SCALING_E+DU(1)/"
end

computeFolder = "ANALYZED/"
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

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

fig, ax = plt.subplots(1, 4, figsize=(15, 3))

if plot_scaling_en

	indSort = numpy.argsort(Ns).+1

	Ns = numpy.array(Ns)[indSort]
	diffsNs = numpy.array(diffsNs)[indSort, :, :, :]

	s = 0
	maxs = []
	global os = []
	for l=1:length(lam)
		for t=1:length(theta)

			linestyles = ["-", "--", ":"]
			cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[1],colors[4],colors[6]], N=256)
			PyPlot.rc("axes", prop_cycle=cycler.cycler(marker=[".", "+", "x"], markersize=2 .*[10, 10, 7.5], markeredgewidth=2 .*[1.5, 1.5, 1.5], color=cmap(numpy.linspace(0, 1, nE-1))))

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
				
			if lam[l] in [0., 0.01, 0.05, 0.1]
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

					# print(popt)

					p, = ax[s].plot(1 ./Ns, diffsNs[:, l, t, k], linewidth=0, zorder=100+k, label=latexstring(L"$E^{\,}_{%$(k)}-E^{\,}_{%$(k-1)}$"))
					p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
					push!(ps, p)

					o, = ax[s].plot(1 ./Ns, py"f"(Ns, popt...), linestyle=linestyles[k], marker="none", color=p.get_color(), linewidth=2, zorder=90+k, markersize=0, label=latexstring(L"$v^{\,}_{%$(k),%$(k-1)}(%$(lam[l])) = %$(round(popt[1]; digits=2)) \pm %$(round(perr[1]; digits=2))$"))
					push!(os, o)

					ax[s].set_xlabel(latexstring(L"$1/(2N)$"))

					ax[s].dataLim.y1 = numpy.max(diffsNs)
					ax[s].autoscale_view()

					ax[s].set_title(latexstring(L"$\lambda=%$(lam[l])$"))

					if s>1
						ax[s].set(yticklabels=[])
					end
				end
			end
		end
	end
end

ax[4].legend(handles=ps, loc="center left", bbox_to_anchor=(1, 0.5), borderaxespad=1, ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(200)

plt.draw()

if savePlot
	py"createFolderIfMissing"(graphFolder)
	fig.savefig(graphFolder * "gaps.pdf")
	fig.clf()
end

plt.close()

fig, ax = plt.subplots(1, 2, figsize=(17, 3))

py"""
def g(x, a, b, c):
	import numpy 

	return a*x**b + c
"""

for t=1:length(theta)

	sites = range(1, length(lam)-4)
	x = lam[sites]

	popt, pcov = scipy.optimize.curve_fit(py"g", x, slopes[sites, t, 1])
	perr = numpy.sqrt(numpy.diag(pcov))
	
	p1, = ax[1].plot(lam, slopes[:, t, 1], marker="x", markersize=10, color=colors[7], linewidth=0, markeredgewidth=3, zorder=100)
	p1.set_marker(matplotlib.markers.MarkerStyle(p1.get_marker(), capstyle="round"))
	o1, = ax[1].plot(x, py"g"(x, popt...), markersize=0, linewidth=3, color=colors[1], zorder=100, label=latexstring(L"$v^{\,}_{1,0} - %$(round(popt[3]; digits=3)) \sim \lambda^{%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3))}$"))

	sites = range(1, length(lam)-4)
	x = lam[sites]

	popt, pcov = scipy.optimize.curve_fit(py"g", x, slopes[sites, t, 3])
	perr = numpy.sqrt(numpy.diag(pcov))

	p2, = ax[2].plot(lam, slopes[:, t, 3], marker="x", markersize=10, color=colors[7], linewidth=0, markeredgewidth=3, zorder=100)
	p2.set_marker(matplotlib.markers.MarkerStyle(p2.get_marker(), capstyle="round"))
	o2, = ax[2].plot(x, py"g"(x, popt...), markersize=0, linewidth=3, color=colors[1], zorder=100, label=latexstring(L"$v^{\,}_{3,2} - %$(round(popt[3]; digits=3)) \sim \lambda^{%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3))}$"))

	ax[1].set_title(latexstring(L"$E^{\,}_{1} - E^{\,}_{0}$"), pad=10)
	ax[2].set_title(latexstring(L"$E^{\,}_{3} - E^{\,}_{2}$"), pad=10)

	ax[1].set_xlabel(latexstring(L"$\lambda$"))
	ax[2].set_xlabel(latexstring(L"$\lambda$"))

	ax[1].minorticks_on()
	ax[2].minorticks_on()

	ax[1].set_ylabel(latexstring(L"$v^{\,}_{1,0}$"))
	ax[2].set_ylabel(latexstring(L"$v^{\,}_{3,2}$"))

	ax[1].legend(fontsize=14, handlelength=1.8, handletextpad=0.6, labelspacing=0.8, loc="lower left").set_zorder(200)
	ax[2].legend(fontsize=14, handlelength=1.8, handletextpad=0.6, labelspacing=0.8, loc="lower left").set_zorder(200)

	plt.draw()

	if savePlot
		py"createFolderIfMissing"(graphFolder)
		fig.savefig(graphFolder * "slopes.pdf")
		fig.clf()
	end

	plt.close()

end
