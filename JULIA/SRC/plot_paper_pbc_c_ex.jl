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

PBCs = true

if PBCs
	graphFolder = "../GRAPHS/PBC/C_EX/"
	dataFolder = "../DATA/PBC/C_EX/"
else
	graphFolder = "../GRAPHS/OBC/C_EX/"
	dataFolder = "../DATA/OBC/C_EX/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_entropies = false
plot_scaling_c = true

win = 8

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
# indWin = numpy.where(Ns.==50)[1].+1
# Ns = Ns[indWin]
# chis = chis[indWin]
# folders = folders[indWin]

ChiNs = numpy.asarray([chis, Ns])
sort = numpy.lexsort(ChiNs, axis=0).+1

Ns = Ns[sort]
chis = chis[sort]
folders = folders[sort]

Ns_ = numpy.unique(Ns)

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

fig, ax = plt.subplots(figsize=(5, 4))
fig_all, ax_all = plt.subplots(figsize=(8.5, 4))

csN = []
for n=1:length(Ns_)

	indN = numpy.where(Ns.==Ns_[n])[1].+1
	println(indN)

	chisN = chis[indN]
	foldersN = folders[indN]

	# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[6],colors[5],colors[3],colors[2],colors[1]], N=256)
	# PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=["--", "--", "-", "-", ":"], marker=[".", "x", ".", "x","."], markersize=[6, 3, 6, 3, 6], color=cmap(numpy.linspace(0, 1, 5))))

	fig_chi, ax_chi = plt.subplots(figsize=(3, 3))

	csX = []
	csXerr = []
	for x=1:length(chisN)
		
		folder = foldersN[x]

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

			win = Int(N/2+(N/2)%2)
			# win = N-4

			println("··· ", dataFilenames[i])

			if plot_entropies | plot_scaling_c
				
				Ss = data["entropies"]

				sites = range(1, size(Ss, 2))

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

				label=latexstring(L"$S = \frac{(%$(round(popt[1]; digits=5)) \pm %$(round(perr[1]; digits=5)))}{6} \ln\left[\frac{2N}{\pi}\sin\frac{\pi \ell}{N} \right] + \mathrm{ const}$")

				ax.scatter(sites, S, marker="x", color=colors[2], zorder=100)
				ax.plot(sites, py"f"(collect(sites), popt...), markersize=0, linestyle=":", color=colors[5], label=label, zorder=100)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2, alpha=0.33)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0).set_zorder(101)

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
			end
		end

		push!(csX, cs[1, 1, 1])
		push!(csXerr, cs[1, 1, 2])

	end

	if n==1
		p, = ax_all.plot([1/Ns_[n]^1 for l=1:length(csX)], csX, linewidth=0, marker=".", markersize=7.5, markeredgewidth=1.5, color=colors[6], zorder=100, label=latexstring(L"$\mathsf{c}^{\,}_{2N,\chi}$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
	else
		p, = ax_all.plot([1/Ns_[n]^1 for l=1:length(csX)], csX, linewidth=0, marker=".", markersize=7.5, markeredgewidth=1.5, color=colors[6], zorder=100)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
	end


	py"""
	def g(N, a, b):
		import numpy 

		return a*N + b
	"""

	csX = csX[end-4:end]
	chisN = chisN[end-4:end]

	popt, pcov = scipy.optimize.curve_fit(py"g", 1 ./chisN, csX)
	perr = numpy.sqrt(numpy.diag(pcov))

	# push!(csN, numpy.max(csX))
	push!(csN, popt[2])

	o, = ax_chi.plot(1 ./chisN, py"g"(1 ./chisN, popt...), linestyle="--", marker="none", color=colors[5], linewidth=1, zorder=90, markersize=0, label=latexstring(L"$\mathsf{c}(\chi \to \infty) = (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3)))$"))

	ax_chi.scatter(1 ./chisN, csX, marker="x", color=colors[2], zorder=100)

	ax_chi.minorticks_on()
	ax_chi.grid(which="minor", linewidth=0.2, alpha=0.33)
	ax_chi.grid(which="major", linewidth=0.6)
	ax_chi.legend(loc="lower center", ncol=1, columnspacing=1, labelspacing=0.6, bbox_to_anchor=(0.5, 1.01), borderaxespad=0).set_zorder(110)

	ax_chi.set_xlabel(latexstring(L"$\chi$"))
	ax_chi.set_ylabel(latexstring(L"$\mathsf{c}$"))

	plt.draw()

	if savePlot
		py"createFolderIfMissing"(graphFolder)
		fig_chi.savefig(graphFolder * "N=$(Ns_[n])" * ".pdf")
		fig_chi.clf()
	end

	if showPlot
		plt.show()
	end

	plt.close()
end

py"""
def g(N, a, b):
	import numpy 

	return a*N + b
"""

py"""
def r2(y, fit):
	import numpy

	ss_res = numpy.sum((y-fit)**2)
	ss_tot = numpy.sum((y-numpy.mean(y))**2)

	return (1 - ss_res/ss_tot)
"""

Ns_ = Ns_.^1
p, = ax_all.plot(1 ./Ns_, csN, marker="x", markersize=10, markeredgewidth=2, linewidth=0, color=colors[4], zorder=100, label=latexstring(L"$\displaystyle{\lim_{\chi\to\infty}}\mathsf{c}^{\,}_{2N,\chi}$"))
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

csN = csN[3:end]
Ns_ = Ns_[3:end]

popt, pcov = scipy.optimize.curve_fit(py"g", 1 ./Ns_, csN)
perr = numpy.sqrt(numpy.diag(pcov))

r2_ = py"r2"(csN, py"g"(1 ./Ns_, popt...))

o, = ax_all.plot(1 ./Ns_, py"g"(1 ./Ns_, popt...), linestyle="--", marker="none", color=colors[1], linewidth=1.5, zorder=90, markersize=0, label=latexstring(L"$\displaystyle{\lim_{2N\to\infty}\lim_{\chi\to\infty}}\mathsf{c}^{\,}_{2N,\chi} = (%$(round(popt[2]; digits=2)) \pm %$(round(perr[2]; digits=2))),\ R^{2} = %$(round(r2_; digits=3))$"))

ax_all.minorticks_on()
# ax_all.grid(which="minor", linewidth=0.2, alpha=0.33)
# ax_all.grid(which="major", linewidth=0.6)
ax_all.legend(loc="lower right", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.8, handlelength=1.6).set_zorder(110)

ax_all.set_xlabel(latexstring(L"$1/(2N)$"))
ax_all.set_ylabel(latexstring(L"$\mathsf{c}$"))

plt.draw()

if savePlot
	py"createFolderIfMissing"(graphFolder)
	fig_all.savefig(graphFolder * "all" * ".pdf")
	fig_all.clf()
end

if showPlot
	plt.show()
end

plt.close()