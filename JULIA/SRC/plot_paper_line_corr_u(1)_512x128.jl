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
	graphFolder = "../GRAPHS/PBC/LINE_CORR_U(1)/"
	dataFolder = "../DATA/PBC/LINE_CORR_U(1)/"
else
	graphFolder = "../GRAPHS/OBC/LINE_CORR_U(1)/"
	dataFolder = "../DATA/OBC/LINE_CORR_U(1)/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_correlations = false
plot_lines_correlations = true

global r = 300
start_ = 6
end_ = 1
nE = 1

showPlot = false
savePlot = true

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

folders = ["N=512_chi=128/"]
fig, ax = plt.subplots(figsize=(6, 3))

for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))
	
	corrs_alg = numpy.zeros((length(lam), length(theta), 2))
	mags = numpy.zeros((length(lam), length(theta), 1))

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

				if N==256
					global r_spins = 200
					global r_corrs = 150
					global start_ft_corrs_z = 20
					global end_ft_corrs_z = 1
				elseif N==512
					global r_spins = 400
					global r_corrs = 400
					global start_ft_corrs_z = 40
					global end_ft_corrs_z = 1
				end

				mZ = data["mags"][1, 3, :]

				r_ = range(1, length(mZ))[Int(N/2)-Int(r_spins/2)+1:Int(N/2)+Int(r_spins/2)]
				mZ = mZ[r_]

				if vals[i]["lam"]==0.24
					mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 0
				else
					mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mZ)
				end


			if plot_correlations | plot_lines_correlations

				for j=1:k
					
					corrX = data["corrs"][j, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
					corrY = data["corrs"][j, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
					corrZ = data["corrs"][j, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

					plt.plot(range(1, r), corrX, marker=4, color=colors[2], ls="-", lw=0.5, markersize=4, label=latexstring(L"$C^{{\,}}_{{x}}$"), zorder=100)
					plt.plot(range(1, r), corrY, marker=5, color=colors[5], ls="-", lw=0.5, markersize=4, label=latexstring(L"$C^{{\,}}_{{y}}$"), zorder=100)
					plt.plot(range(1, r), corrZ, marker=6, color=colors[3], ls="-", lw=0.5, markersize=4, label=latexstring(L"$C^{{\,}}_{{z}}$"), zorder=100)

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
						poptfx, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrX), [0.5, 1.5, 0.01])
						perrfx = numpy.sqrt(numpy.diag(pcov))

						corrs_alg[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= poptfx[2]
						corrs_alg[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= perrfx[2]

						plt.plot(sites, py"f"(collect(sites), poptfx...), markersize=0, ls="--", lw=1, color=colors[2], zorder=102, label=latexstring(L"$|C^{\,}_{x, \, y}| \sim \ell^{- %$(round(poptfx[2]; digits=3)) \pm %$(round(perrfx[2]; digits=3))}$"))
					catch
						poptfx = 0
						perrfx = 0
					end

					try
						poptgx, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrX))
						perrgx = numpy.sqrt(numpy.diag(pcov))

						plt.plot(sites, py"g"(collect(sites), poptgx...), markersize=0, ls=":", lw=1, color=colors[2], zorder=102, label=latexstring(L"$|C^{\,}_{x,\, y}| \sim e^{- \frac{\ell}{%$(round(poptgx[2]; digits=1)) \pm %$(round(perrgx[2]; digits=1))}}$"))
					catch
						poptgx = 0
						perrgx = 0
					end
					
					ax.set_xlim(0, r)
					ax.set_ylim(-0.4, 0.4)
					
					ax.minorticks_on()
					ax.grid(which="minor", linewidth=0.2, alpha=0.33)
					ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="lower right", bbox_to_anchor=(0.97, 0.03), borderaxespad=0, ncols=2, columnspacing=0.8, handletextpad=0.8).set_zorder(101)

					plt.xlabel(latexstring(L"$\ell$"))

					plt.draw()

					if savePlot & plot_correlations
						py"createFolderIfMissing"(graphFolder * folder * "CORRELATIONS/")
						fig.savefig(graphFolder * folder * "CORRELATIONS/" * chop(dataFilenames[i], tail=5) * ".pdf")
					end
					ax.cla()
				end
			end
		end
	end

	py"""
	import numpy 

	def f(s, a, b, c, d):
		return a*numpy.heaviside(s-b, 1)*numpy.abs(s-b)**(c) + d
	def g(s, a, b):
		return a*s + b
	"""
	py"""
	def r2(y, fit):
		import numpy

		ss_res = numpy.sum((y-fit)**2)
		ss_tot = numpy.sum((y-numpy.mean(y))**2)

		return (1 - ss_res/ss_tot)
	"""

	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[7],colors[9],colors[1],colors[6],colors[6],colors[6],colors[4],colors[4],colors[4]], N=length(mags[6:end, 1, 1]))

	fig1, ax1 = plt.subplots(figsize=(6, 3.5))
	fig2, ax2 = plt.subplots(figsize=(3, 3.5))
	fig3, ax3 = plt.subplots(figsize=(5, 3.5))

	#= Compute equation of linear relation =#
	popt, pcov = scipy.optimize.curve_fit(py"g", corrs_alg[6:end, 1, 1], numpy.abs(mags[6:end, 1, 1]), [1, 1])
	perr = numpy.sqrt(numpy.diag(pcov))

	println(popt, perr)

	# r2_ = py"r2"(mags[6:end, 1, 1], py"g"(corrs_alg[6:end, 1, 1], popt...))
	r2_ = py"r2"(corrs_alg[6:end, 1, 1], mags[6:end, 1, 1])
	println(r2_)

	# ax2.plot(corrs_alg[6:end, 1, 1], py"g"(corrs_alg[6:end, 1, 1], popt...), linewidth=0.7, marker=".", color=colors[1], markersize=6, zorder=100, label=latexstring(L"$;$"))

	ax1.plot(lam[6:end], corrs_alg[6:end, 1, 1], linewidth=0.7, marker=".", color=colors[1], markersize=6, zorder=100, label=latexstring(L"$\eta^{x}$"))
	p, = ax1.plot(lam, numpy.abs(mags[:, 1, 1]), linewidth=0.7, marker="x", color=colors[4], markersize=4, markeredgewidth=1.5, zorder=100, label=latexstring(L"$|m^{z}_{\mathrm{uni}}|$"))	
	p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
	for j=1:length(mags[6:end, 1, 1])-1
		ax2.scatter(corrs_alg[5+j, 1, 1], numpy.abs(mags[5+j, 1, 1]), marker="o", color=cmap(j), s=40, zorder=100)
	end

	# ax3.set_yscale("log")
	# ax3.set_xscale("log")

	# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[7],colors[9],colors[1],colors[6],colors[6],colors[6],colors[4],colors[4],colors[4]], N=256)
	# cb = fig2.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=numpy.min(lam[6:end]), vmax=numpy.max(lam[6:end])), cmap=cmap), ax=ax2, drawedges=false)
	# cb.set_label(latexstring(L"$\lambda$"))

	# ax1.annotate("(c)", xy=(0.4, 0.05), xycoords="axes fraction", fontsize=16)
	# ax2.annotate("(d)", xy=(0.4, 0.05), xycoords="axes fraction", fontsize=16)

	ax1.set_xlabel(latexstring(L"$\lambda$"))
	ax2.set_xlabel(latexstring(L"$\eta^{x}$"))
	# ax2.set_ylabel(latexstring(L"$|m^{z}_{\mathrm{uni}}|$"))
	# axins1.set_xlabel(latexstring(L"$\ln |\lambda-\lambda^{\,}_{2N,\mathrm{tri}}|$"))
	# axins1.set_ylabel(latexstring(L"$\ln |m^{z}_{\mathrm{uni}}|$"))

	# ax1.plot(lam[6:end], 100*lam[6:end], ls="-", lw=2, color=colors[7], zorder=101, label=latexstring(L"fit$^{\,}_{\mathrm{log}}$"))
	ax1.legend(loc="lower left", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(0.01, 0.1), ncols=1).set_zorder(200)
	# axins.legend(loc="lower right", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(200)
	# ax3.legend(loc="lower center", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(0.5, 1.01)).set_zorder(200)

	ax2.set_xticks([0, 0.5, 1])
	ax2.set_yticklabels([])
	# axins1.set_yticks([-1.4, -1])

	# ax1.set_ylim([-0.02, 0.41])

	ax1.minorticks_on()
	ax2.minorticks_on()
	# axins1.minorticks_on()

	if savePlot
		py"createFolderIfMissing"(graphFolder * folder)
		fig1.savefig(graphFolder * folder * "sm_together_r=$(r_spins)_theta=$(theta[1])" * ".pdf")
		fig1.clf()
		fig2.savefig(graphFolder * folder * "sm_against_r=$(r_spins)_theta=$(theta[1])" * ".pdf")
		fig2.clf()
		# fig3.savefig(graphFolder * folder * "sm_log_r=$(r_spins)_theta=$(theta[t])" * ".pdf")
		# fig3.clf()
	end

	plt.close()

	if plot_lines_correlations

		fig_line, ax_line = plt.subplots(figsize=(6, 3))

		for t=1:length(theta)

			indSort = range(6, length(lam))
			# indSort = numpy.where(corrs_alg[:, t, 1])[1].+1
			corrs_alg_opt = corrs_alg[indSort, t, 1] 
			corrs_alg_err = corrs_alg[indSort, t, 2]

			plots, caps, errs = ax_line.errorbar(lam[indSort], corrs_alg_opt, yerr=corrs_alg_err, elinewidth=2, linestyle="-", markersize=0, color=colors[7], ecolor=colors[1], linewidth=1.5, barsabove=true, zorder=105)
			for err in errs
				err.set_capstyle("round")
			end
			
			ax_line.minorticks_on()
			# ax_line.grid(which="minor", linewidth=0.2, alpha=0.33)
			# ax_line.grid(which="major", linewidth=0.6)
			
			ax_line.set_xlabel(latexstring(L"$\lambda$"))
			ax_line.set_ylabel(latexstring(L"$\eta^{x}$"))

			# ax_line.set_ylim(0.3, 1.8)

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