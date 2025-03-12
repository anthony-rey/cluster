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
	graphFolder = "../GRAPHS/PBC/LINE_CORR_END_E+DU(1)_SPT/"
	dataFolder = "../DATA/PBC/LINE_CORR_END_E+DU(1)_SPT/"
else
	graphFolder = "../GRAPHS/OBC/LINE_CORR_END_E+DU(1)_SPT/"
	dataFolder = "../DATA/OBC/LINE_CORR_END_E+DU(1)_SPT/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_lines_correlations = true

global r = 96
nE = 4

showPlot = false
savePlot = true

for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))
	
	corrs_ends = numpy.zeros((3, length(lam), length(theta), nE))
	ens = numpy.zeros((length(lam), length(theta), nE))

	for i=1:length(dataFilenames)

		N = Int(vals[i]["N"])
		chi = Int(vals[i]["chi"])

		data = 0
		data = load(dataFolder * folder * computeFolder * dataFilenames[i])
		
		# k = length(data["energies"])
		k = nE

		println("··· ", dataFilenames[i])

		# ------------------------------
		if true
		# ------------------------------

			if plot_lines_correlations

				sort = numpy.argsort(data["energies"]) .+ 1
				data["corrs"] = data["corrs"][sort, :, :, :]
				data["energies"] = data["energies"][sort]

				for j=1:k
					
					corrX = data["ev_ends"][j, 1]
					corrs_ends[1, vals[i]["lam"].==lam, vals[i]["theta"].==theta, j] .= corrX
					corrY = data["ev_ends"][j, 2]
					corrs_ends[2, vals[i]["lam"].==lam, vals[i]["theta"].==theta, j] .= corrY

					corrZ = data["corrs"][j, 3, 1, N]
					corrs_ends[3, vals[i]["lam"].==lam, vals[i]["theta"].==theta, j] .= corrZ

					ens[vals[i]["lam"].==lam, vals[i]["theta"].==theta, j] .= data["energies"][j]

				end
			end
		end
	end

	if plot_lines_correlations

		# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
		colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

		fig = plt.figure(figsize=(4, 4))
		ax1 = plt.subplot2grid((2, 1), (0, 0))
		ax2 = plt.subplot2grid((2, 1), (1, 0))
		# ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
		# plt.subplots_adjust(wspace=0.3)

		for t=1:length(theta)

			p1, = ax2.plot(lam, corrs_ends[3, :, t, 1], color=colors[1], markeredgewidth=1.5, markersize=10*1.2, marker=L"$/$",  ls="-", lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
			p2, = ax2.plot(lam, corrs_ends[3, :, t, 2], color=colors[3], markeredgewidth=1.5, markersize=10*1.2, marker=L"$\setminus$",  ls=":", lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
			p3, = ax1.plot(lam, corrs_ends[3, :, t, 3], color=colors[4], markeredgewidth=1.5, markersize=7*1.2,  marker="o", ls="-", lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
			p4, = ax1.plot(lam, corrs_ends[3, :, t, 4], color=colors[6], markeredgewidth=1.5, markersize=3*1.2,  marker="o", ls=":", lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))
			# ax3.plot(lam, ens[:, t, 1]-ens[:, t, 1], color=colors[1], lw=1, alpha=1, zorder=101)
			# ax3.plot(lam, ens[:, t, 2]-ens[:, t, 1], color=colors[3], lw=1, alpha=1, zorder=101)
			# ax3.plot(lam, ens[:, t, 3]-ens[:, t, 1], color=colors[4], lw=1, alpha=1, zorder=101)
			# ax3.plot(lam, ens[:, t, 4]-ens[:, t, 1], color=colors[6], lw=1, alpha=1, zorder=101)
						
			ax1.minorticks_on()
			ax2.minorticks_on()
			# ax3.minorticks_on()
			
			# ax1.annotate("(a)", xy=(0.91, 0.79), xycoords="axes fraction", fontsize=16)
			# ax3.annotate("(b)", xy=(0.91, 0.89), xycoords="axes fraction", fontsize=16)

			ax2.set_xlabel(latexstring(L"$\lambda$"))
			# ax3.set_xlabel(latexstring(L"$\lambda$"))
			ax1.set_ylabel(latexstring(L"$C^{z}_{1,2N}[\Psi]$"))
			ax2.set_ylabel(latexstring(L"$C^{z}_{1,2N}[\Psi]$"))
			# ax3.set_ylabel(latexstring(L"$E[\Psi]-E^{\,}_{0}$"))
			
			ax1.set_xticklabels([])

			ax2.set_yticks([-1, -0.95])
			ax2.set_yticklabels([L"$-1$", L"$-0.95$"])
			ax1.set_yticks([0.95, 1])
			ax1.set_yticklabels([L"$+0.95$", L"$+1$"])
			# ax3.set_yticks([0, 0.002, 0.004])
			# ax3.set_yticklabels([L"$0$", L"$0.002$", L"$0.004$"])

			ax1.set_xlim(0.84, 1)
			ax2.set_xlim(0.84, 1)
			# ax3.set_xlim(0.84, 1)

			plt.draw()

			ax2.legend(handles=[p1, p2], loc="upper right", ncol=2, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)
			ax1.legend(handles=[p3, p4], loc="lower right", ncol=2, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder)
				fig.savefig(graphFolder * folder * "line_endz" * ".pdf")
				fig.clf()
			end

			if showPlot
				plt.show()
			end

			plt.close()

		end

		# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[1],colors[3],colors[4],colors[6],colors[1],colors[3],colors[4],colors[6]], N=256)
		# PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=["-", ":", "-", ":", "-", ":", "-", ":"], marker=[L"$/$", L"$\setminus$", L"$/$", L"$\setminus$", L"$/$", L"$\setminus$", L"$/$", L"$\setminus$"], markersize=[10, 10, 10, 10, 10, 10, 10, 10], markeredgewidth=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5], color=cmap(numpy.linspace(0, 1, 2*nE))))

		# fig = plt.figure(figsize=(12, 3))
		# ax1 = plt.subplot2grid((2, 2), (0, 0))
		# ax2 = plt.subplot2grid((2, 2), (1, 0))
		# ax3 = plt.subplot2grid((2, 2), (0, 1))
		# ax4 = plt.subplot2grid((2, 2), (1, 1))
		# plt.subplots_adjust(wspace=0.4)

		# for t=1:length(theta)

		# 	p1, = ax1.plot(lam, corrs_ends[1, :, t, 1], color=colors[1], lw=1, alpha=1, zorder=101, label=latexstring(L"$\Psi^{\,}_{0}$"))
		# 	p2, = ax1.plot(lam, corrs_ends[1, :, t, 2], color=colors[3], lw=1, alpha=1, zorder=101, label=latexstring(L"$\Psi^{\,}_{1}$"))
		# 	p3, = ax2.plot(lam, corrs_ends[1, :, t, 3], color=colors[4], lw=1, alpha=1, zorder=101, label=latexstring(L"$\Psi^{\,}_{2}$"))
		# 	p4, = ax2.plot(lam, corrs_ends[1, :, t, 4], color=colors[6], lw=1, alpha=1, zorder=101, label=latexstring(L"$\Psi^{\,}_{3}$"))
		# 	ax3.plot(lam, corrs_ends[2, :, t, 1], color=colors[1], lw=1, alpha=1, zorder=101, label=latexstring(L"$\Psi^{\,}_{0}$"))
		# 	ax3.plot(lam, corrs_ends[2, :, t, 2], color=colors[3], lw=1, alpha=1, zorder=101, label=latexstring(L"$\Psi^{\,}_{1}$"))
		# 	ax4.plot(lam, corrs_ends[2, :, t, 3], color=colors[4], lw=1, alpha=1, zorder=101, label=latexstring(L"$\Psi^{\,}_{2}$"))
		# 	ax4.plot(lam, corrs_ends[2, :, t, 4], color=colors[6], lw=1, alpha=1, zorder=101, label=latexstring(L"$\Psi^{\,}_{3}$"))
						
		# 	ax1.minorticks_on()
		# 	ax2.minorticks_on()
		# 	ax3.minorticks_on()
		# 	ax4.minorticks_on()
			
		# 	ax1.annotate("(c)", xy=(0.91, 0.79), xycoords="axes fraction", fontsize=16)
		# 	ax3.annotate("(d)", xy=(0.91, 0.79), xycoords="axes fraction", fontsize=16)

		# 	ax2.set_xlabel(latexstring(L"$\lambda$"))
		# 	ax4.set_xlabel(latexstring(L"$\lambda$"))
		# 	ax1.set_ylabel(latexstring(L"$C^{\mathrm{SPT},x}_{1,2N}[\Psi]$"))
		# 	ax2.set_ylabel(latexstring(L"$C^{\mathrm{SPT},x}_{1,2N}[\Psi]$"))
		# 	ax3.set_ylabel(latexstring(L"$C^{\mathrm{SPT},y}_{1,2N}[\Psi]$"))
		# 	ax4.set_ylabel(latexstring(L"$C^{\mathrm{SPT},y}_{1,2N}[\Psi]$"))
			
		# 	ax1.set_xticklabels([])
		# 	ax3.set_xticklabels([])

		# 	ax1.set_yticks([0, 0.002])
		# 	ax1.set_yticklabels([L"$0$", L"$+0.002$"])
		# 	ax3.set_yticks([0, 0.002])
		# 	ax3.set_yticklabels([L"$0$", L"$+0.002$"])
		# 	ax2.set_yticks([-0.002, 0])
		# 	ax2.set_yticklabels([L"$-0.002$", L"$0$"])
		# 	ax4.set_yticks([-0.002, 0])
		# 	ax4.set_yticklabels([L"$-0.002$", L"$0$"])

		# 	# ax1.set_xlim(0.84, 1)
		# 	# ax2.set_xlim(0.84, 1)
		# 	# ax3.set_xlim(0.84, 1)

		# 	plt.draw()

		# 	ax3.legend(handles=[p1, p2, p3, p4], loc="upper left", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(1.02, 0.5)).set_zorder(110)

		# 	if savePlot
		# 		py"createFolderIfMissing"(graphFolder * folder)
		# 		fig.savefig(graphFolder * folder * "line_endxy" * ".pdf")
		# 		fig.clf()
		# 	end

		# 	if showPlot
		# 		plt.show()
		# 	end

		# 	plt.close()

		# end
	end
end