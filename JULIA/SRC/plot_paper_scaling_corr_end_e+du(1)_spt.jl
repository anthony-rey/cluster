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
PyPlot.rc("axes.formatter", limits=[-3, 3])
PyPlot.rc("axes.formatter", offset_threshold=3)

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
	graphFolder = "../GRAPHS/PBC/SCALING_CORR_END_E+DU(1)_SPT/"
	dataFolder = "../DATA/PBC/SCALING_CORR_END_E+DU(1)_SPT/"
else
	graphFolder = "../GRAPHS/OBC/SCALING_CORR_END_E+DU(1)_SPT/"
	dataFolder = "../DATA/OBC/SCALING_CORR_END_E+DU(1)_SPT/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

global r = 96
nE = 4

showPlot = false
savePlot = true

Ns = []
corrs_ends_Ns = []

for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))
	push!(Ns, numpy.unique([Int(vals[i]["N"]) for i=1:length(vals)])[1])
	
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

	push!(corrs_ends_Ns, corrs_ends)

end

indSort = numpy.argsort(Ns).+1

Ns = numpy.array(Ns)[indSort]
corrs_ends_Ns = numpy.array(corrs_ends_Ns)[indSort, :, :, :, :]

start = 1
Ns = Ns[start:end]
corrs_ends_Ns = corrs_ends_Ns[start:end, :, :, :, :]

py"""
def f(N, a, b):
	return a/N + b
"""

py"""
def r2(y, fit):
	import numpy

	ss_res = numpy.sum((y-fit)**2)
	ss_tot = numpy.sum((y-numpy.mean(y))**2)

	return (1 - ss_res/ss_tot)
"""

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=repeat([colors[1],colors[3],colors[4],colors[6]], 3), N=256)
PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=repeat(["-", ":", "-", ":"], 3), marker=repeat([L"$/$", L"$\setminus$", "o", "o"], 3), markersize=repeat([10, 10, 7, 3] .*1.2, 3), markeredgewidth=repeat([1.5, 1.5, 1.5, 1.5], 3), color=cmap(numpy.linspace(0, 1, 3*nE))))

fig, ax = plt.subplots(figsize=(6, 4))

for l=1:length(lam)	
	for t=1:length(theta)

		popt, pcov = scipy.optimize.curve_fit(py"f", Ns, corrs_ends_Ns[:, 1, l, t, 1]./corrs_ends_Ns[:, 3, l, t, 1])
		perr = numpy.sqrt(numpy.diag(pcov))
		println(popt, perr)

		r2_ = py"r2"(corrs_ends_Ns[:, 1, l, t, 1]./corrs_ends_Ns[:, 3, l, t, 1], py"f"(Ns, popt...))
		println(r2_)

		p1, = ax.plot(1 ./Ns, corrs_ends_Ns[:, 1, l, t, 1]./corrs_ends_Ns[:, 3, l, t, 1], lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
		p2, = ax.plot(1 ./Ns, corrs_ends_Ns[:, 1, l, t, 2]./corrs_ends_Ns[:, 3, l, t, 2], lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
		p3, = ax.plot(1 ./Ns, corrs_ends_Ns[:, 1, l, t, 3]./corrs_ends_Ns[:, 3, l, t, 3], lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
		p4, = ax.plot(1 ./Ns, corrs_ends_Ns[:, 1, l, t, 4]./corrs_ends_Ns[:, 3, l, t, 4], lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))
		
		ax.minorticks_on()		
					
		ax.set_xlabel(latexstring(L"$1/(2N)$"))
		ax.set_ylabel(latexstring(L"$C^{\mathrm{SPT},x}_{1,2N}[\Psi]/C^{z}_{1,2N}[\Psi]$"))

		ax.set_xlim(0, 0.08)
		ax.set_ylim(-0.001, 0)

		ax.set_yticks([-0.0008, -0.0004, 0])

		# ax.annotate(L"$R^{2} = %$(round(r2_; digits=4))$", xy=(0.9, 0.79), xycoords="axes fraction", fontsize=16)

		axins = ax.inset_axes([0.16, 0.16, 0.55, 0.4])
		axins.set_zorder(102)

		axins.plot(1 ./Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 1]), lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
		axins.plot(1 ./Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 2]), lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
		axins.plot(1 ./Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 3]), lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
		axins.plot(1 ./Ns, numpy.abs(corrs_ends_Ns[:, 3, l, t, 4]), lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))

		axins.minorticks_on()	

		axins.set_ylabel(latexstring(L"$\big|C^{z}_{1,2N}[\Psi]\big|$"))

		axins.set_xlim(0, 0.08)
		axins.set_ylim(0.9953, 0.9956)

		# # Define a custom formatter that subtracts the fixed offset
		# py"""
		# def custom_formatter(val, pos):
		# 	y_offset = 0.995
		# 	return f"{val - y_offset:.6f}"
		# """

		# # Apply formatter to y-axis
		# axins.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(py"custom_formatter"))

		# formatter = matplotlib.ticker.ScalarFormatter(useOffset=true, useMathText=false)
		# formatter.set_scientific(false)  # Disable scientific notation
		# # formatter.set_powerlimits((-3, -2))
		# axins.yaxis.set_major_formatter(formatter)
		# axins.yaxis.get_offset_text().set_text("0.995") 
		# axins.ticklabel_format(axis="y", style="plain", useOffset=0.995)
		# axins.yaxis.offsetText.set_visible(false)
		# axins.text(x = 0.0, y = 1.01, s = "0.995", transform=axins.transAxes)

		plt.draw()

		ax.legend(handles=[p1, p2, p3, p4], loc="lower right", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)

		if savePlot
			py"createFolderIfMissing"(graphFolder)
			fig.savefig(graphFolder * "line_endxy" * ".pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

	end
end






# # colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
# colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=repeat([colors[1],colors[3],colors[4],colors[6]], 3), N=256)
# PyPlot.rc("axes", prop_cycle=cycler.cycler(linestyle=repeat(["-", ":", "-", ":"], 3), marker=repeat([L"$/$", L"$\setminus$", "o", "o"], 3), markersize=repeat([10, 10, 6, 3], 3), markeredgewidth=repeat([1.5, 1.5, 1.5, 1.5], 3), color=cmap(numpy.linspace(0, 1, 3*nE))))

# fig = plt.figure(figsize=(12, 6))
# ax1 = plt.subplot2grid((3, 2), (0, 0))
# ax2 = plt.subplot2grid((3, 2), (1, 0))
# ax3 = plt.subplot2grid((3, 2), (2, 0))
# ax4 = plt.subplot2grid((3, 2), (0, 1))
# ax5 = plt.subplot2grid((3, 2), (1, 1))
# ax6 = plt.subplot2grid((3, 2), (2, 1))
# plt.subplots_adjust(wspace=0.4)

# for l=1:length(lam)	
# 	for t=1:length(theta)

# 		end_ = 6

# 		p1, = ax1.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 1, l, t, 1], lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
# 		p2, = ax1.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 1, l, t, 2], lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
# 		p3, = ax1.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 1, l, t, 3], lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
# 		p4, = ax1.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 1, l, t, 4], lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))
		
# 		ax2.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 2, l, t, 1], lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
# 		ax2.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 2, l, t, 2], lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
# 		ax2.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 2, l, t, 3], lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
# 		ax2.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 2, l, t, 4], lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))
				
# 		ax3.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 3, l, t, 1], lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
# 		ax3.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 3, l, t, 2], lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
# 		ax3.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 3, l, t, 3], lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
# 		ax3.plot(Ns[1:end_], corrs_ends_Ns[1:end_, 3, l, t, 4], lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))
		
# 		start_ = 7

# 		popt, pcov = scipy.optimize.curve_fit(py"f", Ns[start_:end], corrs_ends_Ns[start_:end, 1, l, t, 1])
# 		perr = numpy.sqrt(numpy.diag(pcov))
# 		println(popt, perr)
# 		popt, pcov = scipy.optimize.curve_fit(py"f", Ns[start_:end], corrs_ends_Ns[start_:end, 2, l, t, 1])
# 		perr = numpy.sqrt(numpy.diag(pcov))
# 		println(popt, perr)
# 		popt, pcov = scipy.optimize.curve_fit(py"f", Ns[start_:end], corrs_ends_Ns[start_:end, 3, l, t, 1])
# 		perr = numpy.sqrt(numpy.diag(pcov))
# 		println(popt, perr)
# 		popt, pcov = scipy.optimize.curve_fit(py"f", Ns[start_:end], corrs_ends_Ns[start_:end, 1, l, t, 3])
# 		perr = numpy.sqrt(numpy.diag(pcov))
# 		println(popt, perr)
# 		popt, pcov = scipy.optimize.curve_fit(py"f", Ns[start_:end], corrs_ends_Ns[start_:end, 2, l, t, 3])
# 		perr = numpy.sqrt(numpy.diag(pcov))
# 		println(popt, perr)
# 		popt, pcov = scipy.optimize.curve_fit(py"f", Ns[start_:end], corrs_ends_Ns[start_:end, 3, l, t, 3])
# 		perr = numpy.sqrt(numpy.diag(pcov))
# 		println(popt, perr)

# 		ax4.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 1, l, t, 1], lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
# 		ax4.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 1, l, t, 2], lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
# 		ax4.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 1, l, t, 3], lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
# 		ax4.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 1, l, t, 4], lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))
		
# 		ax5.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 2, l, t, 1], lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
# 		ax5.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 2, l, t, 2], lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
# 		ax5.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 2, l, t, 3], lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
# 		ax5.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 2, l, t, 4], lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))
				
# 		ax6.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 3, l, t, 1] .+ 0.995, lw=1, alpha=1, zorder=103, label=latexstring(L"$\Psi^{\,}_{0}$"))
# 		ax6.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 3, l, t, 2] .+ 0.995, lw=1, alpha=1, zorder=104, label=latexstring(L"$\Psi^{\,}_{1}$"))
# 		ax6.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 3, l, t, 3] .- 0.995, lw=1, alpha=1, zorder=102, label=latexstring(L"$\Psi^{\,}_{2}$"))
# 		ax6.plot(1 ./Ns[start_:end], corrs_ends_Ns[start_:end, 3, l, t, 4] .- 0.995, lw=1, alpha=1, zorder=105, label=latexstring(L"$\Psi^{\,}_{3}$"))
		
					
# 		ax1.minorticks_on()
# 		ax2.minorticks_on()
# 		ax3.minorticks_on()
# 		ax4.minorticks_on()
# 		ax5.minorticks_on()
# 		ax6.minorticks_on()

# 		ax1.set_xticklabels([])
# 		ax2.set_xticklabels([])
# 		ax4.set_xticklabels([])
# 		ax5.set_xticklabels([])
		
# 		ax3.set_xlabel(latexstring(L"$2N$"))
# 		ax6.set_xlabel(latexstring(L"$1/(2N)$"))
# 		ax1.set_ylabel(latexstring(L"$C^{\mathrm{SPT},x}_{1,2N}[\Psi]$"))
# 		ax2.set_ylabel(latexstring(L"$C^{\mathrm{SPT},y}_{1,2N}[\Psi]$"))
# 		ax3.set_ylabel(latexstring(L"$C^{z}_{1,2N}[\Psi]$"))
		
# 		# ax1.annotate("(c)", xy=(0.91, 0.79), xycoords="axes fraction", fontsize=16)
# 		# ax3.annotate("(d)", xy=(0.91, 0.79), xycoords="axes fraction", fontsize=16)

# 		plt.draw()

# 		ax2.legend(handles=[p1, p2, p3, p4], loc="upper left", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(1.02, 0.5)).set_zorder(110)

# 		if savePlot
# 			py"createFolderIfMissing"(graphFolder)
# 			fig.savefig(graphFolder * "line_endxy" * ".pdf")
# 			fig.clf()
# 		end

# 		if showPlot
# 			plt.show()
# 		end

# 		plt.close()

# 	end
# end