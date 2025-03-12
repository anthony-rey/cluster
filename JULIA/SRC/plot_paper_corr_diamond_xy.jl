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
	graphFolder = "../GRAPHS/PBC/CORR_DIAMOND_XY/"
	dataFolder = "../DATA/PBC/CORR_DIAMOND_XY/"
else
	graphFolder = "../GRAPHS/OBC/CORR_DIAMOND_XY/"
	dataFolder = "../DATA/OBC/CORR_DIAMOND_XY/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_correlations = true

global r = 80
start_ft = 20
start_ft_z = 20
start_ = 2
end_ft = 1
end_ft_z = 1
end_ = 1

showPlot = false
savePlot = true

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

# c_ = ["mediumvioletred", "steelblue", "salmon"]
# println([matplotlib.colors.to_hex(c_[i]) for i=1:length(c_)])

folders = ["N=128_chi=128/"]
for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))

	for i=1:length(dataFilenames)
		
		fig, ax = plt.subplots(figsize=(8, 3))

		N = Int(vals[i]["N"])
		chi = Int(vals[i]["chi"])

		data = 0
		data = load(dataFolder * folder * computeFolder * dataFilenames[i])
		
		k = length(data["energies"])

		println("··· ", dataFilenames[i])

		corrX = data["corrs"][1, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrY = data["corrs"][1, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrZ = data["corrs"][1, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

		# corrZ = corrZ - data["mags"][1, 3, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]*data["mags"][1, 3, Int(N/2)-Int(r/2)]

		p, = ax.plot(range(1, r), corrX, marker="3", color=colors[1], ls="-", lw=0.5, markersize=7, markeredgewidth=1.5, label=latexstring(L"$C^{x}$"), zorder=100)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		p, = ax.plot(range(1, r), corrY, marker="4", color=colors[4], ls="-", lw=0.5, markersize=7, markeredgewidth=1.5, label=latexstring(L"$C^{y}$"), zorder=101)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		# p, = axes[2].plot(range(1, r), corrZ, marker="1", color=colors[7], ls="-", lw=0.5, markersize=7, markeredgewidth=1.5, label=latexstring(L"$C^{z}$"), zorder=100)
		# p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		sites = range(1, r)[start_:end+1-end_]

		corrX_ = corrX[start_:end+1-end_]
		corrY_ = corrY[start_:end+1-end_]
		corrZ_ = corrZ[start_:end+1-end_]
		
		py"""
		import numpy 

		def f(l, a, b, c):
			return a*(l**(-b)) + c
		def g(l, a, b, c):
			return a*numpy.exp(-l/b) + c
		def oz(l, a, b, c, d):
			return a*numpy.cos(2*numpy.pi*b*l + c) + d
		"""

		# try
		# 	r_ = range(1, r)[start_ft_z:end+1-end_ft_z]
		# 	corrZ = corrZ[r_]

		# 	poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r_), numpy.abs(corrZ), [0.1, 0.3, -1, 0.2])
		# 	perroz = numpy.sqrt(numpy.diag(pcov))
		# 	chi2 = scipy.stats.chisquare(numpy.abs(corrZ), py"oz"(collect(r_), poptoz...), length(r_)-4)
		# 	poz = chi2[2]
			
		# 	println("Cz", poptoz, perroz)

		# 	axes[2].plot(r_, py"oz"(collect(r_), poptoz...), ls="-", lw=1, color="0", alpha=1, zorder=101, label=latexstring(L"$C^{z}$ fit$^{\,}_{\mathrm{q}}$"))
		# catch
		# 	poptoz = 0
		# 	perroz = 0
		# 	poz = 0
		# end

		try
			poptfx, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrX_))
			perrfx = numpy.sqrt(numpy.diag(pcov))
			chi2 = scipy.stats.chisquare(numpy.abs(corrX_), py"f"(collect(sites), poptfx...), length(sites)-4)
			pfx = chi2[2]

			println("Cx", poptfx, perrfx)

			ax.plot(sites, py"f"(collect(sites), poptfx...), ls="--", lw=1, color=colors[1], zorder=100, label=latexstring(L"$|C^{x}|$ fit$^{\,}_{\mathrm{p}}$"))
		catch
			poptfx = 0
			perrfx = 0
			pfx = 0
		end

		try
			poptgx, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrX_))
			perrgx = numpy.sqrt(numpy.diag(pcov))
			chi2 = scipy.stats.chisquare(numpy.abs(corrX_), py"g"(collect(sites), poptgx...), length(sites)-4)
			pgx = chi2[2]

			ax.plot(sites, py"g"(collect(sites), poptgx...), ls=":", lw=1, color=colors[1], zorder=100, label=latexstring(L"$|C^{x}|$ fit$^{\,}_{\mathrm{e}}$"))
		catch
			poptgx = 0
			perrgx = 0
			pgx = 0
		end

		try
			poptfy, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrY_))
			perrfy = numpy.sqrt(numpy.diag(pcov))
			chi2 = scipy.stats.chisquare(numpy.abs(corrY_), py"f"(collect(sites), poptfy...), length(sites)-4)
			pfy = chi2[2]

			println("Cy", poptfy, perrfy)

			ax.plot(sites, py"f"(collect(sites), poptfy...), ls="--", lw=1, color=colors[4], zorder=101, label=latexstring(L"$|C^{y}|$ fit$^{\,}_{\mathrm{p}}$"))
		catch
			poptfy = 0
			perrfy = 0
			pfy = 0
		end

		try
			poptgy, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrY_))
			perrgy = numpy.sqrt(numpy.diag(pcov))
			chi2 = scipy.stats.chisquare(numpy.abs(corrY_), py"g"(collect(sites), poptgy...), length(sites)-4)
			pgy = chi2[2]

			ax.plot(sites, py"g"(collect(sites), poptgy...), ls=":", lw=1, color=colors[4], zorder=101, label=latexstring(L"$|C^{y}|$ fit$^{\,}_{\mathrm{e}}$"))
		catch
			poptgy = 0
			perrgy = 0
			pgy = 0
		end
				
		ax.minorticks_on()
		# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
		# ax.grid(which="major", linewidth=0.6)	
		# axes[2].minorticks_on()
		# axes[2].grid(which="minor", linewidth=0.2, alpha=0.33)
		# axes[2].grid(which="major", linewidth=0.6)

		ax.set_xlim(0, r)
		ax.set_ylim(-0.1, 0.3)
		# axes[2].set_xlim(0, r)
		# axes[2].set_ylim(-0.1, 0.21)

		if r==72
			ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 72])
			# axes[2].set_xticks([0, 10, 20, 30, 40, 50, 60, 72])
		end

		plt.xlabel(latexstring(L"$s$"))

		ax.legend(loc="upper left", ncol=2, bbox_to_anchor=(0.06, 0.97), borderaxespad=0, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)
		# axes[2].legend(loc="lower left", ncol=1, bbox_to_anchor=(0.06, 0.03), borderaxespad=0, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)

		# if r==72
		# 	ax.annotate("(c)", xy=(0.945, 0.08), xycoords="axes fraction", fontsize=16)
		# 	axes[2].annotate("(d)", xy=(0.945, 0.08), xycoords="axes fraction", fontsize=16)
		# elseif r==80
		# 	ax.annotate("(a)", xy=(0.945, 0.08), xycoords="axes fraction", fontsize=16)
		# 	axes[2].annotate("(b)", xy=(0.945, 0.08), xycoords="axes fraction", fontsize=16)
		# end

		axins1 = ax.inset_axes([0.58, 0.5, 0.4, 0.45])
		axins1.set_zorder(102)
		# axins2 = axes[2].inset_axes([0.5, 0.17, 0.4, 0.45])
		# axins2.set_zorder(102)

		r_ft = range(1, r)[start_ft:end+1-end_ft]

		corrX = corrX[r_ft]
		corrY = corrY[r_ft]

		ft = numpy.fft.fft(corrX)
		ftfreq = numpy.fft.fftfreq(length(corrX))
		sort = numpy.argsort(ftfreq).+1
		axins1.plot(ftfreq[sort], numpy.abs(ft[sort]), marker="x", markersize=3, linewidth=0.5, color=colors[1], zorder=110, label=latexstring(L"FT$^{x}$"))

		ft = numpy.fft.fft(corrY)
		ftfreq = numpy.fft.fftfreq(length(corrY))
		sort = numpy.argsort(ftfreq).+1
		axins1.plot(ftfreq[sort], numpy.abs(ft[sort]), marker="x", markersize=3, linewidth=0.5, color=colors[4], zorder=110, label=latexstring(L"FT$^{y}$"))

		# ft = numpy.fft.fft(corrZ)
		# ftfreq = numpy.fft.fftfreq(length(corrZ))
		# sort = numpy.argsort(ftfreq).+1
		# axins2.plot(ftfreq[sort], numpy.abs(ft[sort]), marker="x", markersize=3, linewidth=0.5, color=colors[7], zorder=110, label=latexstring(L"FT$^{z}$"))

		axins1.set_yscale("log")
		# axins2.set_yscale("log")

		axins1.minorticks_on()
		# axins1.grid(which="minor", linewidth=0.2, alpha=0.33)
		# axins1.grid(which="major", linewidth=0.6)	
		# axins2.minorticks_on()
		# axins2.grid(which="minor", linewidth=0.2, alpha=0.33)
		# axins2.grid(which="major", linewidth=0.6)

		axins1.legend(loc="upper center", ncol=2, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0.3).set_zorder(110)
		# axins2.legend(loc="upper center", ncol=1, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0.3).set_zorder(110)

		# axins1.set_yticks([0.1, 1])
		# axins2.set_yticks([0.01, 1])

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder)
			fig.savefig(graphFolder * folder * "r=$(r)_" * chop(dataFilenames[i], tail=5) * ".pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()
		
	end
end