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
pandas = pyimport("pandas")
seaborn = pyimport("seaborn")
mpl = pyimport("matplotlib")
scipy = pyimport("scipy")

PBCs = false

if PBCs
	graphFolder = "../GRAPHS/PBC/CORR_DIAMOND/"
	dataFolder = "../DATA/PBC/CORR_DIAMOND/"
else
	graphFolder = "../GRAPHS/OBC/CORR_DIAMOND/"
	dataFolder = "../DATA/OBC/CORR_DIAMOND/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_correlations = true

r = 512
start_ft = 100
start_ft_z = 100
start_ft_z_conn = 5
start_ = 10
end_ft = 1
end_ft_z = 1
end_ft_z_conn = 1
end_ = 1

showPlot = false
savePlot = true

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

folders = ["N=1024_chi=256/"]
for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))

	for i=1:length(dataFilenames)
		
		fig, axes = plt.subplots(2, 1, figsize=(8, 6),  sharex="col")
		fig_, axes_ = plt.subplots(2, 1, figsize=(8, 6),  sharex="col")

		N = Int(vals[i]["N"])
		chi = Int(vals[i]["chi"])

		data = 0
		data = load(dataFolder * folder * computeFolder * dataFilenames[i])
		
		k = length(data["energies"])

		println("··· ", dataFilenames[i])

		corrX = data["corrs"][1, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrY = data["corrs"][1, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		corrZ = data["corrs"][1, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
		
		corrZ_conn = corrZ - data["mags"][1, 3, Int(N/2)-Int(r/2)]*data["mags"][1, 3, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

		p, = axes[1].plot(range(1, r), corrZ, marker="1", color=colors[7], ls="-", lw=0.5, markersize=3, markeredgewidth=1.5, label=latexstring(L"$C^{z}$"), zorder=100)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
		p, = axes[2].plot(range(1, r), corrZ_conn, marker="1", color=colors[7], ls="-", lw=0.5, markersize=3, markeredgewidth=1.5, label=latexstring(L"$C^{z,\mathrm{c}}$"), zorder=100)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		p, = axes_[1].plot(range(1, r), corrX, marker="1", color=colors[1], ls="-", lw=0.5, markersize=3, markeredgewidth=1.5, label=latexstring(L"$C^{x}$"), zorder=100)
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		sites = range(1, r)[start_:end+1-end_]

		corrX = corrX[start_:end+1-end_]
		corrY = corrY[start_:end+1-end_]
		# corrZ = corrZ[start_:end+1-end_]
		
		py"""
		import numpy 

		def f(l, a, b, c):
			return a*(l**(-b)) + c
		def g(l, a, b, c):
			return a*numpy.exp(-l/b) + c
		def oz(l, a, b, c, d):
			return a*numpy.cos(2*numpy.pi*b*l + c) + d
		def oz_conn(l, a, b, c, d, e):
			return a*numpy.cos(2*numpy.pi*b*l + c)*l**(-d) + e  
		"""

		try
			r_ = range(1, r)[start_ft_z:end+1-end_ft_z]
			corrZ_ = corrZ[r_]

			poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r_), numpy.abs(corrZ_), [0.1, 0.3, -1, 0.2])
			perroz = numpy.sqrt(numpy.diag(pcov))
			chi2 = scipy.stats.chisquare(numpy.abs(corrZ_), py"oz"(collect(r_), poptoz...), length(r_)-4)
			poz = chi2[2]

			println(poptoz, perroz)

			axes[1].plot(r_, py"oz"(collect(r_), poptoz...), ls="-", lw=1, color="0", zorder=101, label=latexstring(L"$C^{z}$ fit$^{\,}_{\mathrm{q}}$"))
		catch
			poptoz = 0
			perroz = 0
			poz = 0
		end

		try
			r_ = range(1, r)[start_ft_z_conn:end+1-end_ft_z_conn]
			corrZ_ = corrZ_conn[r_]

			poptozconn, pcov = scipy.optimize.curve_fit(py"oz_conn", collect(r_), corrZ_, [0.1, 0.3, -1, 1, 0])
			perrozconn = numpy.sqrt(numpy.diag(pcov))
			chi2 = scipy.stats.chisquare(numpy.abs(corrZ_), py"oz_conn"(collect(r_), poptozconn...), length(r_)-4)
			pozconn = chi2[2]

			println(poptozconn, perrozconn)

			axes[2].plot(r_, py"oz_conn"(collect(r_), poptozconn...), ls="-", lw=1, color="0", zorder=101, label=latexstring(L"$C^{z,\mathrm{c}}$ fit$^{\,}_{\mathrm{d}}$"))
		catch
			poptozconn = 0
			perrozconn = 0
			pozconn = 0
		end

		try
			poptfx, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrX))
			perrfx = numpy.sqrt(numpy.diag(pcov))
			# chi2 = scipy.stats.chisquare(numpy.abs(corrX), py"f"(collect(sites), poptfx...), length(sites)-4)
			# pfx = chi2[2]

			println("alg x : ", poptfx, perrfx)

			axes_[1].plot(sites, py"f"(collect(sites), poptfx...), ls="-", lw=1, color="0", zorder=101, label=latexstring(L"$C^{x}$ fit$^{\,}_{\mathrm{p}}$"))
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
		catch
			poptgx = 0
			perrgx = 0
			pgx = 0
		end

		try
			poptfy, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrY))
			perrfy = numpy.sqrt(numpy.diag(pcov))
			chi2 = scipy.stats.chisquare(numpy.abs(corrY), py"f"(collect(sites), poptfy...), length(sites)-4)
			pfy = chi2[2]
		catch
			poptfy = 0
			perrfy = 0
			pfy = 0
		end

		try
			poptgy, pcov = scipy.optimize.curve_fit(py"g", collect(sites), numpy.abs(corrY))
			perrgy = numpy.sqrt(numpy.diag(pcov))
			chi2 = scipy.stats.chisquare(numpy.abs(corrY), py"g"(collect(sites), poptgy...), length(sites)-4)
			pgy = chi2[2]
		catch
			poptgy = 0
			perrgy = 0
			pgy = 0
		end
				
		axes[1].minorticks_on()
		# axes[1].grid(which="minor", linewidth=0.2, alpha=0.33)
		# axes[1].grid(which="major", linewidth=0.6)	
		axes[2].minorticks_on()
		# axes[2].grid(which="minor", linewidth=0.2, alpha=0.33)
		# axes[2].grid(which="major", linewidth=0.6)

		axes[1].set_xlim(0, r)
		axes[1].set_ylim(-0.1, 0.21)
		axes[2].set_xlim(0, r)
		axes[2].set_ylim(-0.03, 0.01)

		axes[2].set_xlabel(latexstring(L"$s$"))

		axes_[1].set_ylim(-0.01, 0.02)

		axes[1].legend(loc="lower left", ncol=1, bbox_to_anchor=(0.06, 0.03), borderaxespad=0, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)
		axes[2].legend(loc="lower left", ncol=1, bbox_to_anchor=(0.06, 0.03), borderaxespad=0, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(110)

		axes[1].annotate("(b)", xy=(0.945, 0.08), xycoords="axes fraction", fontsize=16)
		axes[2].annotate("(d)", xy=(0.945, 0.08), xycoords="axes fraction", fontsize=16)

		axins1 = axes[1].inset_axes([0.5, 0.17, 0.4, 0.45])
		axins1.set_zorder(102)
		axins2 = axes[2].inset_axes([0.5, 0.17, 0.4, 0.45])
		axins2.set_zorder(102)

		r_ft = range(1, r)[start_ft:end+1-end_ft]

		ft = numpy.fft.fft(corrZ)
		ftfreq = numpy.fft.fftfreq(length(corrZ))
		sort = numpy.argsort(ftfreq).+1
		p, = axins1.plot(ftfreq[sort], numpy.abs(ft[sort]), marker="x", markersize=1, linewidth=0.5, color=colors[7], zorder=110, label=latexstring(L"FT$^{z}$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		ft = numpy.fft.fft(corrZ_conn)
		ftfreq = numpy.fft.fftfreq(length(corrZ_conn))
		sort = numpy.argsort(ftfreq).+1
		p, = axins2.plot(ftfreq[sort], numpy.abs(ft[sort]), marker="x", markersize=1, linewidth=0.5, color=colors[7], zorder=110, label=latexstring(L"FT$^{z,\mathrm{c}}$"))
		p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

		axins1.set_yscale("log")
		axins2.set_yscale("log")

		axins1.minorticks_on()
		# axins1.grid(which="minor", linewidth=0.2, alpha=0.33)
		# axins1.grid(which="major", linewidth=0.6)	
		axins2.minorticks_on()
		# axins2.grid(which="minor", linewidth=0.2, alpha=0.33)
		# axins2.grid(which="major", linewidth=0.6)

		axins1.legend(loc="upper center", ncol=2, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0.3).set_zorder(110)
		axins2.legend(loc="lower center", ncol=1, labelspacing=0.4, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0.3).set_zorder(110)

		# axins1.set_yticks([0.1, 1])
		axins2.set_yticks([0.1, 1])

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder)
			fig.savefig(graphFolder * folder * "r=$(r)_" * chop(dataFilenames[i], tail=5) * ".pdf")
			fig.clf()
		end
		# if savePlot
		# 	py"createFolderIfMissing"(graphFolder * folder)
		# 	fig_.savefig(graphFolder * folder * "x_r=$(r)_" * chop(dataFilenames[i], tail=5) * ".pdf")
		# 	fig_.clf()
		# end

		if showPlot
			plt.show()
		end

		plt.close()
		
	end
end