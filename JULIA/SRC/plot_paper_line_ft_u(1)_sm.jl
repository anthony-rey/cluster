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
	graphFolder = "../GRAPHS/PBC/LINE_FT_U(1)/"
	dataFolder = "../DATA/PBC/LINE_FT_U(1)/"
else
	graphFolder = "../GRAPHS/OBC/LINE_FT_U(1)/"
	dataFolder = "../DATA/OBC/LINE_FT_U(1)/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_lines_spins_ft = true
plot_lines_correlations_ft = false
plot_lines_spins = true

showPlot = false
savePlot = true

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))

	ft_maxs_spins_z = numpy.zeros((length(lam), length(theta), 2))
	ft_maxs_corrs_z = numpy.zeros((length(lam), length(theta), 2))

	mags = numpy.zeros((length(lam), length(theta), 1))

	for i=1:length(dataFilenames)

		global N = Int(vals[i]["N"])
		chi = Int(vals[i]["chi"])

		data = 0
		data = load(dataFolder * folder * computeFolder * dataFilenames[i])
		
		k = length(data["energies"])

		println("··· ", dataFilenames[i])

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

		# ------------------------------
		if true
		# ------------------------------

			if plot_lines_spins_ft & plot_lines_spins

				mZ = data["mags"][1, 3, :]

				r_ = range(1, length(mZ))[Int(N/2)-Int(r_spins/2)+1:Int(N/2)+Int(r_spins/2)]
				mZ = mZ[r_]

				if plot_lines_spins
					if vals[i]["lam"]==0.24
						mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 0
					else
						mags[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mZ)
					end
				end

				ft = numpy.fft.fft(mZ)
				ftfreq = numpy.fft.fftfreq(length(mZ))
				sort = numpy.argsort(ftfreq).+1
		
				ft = numpy.abs(ft[sort])
				ftfreq = ftfreq[sort]
				cond = (ftfreq.>0.2) .& (ftfreq.<0.45)
				ft = ft[cond]
				ftfreq = ftfreq[cond]
				ind_max_freq = numpy.argmax(ft)+1
				max_freq = ftfreq[ind_max_freq]
				max_amp = ft[ind_max_freq]

				if (max_freq<0.25) | (max_amp<1E-3)
					max_freq = 0
				end

				ft_maxs_spins_z[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= max_freq
				ft_maxs_spins_z[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= max_amp
			end

			if plot_lines_correlations_ft

				corrZ = data["corrs"][1, 3, Int(N/2)-Int(r_corrs/2), Int(N/2)-Int(r_corrs/2)+1:Int(N/2)+Int(r_corrs/2)]

				r_ = range(1, r_corrs)[start_ft_corrs_z:end+1-end_ft_corrs_z]
				corrZ = corrZ[r_]

				ft = numpy.fft.fft(corrZ)
				ftfreq = numpy.fft.fftfreq(length(corrZ))
				sort = numpy.argsort(ftfreq).+1
		
				ft = numpy.abs(ft[sort])
				ftfreq = ftfreq[sort]
				cond = (ftfreq.>0.2) .& (ftfreq.<0.45)
				ft = ft[cond]
				ftfreq = ftfreq[cond]
				ind_max_freq = numpy.argmax(ft)+1
				max_freq = ftfreq[ind_max_freq]
				max_amp = ft[ind_max_freq]

				if max_freq<0.25
					max_freq = 0
				end

				ft_maxs_corrs_z[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= max_freq
				ft_maxs_corrs_z[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= max_amp
			end
		end
	end

	if plot_lines_spins_ft
	
		for t=1:length(theta)
			
			fig, ax = plt.subplots(figsize=(3.4, 3))

			ax.plot(lam, ft_maxs_spins_z[:, t, 1], linewidth=1, marker=".", color=colors[1], markersize=5, zorder=100)

			ax.set_xlabel(latexstring(L"$\lambda$"))
			ax.set_ylabel(latexstring(L"$f^{z}$"))

			ax.set_yticks([0, 0.2, 0.4])

			ax.minorticks_on()
			# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
			# ax.grid(which="major", linewidth=0.6)

			axins = ax.inset_axes([0.45, 0.2, 0.45, 0.45])
			axins.set_zorder(102)
			axins.plot(lam, ft_maxs_spins_z[:, t, 1], linewidth=1, marker=".", color=colors[2], markersize=5, zorder=100)

			axins.dataLim.y0 = ft_maxs_spins_z[end, t, 1]
			axins.dataLim.y1 = numpy.max(ft_maxs_spins_z[:, t, 1])
			axins.autoscale_view()

			axins.minorticks_on()
			# axins.grid(which="minor", linewidth=0.2, alpha=0.33)
			# axins.grid(which="major", linewidth=0.6)

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder)
				fig.savefig(graphFolder * folder * "spins_ft_r=$(r_spins)_theta=$(theta[t])" * ".pdf")
				fig.clf()
			end

			plt.close()
		end
	end

	if plot_lines_spins
	
		for t=1:length(theta)

			# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
			colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

			cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[7],colors[9],colors[1],colors[6],colors[6],colors[6],colors[4],colors[4],colors[4]], N=length(mags[6:end, t, 1]))

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

			fig1, ax1 = plt.subplots(figsize=(5, 3.5))
			fig2, ax2 = plt.subplots(figsize=(3.1, 3.5))
			fig3, ax3 = plt.subplots(figsize=(5, 3.5))

			axins1 = ax1.inset_axes([0.55, 0.27, 0.4, 0.4])
			axins1.set_zorder(102)

			if N == 512
				r_ = lam[1:length(numpy.abs(mags[1:18, t, 1]))]
		
				poptf, pcov = scipy.optimize.curve_fit(py"f", collect(r_), numpy.abs(mags[1:18, t, 1]), [1, 0.25, 0.15, 0])
				perrf = numpy.sqrt(numpy.diag(pcov))
				# chi2 = scipy.stats.chisquare(numpy.abs(mags[1:18, t, 1]), py"f"(collect(r_), poptf...), length(r_)-5)
				# poz = chi2[2]
				println(poptf, perrf)
				# ax1.plot(r_, py"f"(collect(r_), poptf...), ls="-", lw=2, color="0", zorder=101)				

				r_ = lam[6:18].-poptf[2]
		
				poptg, pcov = scipy.optimize.curve_fit(py"g", collect(numpy.log(r_)), numpy.log(numpy.abs(mags[6:18, t, 1])), [0.1, 1])
				perrg = numpy.sqrt(numpy.diag(pcov))
				# chi2 = scipy.stats.chisquare(numpy.abs(mags[1:18, t, 1]), py"f"(collect(r_), poptg...), length(r_)-5)
				# poz = chi2[2]
				println(poptg, perrg)

				r2_ = py"r2"(numpy.log(numpy.abs(mags[6:18, t, 1])), py"g"(collect(numpy.log(r_)), poptg...))
				println(r2_)

				axins1.plot(numpy.log(r_), py"g"(collect(numpy.log(r_)), poptg...), ls="-", lw=2, color=colors[7], zorder=101, label=latexstring(L"$\beta^{\,}_{\mathrm{tri}}\,\ln |\lambda-\lambda^{\,}_{2N,\mathrm{tri}}|+\mathrm{const}$"))

				#= Compute equation of linear relation =#
				popt, pcov = scipy.optimize.curve_fit(py"g", ft_maxs_spins_z[6:end, t, 1], numpy.abs(mags[6:end, t, 1]), [-1, 1])
				perr = numpy.sqrt(numpy.diag(pcov))

				println(popt, perr)
			end

			ax1.plot(lam, ft_maxs_spins_z[:, t, 1], linewidth=0.7, marker=".", color=colors[1], markersize=6, zorder=100, label=latexstring(L"$f^{z}$"))
			p, = ax1.plot(lam, numpy.abs(mags[:, t, 1]), linewidth=0.7, marker="x", color=colors[4], markersize=4, markeredgewidth=1.5, zorder=100, label=latexstring(L"$|m^{z}_{\mathrm{uni}}|$"))	
			p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
			for j=1:length(mags[6:end, t, 1])-1
				ax2.scatter(ft_maxs_spins_z[5+j, t, 1], numpy.abs(mags[5+j, t, 1]), marker="o", color=cmap(j), s=40, zorder=100)
			end
			p, = axins1.plot(numpy.log(lam[6:end].-0.243), numpy.log(numpy.abs(mags[6:end, t, 1])), linewidth=0.7, marker="x", color=colors[4], markersize=6, markeredgewidth=1.5, zorder=100)	
			p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

			# ax3.set_yscale("log")
			# ax3.set_xscale("log")
		
			# cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[7],colors[9],colors[1],colors[6],colors[6],colors[6],colors[4],colors[4],colors[4]], N=256)
			# cb = fig2.colorbar(matplotlib.cm.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=numpy.min(lam[6:end]), vmax=numpy.max(lam[6:end])), cmap=cmap), ax=ax2, drawedges=false)
			# cb.set_label(latexstring(L"$\lambda$"))

			# ax1.annotate("(c)", xy=(0.4, 0.05), xycoords="axes fraction", fontsize=16)
			# ax2.annotate("(d)", xy=(0.4, 0.05), xycoords="axes fraction", fontsize=16)

			ax1.set_xlabel(latexstring(L"$\lambda$"))
			ax2.set_xlabel(latexstring(L"$f^{z}$"))
			ax2.set_ylabel(latexstring(L"$|m^{z}_{\mathrm{uni}}|$"))
			axins1.set_xlabel(latexstring(L"$\ln |\lambda-\lambda^{\,}_{2N,\mathrm{tri}}|$"))
			axins1.set_ylabel(latexstring(L"$\ln |m^{z}_{\mathrm{uni}}|$"))

			ax1.plot(lam[6:end], 100*lam[6:end], ls="-", lw=2, color=colors[7], zorder=101, label=latexstring(L"fit$^{\,}_{\mathrm{log}}$"))
			ax1.legend(loc="lower left", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(0.01, 0.1), ncols=1).set_zorder(200)
			# axins.legend(loc="lower right", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(200)
			# ax3.legend(loc="lower center", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8, borderaxespad=0, bbox_to_anchor=(0.5, 1.01)).set_zorder(200)

			ax2.set_xticks([0.3, 0.35, 0.4])
			axins1.set_yticks([-1.4, -1])

			ax1.set_ylim([-0.02, 0.41])

			ax1.minorticks_on()
			ax2.minorticks_on()
			axins1.minorticks_on()

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder)
				fig1.savefig(graphFolder * folder * "sm_together_r=$(r_spins)_theta=$(theta[t])" * ".pdf")
				fig1.clf()
				fig2.savefig(graphFolder * folder * "sm_against_r=$(r_spins)_theta=$(theta[t])" * ".pdf")
				fig2.clf()
				# fig3.savefig(graphFolder * folder * "sm_log_r=$(r_spins)_theta=$(theta[t])" * ".pdf")
				# fig3.clf()
			end

			plt.close()
		end
	end

	if plot_lines_correlations_ft
	
		for t=1:length(theta)

			fig, ax = plt.subplots(figsize=(3.4, 3))

			ax.plot(lam, ft_maxs_corrs_z[:, t, 1], linewidth=1, marker=".", color=colors[1], markersize=5, zorder=100)

			ax.set_xlabel(latexstring(L"$\lambda$"))
			ax.set_ylabel(latexstring(L"$f^{z}$"))

			ax.set_yticks([0, 0.2, 0.4])

			ax.minorticks_on()
			# ax.grid(which="minor", linewidth=0.2, alpha=0.33)
			# ax.grid(which="major", linewidth=0.6)

			axins = ax.inset_axes([0.45, 0.2, 0.45, 0.45])
			axins.set_zorder(102)
			axins.plot(lam, ft_maxs_corrs_z[:, t, 1], linewidth=1, marker=".", color=colors[2], markersize=5, zorder=100)

			axins.dataLim.y0 = ft_maxs_corrs_z[end, t, 1]
			axins.dataLim.y1 = numpy.max(ft_maxs_corrs_z[:, t, 1])
			axins.autoscale_view()

			axins.minorticks_on()
			# axins.grid(which="minor", linewidth=0.2, alpha=0.33)
			# axins.grid(which="major", linewidth=0.6)

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder)
				fig.savefig(graphFolder * folder * "corrs_ft_r=$(r_corrs)_theta=$(theta[t])" * ".pdf")
				fig.clf()
			end

			plt.close()
		end
	end
end