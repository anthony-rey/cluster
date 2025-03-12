using FileIO
using PyCall
using PyPlot

plt.close("all")

PyPlot.rc("text", usetex=true)
PyPlot.rc("font", family="serif")
PyPlot.rc("font", size=14)
PyPlot.rc("legend", fontsize=12)
PyPlot.rc("legend", fancybox=true)
PyPlot.rc("savefig", dpi=250)
PyPlot.rc("savefig", bbox="tight")
PyPlot.rc("savefig", format="png")

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
scipy = pyimport("scipy")
seaborn = pyimport("seaborn")

PBCs = false

if PBCs
	graphFolder = "../GRAPHS/PBC/_/"
	dataFolder = "../DATA/PBC/_/"
else
	graphFolder = "../GRAPHS/OBC/_/"
	dataFolder = "../DATA/OBC/_/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_degeneracies = false
plot_scaling_gaps = false
plot_lines_gaps = false
plot_scaling_en = false
plot_bulkenergies = false

plot_entropies = false
plot_lines_c = false
plot_entanglement = false

plot_spins = true
plot_spins_spt = false
plot_spins_ft = false
plot_lines_spins_ft = false
plot_lines_mag = false

plot_correlations = false
plot_correlations_ends = true
plot_correlations_ft = false
plot_lines_correlations_ft = false

plot_spt_correlations = false

plot_grid_evs = false

r = 6
start_ = 1
start_z = 1
end_ = 1
win = 10
epsDegen = 0.3
nE = 4

savePlot = true

fig, ax = plt.subplots(figsize=(4, 4))

Ns = []
diffsNs = []
ENs = []

# folders = ["N=32_chi=128/"]
for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	global lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	global theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))
	push!(Ns, numpy.unique([Int(vals[i]["N"]) for i=1:length(vals)])[1])

	Es = numpy.zeros((length(lam), length(theta), nE))
	diffs = numpy.zeros((length(lam), length(theta), nE-1))
	degens = numpy.zeros((length(lam), length(theta), 1))
	cs = numpy.zeros((nE, length(lam), length(theta), 2))
	mags = numpy.zeros((nE, length(lam), length(theta), 6))
	ft_maxs_corrs_z = numpy.zeros((nE, length(lam), length(theta), 2))
	ft_maxs_spins_z = numpy.zeros((nE, length(lam), length(theta), 2))
	evs_2 = numpy.zeros((nE, length(lam), length(theta), 3))
	evs_4 = numpy.zeros((nE, length(lam), length(theta), 3))
	evs_6 = numpy.zeros((nE, length(lam), length(theta), 3))
	evs_8 = numpy.zeros((nE, length(lam), length(theta), 3))
	evs_10 = numpy.zeros((nE, length(lam), length(theta), 3))

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

			if plot_degeneracies | plot_scaling_gaps | plot_lines_gaps | plot_scaling_en

				E = numpy.sort(data["energies"])[1:nE]

				# EB = data["bulkEnergies"]			
				# E = [numpy.min(EB[l]) for l=1:nE]
				# E = numpy.sort(E)		

				Es[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = numpy.sort([E[e] for e=1:nE])
				diffs[vals[i]["lam"].==lam, vals[i]["theta"].==theta, :] = [E[e+1]-E[e] for e=1:nE-1]
				degens[vals[i]["lam"].==lam, vals[i]["theta"].==theta] .= length(E[[E[e]-E[1]<epsDegen for e=1:nE]])
			
			end

			if plot_entanglement

				for j=1:k

					b = Int(N//2)
					# b = N-2
					num = min(20, Int(length(data["entanglement"][j, b])))

					ax.scatter(range(1, num), -2 .* numpy.log(data["entanglement"][j, b])[1:num], marker=".", s=5, color="mediumvioletred", zorder=100)
					# println(-2 .* numpy.log(data["entanglement"][j, b]))

					ax.minorticks_on()
					ax.grid(which="minor", linewidth=0.2)
					ax.grid(which="major", linewidth=0.6)

					plt.xlabel(latexstring(L"i"))
					plt.ylabel(latexstring(L"$s^{\,}_{i}$"))

					plt.draw()

					if savePlot
						py"createFolderIfMissing"(graphFolder * folder * "entanglement/")
						fig.savefig(graphFolder * folder * "entanglement/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".png")
						ax.cla()
					end
				end
			end

			if plot_bulkenergies

				for j=1:k

					ax.plot(range(1, length(data["bulkEnergies"][j])), data["bulkEnergies"][j], marker="none", color="mediumvioletred", zorder=100)

					ax.minorticks_on()
					ax.grid(which="minor", linewidth=0.2)
					ax.grid(which="major", linewidth=0.6)

					plt.xlabel(latexstring(L"iteration"))
					plt.ylabel(latexstring(L"$E$"))

					plt.draw()

					if savePlot
						py"createFolderIfMissing"(graphFolder * folder * "bulkEnergies/")
						fig.savefig(graphFolder * folder * "bulkEnergies/" * chop(dataFilenames[i], tail=5) * "_E%$(j-1)" * ".png")
						ax.cla()
					end
				end
			end

			if plot_entropies | plot_lines_c

				Ss = data["entropies"]

				for j=1:k

					sites = range(1, size(Ss, 2))

					sites = sites[Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]
					S = Ss[j, Int(N/2)-Int(win/2):Int(N/2)+Int(win/2)]

					py"""
					def f(l, c, const):
						import numpy

						N = int(l[len(l)//2]*2)

						return c/3 * numpy.log(N/numpy.pi * numpy.sin(numpy.pi*l/N)) + const
						# return c/6 * numpy.log(2*N/numpy.pi * numpy.sin(numpy.pi*l/N)) + const
					"""

					popt, pcov = scipy.optimize.curve_fit(py"f", collect(sites), S)
					perr = numpy.sqrt(numpy.diag(pcov))

					cs[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= popt[1]
					cs[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= perr[1]

					if PBCs
						label=latexstring(L"$S = \frac{(%$(round(popt[1]; digits=3)) \pm %$(round(perr[1]; digits=3)))}{3} \ln\left[\frac{N}{\pi}\sin\frac{\pi \ell}{N} \right] + (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3)))$")
					else	
						label=latexstring(L"$S = \frac{(%$(round(popt[1]; digits=3)) \pm %$(round(perr[1]; digits=3)))}{6} \ln\left[\frac{2N}{\pi}\sin\frac{\pi \ell}{N} \right] + (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3)))$")
					end

					if plot_entropies

						ax.scatter(sites, S, marker="x", color="mediumvioletred", zorder=100)
						ax.plot(sites, py"f"(collect(sites), popt...), linestyle=":", color="steelblue", label=label, zorder=100)

						ax.minorticks_on()
						ax.grid(which="minor", linewidth=0.2)
						ax.grid(which="major", linewidth=0.6)
						ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), borderaxespad=0).set_zorder(101)

						plt.xlabel(latexstring(L"$\ell$"))
						plt.ylabel(latexstring(L"$S$"))

						plt.draw()
					
					end

					if savePlot
						py"createFolderIfMissing"(graphFolder * folder * "entropies/")
						fig.savefig(graphFolder * folder * "entropies/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".png")
						ax.cla()
					end
				end
			end

			if plot_spins | plot_lines_mag | plot_spins_ft | plot_lines_spins_ft

				for j=1:k

					mX = data["mags"][j, 1, :]
					mY = data["mags"][j, 2, :]
					mZ = data["mags"][j, 3, :]

					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(mX)
					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean([mX[l]*(-1)^l for l=1:length(mX)])
					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(mY)
					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 4] .= numpy.mean([mY[l]*(-1)^l for l=1:length(mY)])
					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 5] .= numpy.mean(mZ)
					mags[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 6] .= numpy.mean([mZ[l]*(-1)^l for l=1:length(mZ)])

					if plot_spins
						ax.plot(range(1, length(mX)), mX, label=latexstring(L"$\langle \sigma^x_\ell \rangle$"), zorder=100)
						ax.plot(range(1, length(mY)), mY, label=latexstring(L"$\langle \sigma^y_\ell \rangle$"), zorder=100)
						ax.plot(range(1, length(mZ)), mZ, label=latexstring(L"$\langle \sigma^z_\ell \rangle$"), zorder=100)
						
						ax.minorticks_on()
						ax.grid(which="minor", linewidth=0.2)
						ax.grid(which="major", linewidth=0.6)
						ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

						plt.xlabel(latexstring(L"$\ell$"))

						plt.draw()
					end

					if savePlot & plot_spins
						py"createFolderIfMissing"(graphFolder * folder * "spins/")
						fig.savefig(graphFolder * folder * "spins/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".png")
						ax.cla()
					end

					if plot_spins_ft | plot_lines_spins_ft

						mX = mX[start_:end+1-end_]
						mY = mY[start_:end+1-end_]
						mZ = mZ[start_:end+1-end_]

						ft = numpy.fft.fft(mX)
						ftfreq = numpy.fft.fftfreq(length(mX))
						sort = numpy.argsort(ftfreq).+1
						plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$x$"))

						ft = numpy.fft.fft(mY)
						ftfreq = numpy.fft.fftfreq(length(mY))
						sort = numpy.argsort(ftfreq).+1
						plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$y$"))

						ft = numpy.fft.fft(mZ)
						ftfreq = numpy.fft.fftfreq(length(mZ))
						sort = numpy.argsort(ftfreq).+1
						plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$z$"))
						
						if plot_lines_spins_ft

							ft = numpy.abs(ft[sort])
							ftfreq = ftfreq[sort]
							cond = (ftfreq.>0.2) .& (ftfreq.<0.45)
							ft = ft[cond]
							ftfreq = ftfreq[cond]
							ind_max_freq = numpy.argmax(ft)+1
							max_freq = ftfreq[ind_max_freq]
							max_amp = ft[ind_max_freq]

							if (max_freq<0.25) | (max_freq>0.4)
								max_freq = 0
							end

							ft_maxs_spins_z[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= max_freq
							ft_maxs_spins_z[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= max_amp

						end

						ax.minorticks_on()
						ax.grid(which="minor", linewidth=0.2)
						ax.grid(which="major", linewidth=0.6)
						ax.legend(loc="center right").set_zorder(101)

						plt.xlabel("frequency")
						plt.ylabel("amplitude")

						ax.set_yscale("log")

						plt.draw()

						if savePlot
							py"createFolderIfMissing"(graphFolder * folder * "spins_ft/")
							fig.savefig(graphFolder * folder * "spins_ft/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".png")
							ax.cla()
						end
					end
				end
			end

			if plot_spins_spt

				for j=1:k

					mX = data["opSPT"][j, 1, :]
					mY = data["opSPT"][j, 2, :]
					mZ = data["opSPT"][j, 3, :]

					ax.plot(range(1, length(mX)), mX, label=latexstring(L"$\langle \sigma^x_\ell \rangle$"), zorder=100)
					ax.plot(range(1, length(mY)), mY, label=latexstring(L"$\langle \sigma^y_\ell \rangle$"), zorder=100)
					ax.plot(range(1, length(mZ)), mZ, label=latexstring(L"$\langle \sigma^z_\ell \rangle$"), zorder=100)
					
					ax.minorticks_on()
					ax.grid(which="minor", linewidth=0.2)
					ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

					plt.xlabel(latexstring(L"$\ell$"))

					plt.draw()

					if savePlot & plot_spins_spt
						py"createFolderIfMissing"(graphFolder * folder * "spinsSPT/")
						fig.savefig(graphFolder * folder * "spinsSPT/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".png")
						ax.cla()
					end
				end
			end

			if plot_correlations | plot_correlations_ft | plot_lines_correlations_ft | plot_correlations_ends

				for j=1:k

					corrX = data["corrs"][j, 1, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
					corrY = data["corrs"][j, 2, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]
					corrZ = data["corrs"][j, 3, Int(N/2)-Int(r/2), Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]

					println("X_1, X_2N ", data["corrs"][j, 1, 1, N])
					println("Y_1, Y_2N ", data["corrs"][j, 2, 1, N])
					println("Z_1, Z_2N ", data["corrs"][j, 3, 1, N])
					println("X_1 Z_2, Z_2N-1 X_2N ", data["ev_ends"][j, 1])
					println("Y_1 Z_2, Z_2N-1 Y_2N ", data["ev_ends"][j, 2])
					
					# corrX = corrX - data["mags"][j, 1, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]*data["mags"][j, 1, Int(N/2)-Int(r/2)]
					# corrY = corrY - data["mags"][j, 2, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)]*data["mags"][j, 2, Int(N/2)-Int(r/2)]
					# corrZ = corrZ - data["mags"][j, 3, Int(N/2)-Int(r/2)+1:Int(N/2)+Int(r/2)].*data["mags"][j, 3, Int(N/2)-Int(r/2)]

					plt.plot(range(1, r), corrX, marker=4, color="mediumvioletred", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{{\,}}_{{x}}$"), zorder=100)
					plt.plot(range(1, r), corrY, marker=5, color="steelblue", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{{\,}}_{{y}}$"), zorder=100)
					plt.plot(range(1, r), corrZ, marker=6, color="salmon", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{{\,}}_{{z}}$"), zorder=100)

					sites = range(1, r)[start_:end+1-end_]

					corrX = corrX[start_:end+1-end_]
					corrY = corrY[start_:end+1-end_]
					
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

						plt.plot(sites, py"f"(collect(sites), poptfx...), ls="--", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\,}_{x}| = %$(round(poptfx[1]; digits=2)) \ell^{- %$(round(poptfx[2]; digits=2))} + %$(round(poptfx[3]; digits=2)), \ p = %$(round(pfx[1]; digits=3))$"))
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

						plt.plot(sites, py"g"(collect(sites), poptgx...), ls=":", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\,}_{x}| = %$(round(poptgx[1]; digits=2)) e^{- \frac{\ell}{%$(round(poptgx[2]; digits=2))}} + %$(round(poptgx[3]; digits=2)), \ p = %$(round(pgx[1]; digits=3))$"))
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

						plt.plot(sites, py"f"(collect(sites), poptfy...), ls="--", lw=0.7, color="steelblue", zorder=102, label=latexstring(L"$|C^{\,}_{y}| = %$(round(poptfy[1]; digits=2)) \ell^{- %$(round(poptfy[2]; digits=2))} + %$(round(poptfy[3]; digits=2)), \ p = %$(round(pfy[1]; digits=3))$"))
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

						plt.plot(sites, py"g"(collect(sites), poptgy...), ls=":", lw=0.7, color="steelblue", zorder=102, label=latexstring(L"$|C^{\,}_{y}| = %$(round(poptgy[1]; digits=2)) e^{- \frac{\ell}{%$(round(poptgy[2]; digits=2))}} + %$(round(poptgy[3]; digits=2)), \ p = %$(round(pgy[1]; digits=3))$"))
					catch
						poptgy = 0
						perrgy = 0
						pgy = 0
					end

					try
						sites = range(1, r)[start_z:end+1-end_]
						corrZ = corrZ[sites]

						poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(sites), numpy.abs(corrZ), [0.1, 0.3, -1, 0.2])
						perroz = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrZ), py"oz"(collect(sites), poptoz...), length(sites)-4)
						poz = chi2[2]

						plt.plot(sites, py"oz"(collect(sites), poptoz...), ls="-", lw=0.7, color="k", zorder=102, label=latexstring(L"$|C^{\,}_{z}| = %$(round(poptoz[1]; digits=2)) \cos(2 \pi \cdot %$(round(poptoz[2]; digits=4)) + %$(round(poptoz[3]; digits=2))) + %$(round(poptoz[4]; digits=2)), \ p = %$(round(poz[1]; digits=3))$"))
					catch
						poptoz = 0
						perroz = 0
						poz = 0
					end
							
					ax.minorticks_on()
					ax.grid(which="minor", linewidth=0.2)
					ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

					plt.xlabel(latexstring(L"$\ell$"))

					plt.draw()

					if savePlot
						py"createFolderIfMissing"(graphFolder * folder * "correlations/")
						fig.savefig(graphFolder * folder * "correlations/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".png")
						ax.cla()
					end

					if plot_correlations_ft | plot_lines_correlations_ft

						ft = numpy.fft.fft(corrX)
						ftfreq = numpy.fft.fftfreq(length(corrX))
						sort = numpy.argsort(ftfreq).+1
						plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$x$"))

						ft = numpy.fft.fft(corrY)
						ftfreq = numpy.fft.fftfreq(length(corrY))
						sort = numpy.argsort(ftfreq).+1
						plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$y$"))

						ft = numpy.fft.fft(corrZ)
						ftfreq = numpy.fft.fftfreq(length(corrZ))
						sort = numpy.argsort(ftfreq).+1
						plt.plot(ftfreq[sort], numpy.abs(ft[sort]), label=latexstring(L"$z$"))
						
						if plot_lines_correlations_ft

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

							ft_maxs_corrs_z[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= max_freq
							ft_maxs_corrs_z[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= max_amp

						end

						ax.minorticks_on()
						ax.grid(which="minor", linewidth=0.2)
						ax.grid(which="major", linewidth=0.6)
						ax.legend(loc="center right").set_zorder(101)

						plt.xlabel("frequency")
						plt.ylabel("amplitude")

						ax.set_yscale("log")

						plt.draw()

						if savePlot
							py"createFolderIfMissing"(graphFolder * folder * "correlations_ft/")
							fig.savefig(graphFolder * folder * "correlations_ft/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".png")
							ax.cla()
						end
					end
				end
			end

			if plot_spt_correlations
				
				for j=1:k	

					py"""
					import numpy

					def f(l, a, b, c):
						return a*(l**(-b)) + c
					def g(l, a, b, c):
						return a*numpy.exp(-l/b) + c
					"""

					corrX = data["corrsSPT"][j, 1, :]
					plt.plot(range(1, length(corrX)), corrX, marker=4, color="mediumvioletred", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{\,}_{x, \mathrm{SPT}}$"), zorder=100)

					corrY = data["corrsSPT"][j, 2, :]
					plt.plot(range(1, length(corrY)), corrY, marker=5, color="steelblue", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{\,}_{y, \mathrm{SPT}}$"), zorder=100)
					
					corrZ = data["corrsSPT"][j, 3, :]
					plt.plot(range(1, length(corrZ)), corrZ, marker=5, color="salmon", ls="-", lw=0.5, markersize=5, label=latexstring(L"$C^{\,}_{z, \mathrm{SPT}}$"), zorder=100)

					sites = range(1, length(corrX))[start_:end+1-end_]

					corrX = corrX[start_:end+1-end_]
					corrY = corrY[start_:end+1-end_]

					try
						poptfx, pcov = scipy.optimize.curve_fit(py"f", collect(sites), numpy.abs(corrX))
						perrfx = numpy.sqrt(numpy.diag(pcov))
						chi2 = scipy.stats.chisquare(numpy.abs(corrX), py"f"(collect(sites), poptfx...), length(sites)-4)
						pfx = chi2[2]

						plt.plot(sites, py"f"(collect(sites), poptfx...), ls="--", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\mathrm{SPT}}_{x}| = %$(round(poptfx[1]; digits=2)) \ell^{- %$(round(poptfx[2]; digits=2))} + %$(round(poptfx[3]; digits=2)), \ p = %$(round(pfx[1]; digits=3))$"))
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

						plt.plot(sites, py"g"(collect(sites), poptgx...), ls=":", lw=0.7, color="mediumvioletred", zorder=102, label=latexstring(L"$|C^{\mathrm{SPT}}_{x}| = %$(round(poptgx[1]; digits=2)) e^{- \frac{\ell}{%$(round(poptgx[2]; digits=2))}} + %$(round(poptgx[3]; digits=2)), \ p = %$(round(pgx[1]; digits=3))$"))
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

						plt.plot(sites, py"f"(collect(sites), poptfy...), ls="--", lw=0.7, color="steelblue", zorder=102, label=latexstring(L"$|C^{\mathrm{SPT}}_{y}| = %$(round(poptfy[1]; digits=2)) \ell^{- %$(round(poptfy[2]; digits=2))} + %$(round(poptfy[3]; digits=2)), \ p = %$(round(pfy[1]; digits=3))$"))
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

						plt.plot(sites, py"g"(collect(sites), poptgy...), ls=":", lw=0.7, color="steelblue", zorder=102, label=latexstring(L"$|C^{\mathrm{SPT}}_{y}| = %$(round(poptgy[1]; digits=2)) e^{- \frac{\ell}{%$(round(poptgy[2]; digits=2))}} + %$(round(poptgy[3]; digits=2)), \ p = %$(round(pgy[1]; digits=3))$"))
					catch
						poptgy = 0
						perrgy = 0
						pgy = 0
					end

					ax.minorticks_on()
					ax.grid(which="minor", linewidth=0.2)
					ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0).set_zorder(101)

					plt.xlabel(latexstring(L"$\ell$"))
					plt.ylabel(latexstring(L"$C^{\,}_{\alpha, \mathrm{SPT}} = \langle O^z_{%$(Int(N/2-r/2-2))} O^\alpha_{%$(Int(N/2-r/2-1))} O^z_{%$(Int(N/2-r/2))} O^z_{%$(Int(N/2-r/2))+\ell} O^\alpha_{%$(Int(N/2-r/2+1))+\ell} O^z_{%$(Int(N/2-r/2+2))+\ell} \rangle$"))

					plt.draw()

					if savePlot
						py"createFolderIfMissing"(graphFolder * folder * "spt_correlations/")
						fig.savefig(graphFolder * folder * "spt_correlations/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".png")
						ax.cla()
					end
				end
			end

			if plot_grid_evs

				for j=1:nE

					evs_2[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(data["evs_2"][j, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					evs_2[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean(data["evs_2"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					evs_2[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_2"][j, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

					evs_4[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 4*numpy.mean(data["evs_4"][j, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					evs_4[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 4*numpy.mean(data["evs_4"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					evs_4[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 4*numpy.mean(data["evs_4"][j, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

					# evs_6[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 16*numpy.mean(data["evs_6"][j, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_6[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 16*numpy.mean(data["evs_6"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_6[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*numpy.mean(data["evs_6"][j, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

					# evs_8[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 16*4*numpy.mean(data["evs_8"][j, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_8[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 16*4*numpy.mean(data["evs_8"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_8[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*4*numpy.mean(data["evs_8"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

					# evs_10[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 16*16*numpy.mean(data["evs_10"][j, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_10[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 16*16*numpy.mean(data["evs_10"][j, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
					# evs_10[j, vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*16*numpy.mean(data["evs_10"][j, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

				end
			end

		end

	end

	if plot_grid_evs

		for j=1:nE

			data = pandas.DataFrame(evs_2[j, end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{2}_{x}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
				fig.savefig(graphFolder * folder * "grid_evs/" * "x2_k=$j.png")
				fig.clf()
			end

			data = pandas.DataFrame(evs_2[j, end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{2}_{y}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
				fig.savefig(graphFolder * folder * "grid_evs/" * "y2_k=$j.png")
				fig.clf()
			end

			data = pandas.DataFrame(evs_2[j, end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{2}_{z}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
				fig.savefig(graphFolder * folder * "grid_evs/" * "z2_k=$j.png")
				fig.clf()
			end

			data = pandas.DataFrame(evs_4[j, end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{4}_{x}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
				fig.savefig(graphFolder * folder * "grid_evs/" * "x4_k=$j.png")
				fig.clf()
			end

			data = pandas.DataFrame(evs_4[j, end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{4}_{y}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
				fig.savefig(graphFolder * folder * "grid_evs/" * "y4_k=$j.png")
				fig.clf()
			end

			data = pandas.DataFrame(evs_4[j, end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{4}_{z}}$")))

			plt.xlabel(latexstring(L"$\theta$"))
			plt.ylabel(latexstring(L"$\lambda$"))

			plt.draw()

			if savePlot
				py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
				fig.savefig(graphFolder * folder * "grid_evs/" * "z4_k=$j.png")
				fig.clf()
			end

			# data = pandas.DataFrame(evs_6[j, end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{6}_{x}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if savePlot
			# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			# 	fig.savefig(graphFolder * folder * "grid_evs/" * "x6_k=$j.png")
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_6[j, end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{6}_{y}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if savePlot
			# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			# 	fig.savefig(graphFolder * folder * "grid_evs/" * "y6_k=$j.png")
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_6[j, end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{6}_{z}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if savePlot
			# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			# 	fig.savefig(graphFolder * folder * "grid_evs/" * "z6_k=$j.png")
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_8[j, end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{8}_{x}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if savePlot
			# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			# 	fig.savefig(graphFolder * folder * "grid_evs/" * "x8_k=$j.png")
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_8[j, end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{8}_{y}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if savePlot
			# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			# 	fig.savefig(graphFolder * folder * "grid_evs/" * "y8_k=$j.png")
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_8[j, end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{8}_{z}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if savePlot
			# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			# 	fig.savefig(graphFolder * folder * "grid_evs/" * "z8_k=$j.png")
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_10[j, end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{10}_{x}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if savePlot
			# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			# 	fig.savefig(graphFolder * folder * "grid_evs/" * "x10_k=$j.png")
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_10[j, end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{10}_{y}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if savePlot
			# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			# 	fig.savefig(graphFolder * folder * "grid_evs/" * "y10_k=$j.png")
			# 	fig.clf()
			# end

			# data = pandas.DataFrame(evs_10[j, end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

			# seaborn.heatmap(data, square=true, cbar_kws=Dict("label" => latexstring(L"$\overline{G^{10}_{z}}$")))

			# plt.xlabel(latexstring(L"$\theta$"))
			# plt.ylabel(latexstring(L"$\lambda$"))

			# plt.draw()

			# if savePlot
			# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			# 	fig.savefig(graphFolder * folder * "grid_evs/" * "z10_k=$j.png")
			# 	fig.clf()
			# end
		end
	end

	if plot_degeneracies

		data = pandas.DataFrame(degens[end:-1:1, :], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		seaborn.heatmap(data, cbar_kws=Dict("label" => latexstring(L"$n \, | \, E^{\,}_{m} - E^{\,}_{0} < \Delta \varepsilon, \quad 0 \leq m < n$")))

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.title(latexstring(L"$\Delta \varepsilon = %$(epsDegen)$"))

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "degeneracies/")
			fig.savefig(graphFolder * folder * "degeneracies/" * "eps=$(epsDegen).png")
			ax.cla()
		end
	end

	if plot_lines_c

		# for j=1:nE

		# 	for l=1:length(lam)

		# 		ax.errorbar(theta, cs[j, l, :, 1], cs[j, l, :, 2], ecolor="steelblue", elinewidth=0.5, color="mediumvioletred", marker=".", linewidth=0.5, markersize=5, zorder=105)

		# 		ax.minorticks_on()
		# 		ax.grid(which="minor", linewidth=0.2)
		# 		ax.grid(which="major", linewidth=0.6)

		# 		plt.xlabel(latexstring(L"$\theta$"))

		# 		plt.draw()

		# 		if savePlot
		# 			py"createFolderIfMissing"(graphFolder * folder * "lines_c/")
		# 			fig.savefig(graphFolder * folder * "lines_c/" * "lam=$(lam[l])_k=$j.png")
		# 			ax.cla()
		# 		end
		# 	end
		# end

		for j=1:nE

			for t=1:length(theta)

				ax.errorbar(lam, cs[j, :, t, 1], cs[j, :, t, 2], ecolor="steelblue", elinewidth=0.5, color="mediumvioletred", marker=".", linewidth=0.5, markersize=5, zorder=105)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)

				plt.xlabel(latexstring(L"$\theta$"))

				plt.draw()

				if savePlot
					py"createFolderIfMissing"(graphFolder * folder * "lines_c/")
					fig.savefig(graphFolder * folder * "lines_c/" * "theta=$(theta[t])_k=$j.png")
					ax.cla()
				end
			end
		end
	end

	if plot_lines_mag

		for j=1:nE

			# for l=1:length(lam)
				
			# 	py"""
			# 	def f(theta, a, b, c):
			# 		import numpy

			# 		return a*numpy.power(numpy.abs(theta-b), c)*numpy.heaviside(b-theta, 1)
			# 	"""

			# 	ax.plot(theta, numpy.abs(mags[j, l, :, 1]), zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{x}|$"))
			# 	ax.plot(theta, numpy.abs(mags[j, l, :, 2]), zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{x}|$"))
			# 	ax.plot(theta, numpy.abs(mags[j, l, :, 3]), zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{y}|$"))
			# 	ax.plot(theta, numpy.abs(mags[j, l, :, 4]), zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{y}|$"))
			# 	ax.plot(theta, numpy.abs(mags[j, l, :, 5]), zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{z}|$"))
			# 	ax.plot(theta, numpy.abs(mags[j, l, :, 6]), zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{z}|$"))

			# 	ax.minorticks_on()
			# 	ax.grid(which="minor", linewidth=0.2)
			# 	ax.grid(which="major", linewidth=0.6)
			# 	ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.2), borderaxespad=0, ncol=3).set_zorder(101)

			# 	plt.xlabel(latexstring(L"$\theta$"))

			# 	plt.draw()

			# 	if savePlot
			# 		py"createFolderIfMissing"(graphFolder * folder * "lines_mag/")
			# 		fig.savefig(graphFolder * folder * "lines_mag/" * "lam=$(lam[l])_k=$j.png")
			# 		ax.cla()
			# 	end
			# end

			for t=1:length(theta)
				
				py"""
				def f(lam, a, b, c):
					import numpy

					return a*numpy.power(numpy.abs(lam-b), c)*numpy.heaviside(b-lam, 1)
				"""

				ax.plot(lam, numpy.abs(mags[j, :, t, 1]), zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{x}|$"))
				ax.plot(lam, numpy.abs(mags[j, :, t, 2]), zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{x}|$"))
				ax.plot(lam, numpy.abs(mags[j, :, t, 3]), zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{y}|$"))
				ax.plot(lam, numpy.abs(mags[j, :, t, 4]), zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{y}|$"))
				ax.plot(lam, numpy.abs(mags[j, :, t, 5]), zorder=105, label=latexstring(L"$|m^{\mathrm{uni}}_{z}|$"))
				ax.plot(lam, numpy.abs(mags[j, :, t, 6]), zorder=105, label=latexstring(L"$|m^{\mathrm{sta}}_{z}|$"))

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="lower right", bbox_to_anchor=(0.99, 0.2), borderaxespad=0, ncol=3).set_zorder(101)

				plt.xlabel(latexstring(L"$\lambda$"))

				plt.draw()

				if savePlot
					py"createFolderIfMissing"(graphFolder * folder * "lines_mag/")
					fig.savefig(graphFolder * folder * "lines_mag/" * "theta=$(theta[t])_k=$j.png")
					ax.cla()
				end
			end
		end
	end

	if plot_lines_correlations_ft
		
		for j=1:nE

			# for l=1:length(lam)

			# 	color = "tab:blue"
			# 	ax.plot(theta, ft_maxs_corrs_z[j, l, :, 1], marker="x", markersize=2, zorder=100, color=color)
			# 	ax.set_xlabel(latexstring(L"$\theta$"))
			# 	ax.set_ylabel(latexstring(L"$\mathrm{frequency}$"), color=color)
			# 	ax.tick_params(axis="y", labelcolor=color)
				
			# 	colortwin = "tab:orange"
			# 	axtwin = ax.twinx()
			# 	axtwin.plot(theta, ft_maxs_corrs_z[j, l, :, 2], marker="x", markersize=3, zorder=101, color=colortwin)
			# 	axtwin.set_ylabel(latexstring(L"$\mathrm{ampltitude}$"), color=colortwin)
			# 	axtwin.tick_params(axis="y", labelcolor=colortwin)

			# 	ax.minorticks_on()
			# 	ax.grid(which="minor", linewidth=0.2)
			# 	ax.grid(which="major", linewidth=0.6)

			# 	plt.draw()

			# 	if savePlot
			# 		py"createFolderIfMissing"(graphFolder * folder * "lines_correlations_ft/")
			# 		fig.savefig(graphFolder * folder * "lines_correlations_ft/" * "lam=$(lam[l])_k=$j.png")
			# 		ax.cla()
			# 	end
			# end

			for t=1:length(theta)

				color = "tab:blue"
				ax.plot(lam, ft_maxs_corrs_z[j, :, t, 1], marker="x", markersize=2, zorder=100, color=color)
				ax.set_xlabel(latexstring(L"$\lambda$"))
				ax.set_ylabel(latexstring(L"$\mathrm{frequency}$"), color=color)
				ax.tick_params(axis="y", labelcolor=color)
				
				colortwin = "tab:orange"
				axtwin = ax.twinx()
				axtwin.plot(lam, ft_maxs_corrs_z[j, :, t, 2], marker="x", markersize=3, zorder=101, color=colortwin)
				axtwin.set_ylabel(latexstring(L"$\mathrm{ampltitude}$"), color=colortwin)
				axtwin.tick_params(axis="y", labelcolor=colortwin)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)

				plt.draw()

				if savePlot
					py"createFolderIfMissing"(graphFolder * folder * "lines_correlations_ft/")
					fig.savefig(graphFolder * folder * "lines_correlations_ft/" * "theta=$(theta[t])_k=$j.png")
					ax.cla()
					axtwin.cla()
				end
			end
		end
	end

	if plot_lines_spins_ft

		for j=1:nE

			# for l=1:length(lam)

			# 	color = "tab:blue"
			# 	ax.plot(theta, ft_maxs_spins_z[j, l, :, 1], marker="x", markersize=5, zorder=100, color=color)
			# 	ax.set_xlabel(latexstring(L"$\theta$"))
			# 	ax.set_ylabel(latexstring(L"$\mathrm{frequency}$"), color=color)
			# 	ax.tick_params(axis="y", labelcolor=color)
				
			# 	colortwin = "tab:orange"
			# 	axtwin = ax.twinx()
			# 	axtwin.plot(theta, ft_maxs_spins_z[j, l, :, 2], marker="x", markersize=5, zorder=101, color=colortwin)
			# 	axtwin.set_ylabel(latexstring(L"$\mathrm{ampltitude}$"), color=colortwin)
			# 	axtwin.tick_params(axis="y", labelcolor=colortwin)

			# 	ax.minorticks_on()
			# 	ax.grid(which="minor", linewidth=0.2)
			# 	ax.grid(which="major", linewidth=0.6)

			# 	plt.draw()

			# 	if savePlot
			# 		py"createFolderIfMissing"(graphFolder * folder * "lines_spins_ft/")
			# 		fig.savefig(graphFolder * folder * "lines_spins_ft/" * "lam=$(lam[l])_k=$j.png")
			# 		ax.cla()
			# 	end
			# end

			for t=1:length(theta)

				color = "tab:blue"
				ax.plot(lam, ft_maxs_spins_z[j, :, t, 1], marker="x", markersize=5, zorder=100, color=color)
				ax.set_xlabel(latexstring(L"$\lambda$"))
				ax.set_ylabel(latexstring(L"$\mathrm{frequency}$"), color=color)
				ax.tick_params(axis="y", labelcolor=color)
				
				colortwin = "tab:orange"
				axtwin = ax.twinx()
				axtwin.plot(lam, ft_maxs_spins_z[j, :, t, 2], marker="x", markersize=5, zorder=101, color=colortwin)
				axtwin.set_ylabel(latexstring(L"$\mathrm{ampltitude}$"), color=colortwin)
				axtwin.tick_params(axis="y", labelcolor=colortwin)

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)

				plt.draw()

				if savePlot
					py"createFolderIfMissing"(graphFolder * folder * "lines_spins_ft/")
					fig.savefig(graphFolder * folder * "lines_spins_ft/" * "theta=$(theta[t])_k=$j.png")
					ax.cla()
					axtwin.cla()
				end
			end
		end
	end

	if plot_scaling_gaps | plot_lines_gaps
		push!(diffsNs, diffs)
	end
	if plot_scaling_en
		push!(ENs, Es)
	end

end

if plot_scaling_gaps

	indSort = numpy.argsort(Ns).+1

	Ns = numpy.array(Ns)[indSort]
	diffsNs = numpy.array(diffsNs)[indSort, :, :, :]

	for l=1:length(lam)
		for t=1:length(theta)

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

				r2_ = py"r2"(diffsNs[:, l, t, k], py"f"(Ns, popt...))

				ax.plot(1 ./Ns, py"f"(Ns, popt...), color="mediumvioletred", linestyle="--", zorder=99, label=latexstring(L"$\frac{%$(round(popt[1]; digits=3)) \pm %$(round(perr[1]; digits=3))}{N} + (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3))),\ R^2 = %$(round(r2_; digits=3))$"))

				ax.scatter(1 ./Ns, diffsNs[:, l, t, k], marker="x", color="steelblue", zorder=100)

				ax.set_xlabel(L"$1/2N$")
				ax.set_ylabel(L"$E_1-E_0$")

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0).set_zorder(101)

				plt.draw()

				if savePlot
					py"createFolderIfMissing"(graphFolder * "scaling/" * "lam=$(lam[l])_theta=$(theta[t])/")
					fig.savefig(graphFolder * "scaling/" * "lam=$(lam[l])_theta=$(theta[t])/" * "E$(k)-E$(k-1).png")
					ax.cla()
				end
			end
		end
	end
end

if plot_lines_gaps

	indSort = numpy.argsort(Ns).+1

	Ns = numpy.array(Ns)[indSort]
	diffsNs = numpy.array(diffsNs)[indSort, :, :, :]

	for n=1:length(Ns)
		for t=1:length(theta)

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

				# popt, pcov = scipy.optimize.curve_fit(py"f", Ns, diffsNs[n, :, t, k])
				# perr = numpy.sqrt(numpy.diag(pcov))

				# r2_ = py"r2"(diffsNs[n, :, t, k], py"f"(Ns, popt...))

				# ax.plot(lam, py"f"(Ns, popt...), color="mediumvioletred", linestyle="--", zorder=99, label=latexstring(L"$\frac{%$(round(popt[1]; digits=3)) \pm %$(round(perr[1]; digits=3))}{N} + (%$(round(popt[2]; digits=3)) \pm %$(round(perr[2]; digits=3))),\ R^2 = %$(round(r2_; digits=3))$"))

				ax.scatter(lam, diffsNs[n, :, t, k], marker="x", color="steelblue", zorder=100)

				ax.set_xlabel(L"$\lambda$")
				ax.set_ylabel(L"$E_1-E_0$")

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0).set_zorder(101)

				plt.draw()

				if savePlot
					py"createFolderIfMissing"(graphFolder * "scaling/" * "N=$(Ns[n])_theta=$(theta[t])/")
					fig.savefig(graphFolder * "scaling/" * "N=$(Ns[n])_theta=$(theta[t])/" * "E$(k)-E$(k-1).png")
					ax.cla()
				end
			end
		end
	end
end

if plot_scaling_en

	indSort = numpy.argsort(Ns).+1

	Ns = numpy.array(Ns)[indSort]
	ENs = numpy.array(ENs)[indSort, :, :, :]

	for l=1:length(lam)
		for t=1:length(theta)

			for k=1:nE
								
				py"""
				def f(N, a, b):
					import numpy 

					return a/N + b
				"""								
				py"""
				def g(N, a, b):
					import numpy 

					return a*N - b*0.713*numpy.pi/(6*N)
				"""

				py"""
				def r2(y, fit):
					import numpy

					ss_res = numpy.sum((y-fit)**2)
					ss_tot = numpy.sum((y-numpy.mean(y))**2)

					return (1 - ss_res/ss_tot)
				"""

				popt, pcov = scipy.optimize.curve_fit(py"g", Ns, ENs[:, l, t, k])
				perr = numpy.sqrt(numpy.diag(pcov))

				println(Ns,  ENs[:, l, t, k])
				println(popt, perr)

				r2_ = py"r2"(ENs[:, l, t, k], py"g"(Ns, popt...))

				ax.plot(Ns, py"g"(Ns, popt...), color="mediumvioletred", linestyle="--", zorder=99, label=latexstring(L"$(%$(round(popt[1]; digits=2)) \pm %$(round(perr[1]; digits=2))) \, (2N) - \frac{\pi \, (%$(round(popt[2]; digits=2)) \pm %$(round(perr[2]; digits=2)))}{6 \, (2N)},\ R^2 = %$(round(r2_; digits=3))$"))

				ax.scatter(Ns, ENs[:, l, t, k], marker="x", color="steelblue", zorder=100)

				ax.set_xlabel(latexstring(L"$2N$"))
				ax.set_ylabel(latexstring(L"$E_{%$(k-1)}$"))

				ax.minorticks_on()
				ax.grid(which="minor", linewidth=0.2)
				ax.grid(which="major", linewidth=0.6)
				ax.legend(loc="upper left", bbox_to_anchor=(0.01, 0.99), borderaxespad=0).set_zorder(101)

				plt.draw()

				if savePlot
					py"createFolderIfMissing"(graphFolder * "energies/" * "lam=$(lam[l])_theta=$(theta[t])/")
					fig.savefig(graphFolder * "energies/" * "lam=$(lam[l])_theta=$(theta[t])/" * "E$(k-1).png")
					ax.cla()
				end
			end
		end
	end
end
