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
seaborn = pyimport("seaborn")
scipy = pyimport("scipy")
cycler = pyimport("cycler")

PBCs = false

if PBCs
	graphFolder = "../graphs/pbc/grid•51/"
	dataFolder = "../data/pbc/grid•51/"
else
	graphFolder = "../graphs/obc/grid•51/"
	dataFolder = "../data/obc/grid•51/"
end

computeFolder = "analyzed/"
folders = py"folderNames"(dataFolder)

plot_grid_evs = true
plot_grid_evs_stag = false

r = 64

savePlot = true
showPlot = false

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

cmap = matplotlib.colors.LinearSegmentedColormap.from_list("...", colors=[colors[7],colors[9],colors[1],colors[6],colors[4]], N=256)

for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder * computeFolder)

	lam = numpy.sort(numpy.unique([vals[i]["lam"] for i=1:length(vals)]))
	theta = numpy.sort(numpy.unique([vals[i]["theta"] for i=1:length(vals)]))

	evs_2 = numpy.zeros((length(lam), length(theta), 3))
	evs_3 = numpy.zeros((length(lam), length(theta), 3))
	evs_4 = numpy.zeros((length(lam), length(theta), 3))
	evs_6 = numpy.zeros((length(lam), length(theta), 3))
	evs_8 = numpy.zeros((length(lam), length(theta), 3))
	evs_10 = numpy.zeros((length(lam), length(theta), 3))
	evs_2_stag = numpy.zeros((length(lam), length(theta), 3))
	evs_4_stag = numpy.zeros((length(lam), length(theta), 3))
	evs_6_stag = numpy.zeros((length(lam), length(theta), 3))
	evs_8_stag = numpy.zeros((length(lam), length(theta), 3))
	evs_10_stag = numpy.zeros((length(lam), length(theta), 3))

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

			if plot_grid_evs

				evs_2[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.mean(data["evs_2"][1, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
				evs_2[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.mean(data["evs_2"][1, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
				evs_2[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.mean(data["evs_2"][1, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

				# evs_3[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 2*numpy.abs(data["evs_3"][1, 1, Int(N/2)])
				# evs_3[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 2*numpy.abs(data["evs_3"][1, 2, Int(N/2)])
				# evs_3[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 2*numpy.abs(data["evs_3"][1, 3, Int(N/2)])

				evs_4[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 4*numpy.mean(data["evs_4"][1, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
				evs_4[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 4*numpy.mean(data["evs_4"][1, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
				evs_4[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 4*numpy.mean(data["evs_4"][1, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

				evs_6[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 16*numpy.mean(data["evs_6"][1, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
				evs_6[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 16*numpy.mean(data["evs_6"][1, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
				evs_6[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*numpy.mean(data["evs_6"][1, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

				evs_8[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 16*4*numpy.mean(data["evs_8"][1, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
				evs_8[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 16*4*numpy.mean(data["evs_8"][1, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
				evs_8[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*4*numpy.mean(data["evs_8"][1, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

				evs_10[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 16*16*numpy.mean(data["evs_10"][1, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
				evs_10[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 16*16*numpy.mean(data["evs_10"][1, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])
				evs_10[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 16*16*numpy.mean(data["evs_10"][1, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)])

			end

			# if plot_grid_evs_stag

			# 	evs_2_stag[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= numpy.abs(numpy.mean(data["evs_2"][1, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)].*[(-1)^j for j=Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)]))
			# 	evs_2_stag[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= numpy.abs(numpy.mean(data["evs_2"][1, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)].*[(-1)^j for j=Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)]))
			# 	evs_2_stag[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= numpy.abs(numpy.mean(data["evs_2"][1, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)].*[(-1)^j for j=Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)]))

			# 	evs_4_stag[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 1] .= 4*numpy.abs(numpy.mean(data["evs_4"][1, 1, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)].*[(-1)^j for j=Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)]))
			# 	evs_4_stag[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 2] .= 4*numpy.abs(numpy.mean(data["evs_4"][1, 2, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)].*[(-1)^j for j=Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)]))
			# 	evs_4_stag[vals[i]["lam"].==lam, vals[i]["theta"].==theta, 3] .= 4*numpy.abs(numpy.mean(data["evs_4"][1, 3, Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)].*[(-1)^j for j=Int(N/2)-Int(r/2):Int(N/2)+Int(r/2)]))

			# end
		end
	end

	if plot_grid_evs

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_2[end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{2,x}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "x2.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_2[end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{2,y}}$"), fontsize=18, pad=10)
		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "y2.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_2[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{2,z}}$"), fontsize=18, pad=10)
		
		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "z2.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		# fig, ax = plt.subplots(figsize=(5, 4))

		# data = pandas.DataFrame(evs_3[end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		# axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		# cb = fig.colorbar(axs.collections[1]"))
		# cb.ax.tick_params(labelsize=18)

		# plt.xlabel(latexstring(L"$\theta$"))
		# plt.ylabel(latexstring(L"$\lambda$"))

		# plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		# plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		# plt.xticks(rotation=0)
		# plt.yticks(rotation=90)

		# ax.minorticks_on()

		# for spine in axs.spines.items()
		# 	spine[2].set_visible(true)
		# end

		# plt.draw()

		# if savePlot
		# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
		# 	fig.savefig(graphFolder * folder * "grid_evs/" * "x3.pdf")
		# 	fig.clf()
		# end

		# if showPlot
		# 	plt.show()
		# end

		# plt.close()

		# fig, ax = plt.subplots(figsize=(5, 4))

		# data = pandas.DataFrame(evs_3[end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		# axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		# cb = fig.colorbar(axs.collections[1]"))
		# cb.ax.tick_params(labelsize=18)

		# plt.xlabel(latexstring(L"$\theta$"))
		# plt.ylabel(latexstring(L"$\lambda$"))

		# plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		# plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		# plt.xticks(rotation=0)
		# plt.yticks(rotation=90)

		# ax.minorticks_on()

		# for spine in axs.spines.items()
		# 	spine[2].set_visible(true)
		# end

		# plt.draw()

		# if savePlot
		# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
		# 	fig.savefig(graphFolder * folder * "grid_evs/" * "y3.pdf")
		# 	fig.clf()
		# end

		# if showPlot
		# 	plt.show()
		# end

		# plt.close()

		# fig, ax = plt.subplots(figsize=(5, 4))

		# data = pandas.DataFrame(evs_3[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		# axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		# cb = fig.colorbar(axs.collections[1]"))
		# cb.ax.tick_params(labelsize=18)

		# plt.xlabel(latexstring(L"$\theta$"))
		# plt.ylabel(latexstring(L"$\lambda$"))

		# plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		# plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		# plt.xticks(rotation=0)
		# plt.yticks(rotation=90)

		# ax.minorticks_on()

		# for spine in axs.spines.items()
		# 	spine[2].set_visible(true)
		# end

		# plt.draw()

		# if savePlot
		# 	py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
		# 	fig.savefig(graphFolder * folder * "grid_evs/" * "z3.pdf")
		# 	fig.clf()
		# end

		# if showPlot
		# 	plt.show()
		# end

		# plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_4[end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{4,x}}$"), fontsize=18, pad=10)
		
		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "x4.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_4[end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{4,y}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "y4.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_4[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{4,z}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "z4.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_6[end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{6,x}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "x6.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_6[end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{6,y}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "y6.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_6[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{6,z}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "z6.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_8[end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{8,x}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "x8.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_8[end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{8,y}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "y8.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_8[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{8,z}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "z8.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_10[end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{10,x}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "x10.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_10[end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{10,y}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "y10.pdf")
			fig.clf()
		end

		plt.close()

		if showPlot
			plt.show()
		end

		fig, ax = plt.subplots(figsize=(5, 4))

		data = pandas.DataFrame(evs_10[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

		axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
		cb = fig.colorbar(axs.collections[1])
		cb.ax.tick_params(labelsize=18)

		plt.title(label=latexstring(L"$\overline{G^{10,z}}$"), fontsize=18, pad=10)

		plt.xlabel(latexstring(L"$\theta$"))
		plt.ylabel(latexstring(L"$\lambda$"))

		plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
		plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
		plt.xticks(rotation=0)
		plt.yticks(rotation=90)

		ax.minorticks_on()

		for spine in axs.spines.items()
			spine[2].set_visible(true)
		end

		plt.draw()

		if savePlot
			py"createFolderIfMissing"(graphFolder * folder * "grid_evs/")
			fig.savefig(graphFolder * folder * "grid_evs/" * "z10.pdf")
			fig.clf()
		end

		if showPlot
			plt.show()
		end

		plt.close()
	end

	# if plot_grid_evs_stag

	# 	fig, ax = plt.subplots(figsize=(5, 4))

	# 	data = pandas.DataFrame(evs_2_stag[end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

	# 	axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
	# 	cb = fig.colorbar(axs.collections[1]hrm{sta},x}}$"))
	# 	cb.ax.tick_params(labelsize=18)

	# 	plt.xlabel(latexstring(L"$\theta$"))
	# 	plt.ylabel(latexstring(L"$\lambda$"))

	# 	plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
	# 	plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
	# 	plt.xticks(rotation=0)
	# 	plt.yticks(rotation=90)

	# 	ax.minorticks_on()

	# 	for spine in ax.spines.items()
	# 		spine[2].set_visible(true)
	# 		spine[2].set_linewidth(0.8)
	# 	end

	# 	plt.draw()

	# 	if savePlot
	# 		py"createFolderIfMissing"(graphFolder * folder * "grid_evs_stag/")
	# 		fig.savefig(graphFolder * folder * "grid_evs_stag/" * "x2.pdf")
	# 		fig.clf()
	# 	end

	# 	if showPlot
	# 		plt.show()
	# 	end

	# 	plt.close()

	# 	fig, ax = plt.subplots(figsize=(5, 4))

	# 	data = pandas.DataFrame(evs_2_stag[end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

	# 	axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
	# 	cb = fig.colorbar(axs.collections[1]hrm{sta}}_{y}}$"))
	# 	cb.ax.tick_params(labelsize=18)

	# 	plt.xlabel(latexstring(L"$\theta$"))
	# 	plt.ylabel(latexstring(L"$\lambda$"))

	# 	plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
	# 	plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
	# 	plt.xticks(rotation=0)
	# 	plt.yticks(rotation=90)

	# 	ax.minorticks_on()

	# 	for spine in ax.spines.items()
	# 		spine[2].set_visible(true)
	# 		spine[2].set_linewidth(0.8)
	# 	end

	# 	plt.draw()

	# 	if savePlot
	# 		py"createFolderIfMissing"(graphFolder * folder * "grid_evs_stag/")
	# 		fig.savefig(graphFolder * folder * "grid_evs_stag/" * "y2.pdf")
	# 		fig.clf()
	# 	end

	# 	if showPlot
	# 		plt.show()
	# 	end

	# 	plt.close()

	# 	fig, ax = plt.subplots(figsize=(5, 4))

	# 	data = pandas.DataFrame(evs_2_stag[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

	# 	axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
	# 	cb = fig.colorbar(axs.collections[1]hrm{sta}}_{z}}$"))
	# 	cb.ax.tick_params(labelsize=18)

	# 	plt.xlabel(latexstring(L"$\theta$"))
	# 	plt.ylabel(latexstring(L"$\lambda$"))

	# 	plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
	# 	plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
	# 	plt.xticks(rotation=0)
	# 	plt.yticks(rotation=90)

	# 	ax.minorticks_on()

	# 	for spine in ax.spines.items()
	# 		spine[2].set_visible(true)
	# 		spine[2].set_linewidth(0.8)
	# 	end

	# 	plt.draw()

	# 	if savePlot
	# 		py"createFolderIfMissing"(graphFolder * folder * "grid_evs_stag/")
	# 		fig.savefig(graphFolder * folder * "grid_evs_stag/" * "z2.pdf")
	# 		fig.clf()
	# 	end

	# 	if showPlot
	# 		plt.show()
	# 	end

	# 	plt.close()

	# 	fig, ax = plt.subplots(figsize=(5, 4))

	# 	data = pandas.DataFrame(evs_4_stag[end:-1:1, :, 1], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

	# 	axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
	# 	cb = fig.colorbar(axs.collections[1]hrm{sta}}_{x}}$"))
	# 	cb.ax.tick_params(labelsize=18)

	# 	plt.xlabel(latexstring(L"$\theta$"))
	# 	plt.ylabel(latexstring(L"$\lambda$"))

	# 	plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
	# 	plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
	# 	plt.xticks(rotation=0)
	# 	plt.yticks(rotation=90)

	# 	ax.minorticks_on()

	# 	for spine in ax.spines.items()
	# 		spine[2].set_visible(true)
	# 		spine[2].set_linewidth(0.8)
	# 	end

	# 	plt.draw()

	# 	if savePlot
	# 		py"createFolderIfMissing"(graphFolder * folder * "grid_evs_stag/")
	# 		fig.savefig(graphFolder * folder * "grid_evs_stag/" * "x4.pdf")
	# 		fig.clf()
	# 	end

	# 	if showPlot
	# 		plt.show()
	# 	end

	# 	plt.close()

	# 	fig, ax = plt.subplots(figsize=(5, 4))

	# 	data = pandas.DataFrame(evs_4_stag[end:-1:1, :, 2], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

	# 	axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
	# 	cb = fig.colorbar(axs.collections[1]hrm{sta}}_{y}}$"))
	# 	cb.ax.tick_params(labelsize=18)

	# 	plt.xlabel(latexstring(L"$\theta$"))
	# 	plt.ylabel(latexstring(L"$\lambda$"))

	# 	plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
	# 	plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
	# 	plt.xticks(rotation=0)
	# 	plt.yticks(rotation=90)

	# 	ax.minorticks_on()

	# 	for spine in ax.spines.items()
	# 		spine[2].set_visible(true)
	# 		spine[2].set_linewidth(0.8)
	# 	end

	# 	plt.draw()

	# 	if savePlot
	# 		py"createFolderIfMissing"(graphFolder * folder * "grid_evs_stag/")
	# 		fig.savefig(graphFolder * folder * "grid_evs_stag/" * "y4.pdf")
	# 		fig.clf()
	# 	end

	# 	if showPlot
	# 		plt.show()
	# 	end

	# 	plt.close()

	# 	fig, ax = plt.subplots(figsize=(5, 4))

	# 	data = pandas.DataFrame(evs_4_stag[end:-1:1, :, 3], index=numpy.round(lam, 5)[end:-1:1], columns=numpy.round(theta, 4))

	# 	axs = seaborn.heatmap(data, square=true, cbar=false, rasterized=true, cmap=cmap)
		
	# 	cb = fig.colorbar(axs.collections[1]hrm{sta}}_{z}}$"))
	# 	cb.ax.tick_params(labelsize=18)

	# 	plt.xlabel(latexstring(L"$\theta$"))
	# 	plt.ylabel(latexstring(L"$\lambda$"))

	# 	plt.xticks([0*length(lam), 0.5*length(lam), 1*length(lam)], [latexstring(L"$0$"), latexstring(L"$\pi/4$"), latexstring(L"$\pi/2$")])
	# 	plt.yticks([1*length(lam), 0.5*length(lam), 0*length(lam)], [latexstring(L"$0$"), latexstring(L"$1/2$"), latexstring(L"$1$")])
		
	# 	plt.xticks(rotation=0)
	# 	plt.yticks(rotation=90)

	# 	ax.minorticks_on()

	# 	for spine in ax.spines.items()
	# 		spine[2].set_visible(true)
	# 		spine[2].set_linewidth(0.8)
	# 	end

	# 	plt.draw()

	# 	if savePlot
	# 		py"createFolderIfMissing"(graphFolder * folder * "grid_evs_stag/")
	# 		fig.savefig(graphFolder * folder * "grid_evs_stag/" * "z4.pdf")
	# 		fig.clf()
	# 	end

	# 	if showPlot
	# 		plt.show()
	# 	end

	# 	plt.close()
	# end
end
