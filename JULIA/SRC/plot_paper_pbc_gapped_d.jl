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
	graphFolder = "../GRAPHS/PBC/GAPPED_D/"
	dataFolder = "../DATA/PBC/GAPPED_D/"
else
	graphFolder = "../GRAPHS/OBC/GAPPED_D/"
	dataFolder = "../DATA/OBC/GAPPED_D/"
end

computeFolder = "ANALYZED/"
folders = py"folderNames"(dataFolder)

plot_spins = false
plot_scaling_en = true

nE = 1

savePlot = true

Ns = []
diffsNs = []

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

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


			if plot_spins

				for j=1:k

					fig, ax = plt.subplots(figsize=(4, 4))

					mX = numpy.abs(data["mags"][j, 1, :])
					mY = numpy.abs(data["mags"][j, 2, :])
					mZ = numpy.abs(data["mags"][j, 3, :])

					ax.plot(range(1, length(mX)), mX, color=colors[1], label=latexstring(L"$\langle \widehat{X}^{\,}_{j} \rangle$"), zorder=100)
					ax.plot(range(1, length(mY)), mY, color=colors[4], label=latexstring(L"$\langle \widehat{Y}^{\,}_{j} \rangle$"), zorder=100)
					ax.plot(range(1, length(mZ)), mZ, color=colors[7], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j} \rangle$"), zorder=100)
					
					ax.minorticks_on()
					# ax.grid(which="minor", linewidth=0.2)
					# ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper left", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(101)

					plt.xlabel(latexstring(L"$j$"))

					plt.draw()
			
					if savePlot
						py"createFolderIfMissing"(graphFolder * folder * "spins/")
						fig.savefig(graphFolder * folder * "spins/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".pdf")
						ax.cla()
					end

					plt.close()

					fig, ax = plt.subplots(figsize=(4, 4))

					mXSPT = numpy.abs(data["opSPT"][j, 1, :])
					mYSPT = numpy.abs(data["opSPT"][j, 2, :])

					ax.plot(range(2, length(mXSPT)+1), mXSPT, color=colors[1], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \widehat{X}^{\,}_{j} \widehat{Z}^{\,}_{j+1} \rangle$"), zorder=100)
					ax.plot(range(2, length(mYSPT)+1), mYSPT, color=colors[4], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \widehat{Y}^{\,}_{j} \widehat{Z}^{\,}_{j+1} \rangle$"), zorder=100)
					
					ax.minorticks_on()
					# ax.grid(which="minor", linewidth=0.2)
					# ax.grid(which="major", linewidth=0.6)
					ax.legend(loc="upper left", ncol=1, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(101)

					plt.xlabel(latexstring(L"$j$"))

					plt.draw()
			
					if savePlot
						py"createFolderIfMissing"(graphFolder * folder * "spinsSPT/")
						fig.savefig(graphFolder * folder * "spinsSPT/" * chop(dataFilenames[i], tail=5) * "_energy=$(data["energies"][j])" * ".pdf")
						ax.cla()
					end

					plt.close()
				end
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

fig = plt.figure(figsize=(16.5, 3.5))
axt = plt.subplot2grid((2, 3), (0, 0))
axl = plt.subplot2grid((2, 3), (1, 0))
axt_ = plt.subplot2grid((2, 3), (0, 1))
axl_ = plt.subplot2grid((2, 3), (1, 1))
axt__ = plt.subplot2grid((2, 3), (0, 2))
axl__ = plt.subplot2grid((2, 3), (1, 2))
plt.subplots_adjust(wspace=0.15)

ps = []
data = load(dataFolder * "N=32_chi=128/ANALYZED/N=32_chi=128_lam=0.5_theta=0.785398163397448.jld2")

println(data["energies"])

mX = -data["opSPT"][1, 1, :]
mY = -data["opSPT"][1, 2, :]
mZ = -data["opSPT"][1, 3, :]
println(numpy.mean(mZ))

p, = axt.plot(range(1, length(mX)), mX, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[1], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{X}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = axt.plot(range(1, length(mY)), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{Y}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = axt.plot(range(1, length(mZ)), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{Z}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)

mX = -data["mags"][1, 1, :]
mY = -data["mags"][1, 2, :]
mZ = -data["mags"][1, 3, :]
println(numpy.mean(mZ))

p, = axl.plot(range(1, length(mX)), mX, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[1], label=latexstring(L"$\langle \widehat{X}^{\,}_{j} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = axl.plot(range(1, length(mY)), mY,  markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[4], label=latexstring(L"$\langle \widehat{Y}^{\,}_{j} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = axl.plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[7], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)

axt.set(xticklabels=[])

axt.minorticks_on()
axl.minorticks_on()

axl.set_xlabel(L"$j$")

axt.set_xticks([0, 10, 20, 32])
axl.set_xticks([0, 10, 20, 32])

axt.set_xlim(0, 33)
axl.set_xlim(0, 33)

axt.set_ylim(-1.15, 0.15)
axl.set_ylim(-0.15, 1.15)

axt.set_title(latexstring(L"(a) \ PBC, \ $(\pi/4,\ 1/2)$"), pad=8)

# axt.annotate("(a)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# axl.annotate("(b)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

data = load(dataFolder * "N=32_chi=128/ANALYZED/N=32_chi=128_lam=0.3_theta=0.785398163397448.jld2")

println(data["energies"])

mX = data["opSPT"][1, 1, :]
mY = data["opSPT"][1, 2, :]
mZ = data["opSPT"][1, 3, :]
println(numpy.mean(mZ))

p, = axt_.plot(range(1, length(mX)), mX, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axt_.plot(range(1, length(mY)), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axt_.plot(range(1, length(mZ)), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

mX = data["mags"][1, 1, :]
mY = data["mags"][1, 2, :]
mZ = data["mags"][1, 3, :]
println(numpy.mean(mZ))

p, = axl_.plot(range(1, length(mX)), mX, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axl_.plot(range(1, length(mY)), mY,  markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axl_.plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

axt_.set(xticklabels=[])

axt_.minorticks_on()
axl_.minorticks_on()

axl_.set_xlabel(L"$j$")

axt_.set_xticks([0, 10, 20, 32])
axl_.set_xticks([0, 10, 20, 32])

axt_.set_xlim(0, 33)
axl_.set_xlim(0, 33)

axt_.set_ylim(-1.15, 0.15)
axl_.set_ylim(-0.15, 1.15)

axt_.set(yticklabels=[])
axl_.set(yticklabels=[])

axt_.set_title(latexstring(L"(b) \ PBC, \ $(\pi/4,\ 3/10)$"), pad=8)

# axt_.annotate("(c)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# axl_.annotate("(d)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

data = load(dataFolder * "N=32_chi=128/ANALYZED/N=32_chi=128_lam=0.7_theta=0.785398163397448.jld2")
# data = load(dataFolder * "N=128_chi=256/ANALYZED/N=128_chi=256_lam=0.5_theta=0.785398163397448.jld2")

println(data["energies"])

mX = -data["opSPT"][1, 1, :]
mY = -data["opSPT"][1, 2, :]
mZ = -data["opSPT"][1, 3, :]
println(numpy.mean(mZ))

p, = axt__.plot(range(1, length(mX)), mX, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axt__.plot(range(1, length(mY)), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axt__.plot(range(1, length(mZ)), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

mX = -data["mags"][1, 1, :]
mY = -data["mags"][1, 2, :]
mZ = -data["mags"][1, 3, :]
println(numpy.mean(mZ))

p, = axl__.plot(range(1, length(mX)), mX, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axl__.plot(range(1, length(mY)), mY,  markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axl__.plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

axt__.set(xticklabels=[])

axt__.minorticks_on()
axl__.minorticks_on()

axl__.set_xlabel(L"$j$")

axt__.set_xticks([0, 10, 20, 32])
axl__.set_xticks([0, 10, 20, 32])

axt__.set_xlim(0, 33)
axl__.set_xlim(0, 33)

axt__.set_ylim(-1.15, 0.15)
axl__.set_ylim(-0.15, 1.15)

axt__.set(yticklabels=[])
axl__.set(yticklabels=[])

axt__.set_title(latexstring(L"(c) \ PBC, \ $(\pi/4,\ 7/10)$"), pad=8)

# axt__.annotate("(e)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# axl__.annotate("(f)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

axt_.legend(handles=ps, loc="lower center", markerscale=1.5, bbox_to_anchor=(0.5, 1.4), borderaxespad=0, ncol=6, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(200)

plt.draw()

if savePlot
	py"createFolderIfMissing"(graphFolder)
	fig.savefig(graphFolder * "spins.pdf")
	fig.clf()
end

plt.close()
