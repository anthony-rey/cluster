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

nE = 6

savePlot = true

Ns = []
diffsNs = []

# colors = ["red","palered","lightred","blue","paleblue","lightblue","orange","paleorange","lightorange"]
colors = ["#b1257b","#da8ba7","#dfbed4","#517da3","#8ea7c3","#b4d5e9","#f9a515","#f2cd8d","#ffe4c9"]

fig = plt.figure(figsize=(16.5, 7))
axt = plt.subplot2grid((4, 4), (0, 0))
axl = plt.subplot2grid((4, 4), (1, 0))
axt_ = plt.subplot2grid((4, 4), (0, 1))
axl_ = plt.subplot2grid((4, 4), (1, 1))
axt__ = plt.subplot2grid((4, 4), (0, 2))
axl__ = plt.subplot2grid((4, 4), (1, 2))
axt___ = plt.subplot2grid((4, 4), (0, 3))
axl___ = plt.subplot2grid((4, 4), (1, 3))
ax = plt.subplot2grid((4, 4), (3, 0), colspan=2)
ax_ = plt.subplot2grid((4, 4), (3, 2), colspan=2)
plt.subplots_adjust(wspace=0.15)

py"""
import numpy 

def f(l, a, b, c):
	return a*(l**(-b)) + c
def g(l, a, b, c):
	return a*numpy.exp(-l/b) + c
def oz(l, a, b, c, d):
	return a*numpy.cos(2*numpy.pi*b*l + c) + d
"""

ps = []
data = load(dataFolder * "N=32_chi=128/ANALYZED/N=32_chi=128_lam=0.5_theta=0.785398163397448.jld2")

mX = data["opSPT"][1, 1, :]
mY = data["opSPT"][1, 2, :]
mZ = data["opSPT"][1, 3, :]
# println(numpy.mean(mZ))

r = range(1, length(mZ))
mZ = mZ[r]
poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r), numpy.abs(mZ), [0.1, 0.3, -1, 0.2])
perroz = numpy.sqrt(numpy.diag(pcov))
# println(poptoz, perroz)

p, = axt.plot(range(2, length(mX)+1), mX, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[1], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{X}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = axt.plot(range(2, length(mY)+1), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{Y}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)
p, = axt.plot(range(2, length(mZ)+1), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], label=latexstring(L"$\langle \widehat{Z}^{\,}_{j-1} \, \widehat{Z}^{\,}_{j} \, \widehat{Z}^{\,}_{j+1} \rangle^{\,}_{\Psi}$"), zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
push!(ps, p)

mX = data["mags"][1, 1, :]
mY = data["mags"][1, 2, :]
mZ = data["mags"][1, 3, :]
# println(numpy.mean(mZ))

r = range(1, length(mZ))
mZ = mZ[r]
poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r), numpy.abs(mZ), [0.1, 0.3, -1, 0.2])
perroz = numpy.sqrt(numpy.diag(pcov))
# println(poptoz, perroz)

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

axt.set_title(latexstring(L"(a) \ OBC, \ $(\pi/4,\ 1/2)$"), pad=8)

# axt.annotate("(a)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# axl.annotate("(b)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

data = load(dataFolder * "N=32_chi=128/ANALYZED/N=32_chi=128_lam=0.3_theta=0.785398163397448.jld2")

mX = data["opSPT"][1, 1, :]
mY = data["opSPT"][1, 2, :]
mZ = data["opSPT"][1, 3, :]
# println(numpy.mean(mZ))

r = range(1, length(mZ))
mZ = mZ[r]
poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r), numpy.abs(mZ), [0.1, 0.3, -1, 0.2])
perroz = numpy.sqrt(numpy.diag(pcov))
# println(poptoz, perroz)

p, = axt_.plot(range(2, length(mX)+1), mX, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axt_.plot(range(2, length(mY)+1), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axt_.plot(range(2, length(mZ)+1), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

mX = data["mags"][1, 1, :]
mY = data["mags"][1, 2, :]
mZ = data["mags"][1, 3, :]
# println(numpy.mean(mZ))

r = range(1, length(mZ))
mZ = mZ[r]
poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r), numpy.abs(mZ), [0.1, 0.3, -1, 0.2])
perroz = numpy.sqrt(numpy.diag(pcov))
# println(poptoz, perroz)

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

axt_.set_title(latexstring(L"(b) \ OBC, \ $(\pi/4,\ 3/10)$"), pad=8)

# axt_.annotate("(c)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# axl_.annotate("(d)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

data = load(dataFolder * "N=32_chi=128/ANALYZED/N=32_chi=128_lam=0.7_theta=0.785398163397448.jld2")

mX = data["opSPT"][1, 1, :]
mY = data["opSPT"][1, 2, :]
mZ = data["opSPT"][1, 3, :]
# println(numpy.mean(mZ))

r = range(1, length(mZ))
mZ = mZ[r]
poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r), numpy.abs(mZ), [0.1, 0.3, -1, 0.2])
perroz = numpy.sqrt(numpy.diag(pcov))
# println(poptoz, perroz)

p, = axt__.plot(range(2, length(mX)+1), mX, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axt__.plot(range(2, length(mY)+1), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axt__.plot(range(2, length(mZ)+1), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

mX = data["mags"][1, 1, :]
mY = data["mags"][1, 2, :]
mZ = data["mags"][1, 3, :]
# println(numpy.mean(mZ))

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

axt__.set_title(latexstring(L"(c) \ OBC, \ $(\pi/4,\ 7/10)$"), pad=8)

# axt__.annotate("(e)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# axl__.annotate("(f)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

data = load(dataFolder * "N=32_chi=128/ANALYZED/N=32_chi=128_lam=0.5_theta=0.685398163397448.jld2")

mX = -data["opSPT"][1, 1, :]
mY = -data["opSPT"][1, 2, :]
mZ = -data["opSPT"][1, 3, :]
# println(numpy.mean(mZ))

r = range(1, length(mZ))
mZ = mZ[r]
poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r), numpy.abs(mZ), [0.1, 0.3, -1, 0.2])
perroz = numpy.sqrt(numpy.diag(pcov))
# println(poptoz, perroz)

p, = axt___.plot(range(2, length(mX)+1), mX, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axt___.plot(range(2, length(mY)+1), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axt___.plot(range(2, length(mZ)+1), mZ, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

mX = -data["mags"][1, 1, :]
mY = -data["mags"][1, 2, :]
mZ = -data["mags"][1, 3, :]
# println(numpy.mean(mZ))

r = range(1, length(mZ))
mZ = mZ[r]
poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r), numpy.abs(mZ), [0.1, 0.3, -1, 0.2])
perroz = numpy.sqrt(numpy.diag(pcov))
# println(poptoz, perroz)

p, = axl___.plot(range(1, length(mX)), mX, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axl___.plot(range(1, length(mY)), mY,  markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = axl___.plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

axt___.set(xticklabels=[])

axt___.minorticks_on()
axl___.minorticks_on()

axl___.set_xlabel(L"$j$")

axt___.set_xticks([0, 10, 20, 32])
axl___.set_xticks([0, 10, 20, 32])

axt___.set_xlim(0, 33)
axl___.set_xlim(0, 33)

axt___.set_ylim(-1.15, 0.15)
axl___.set_ylim(-0.15, 1.15)

axt___.set(yticklabels=[])
axl___.set(yticklabels=[])

axt___.set_title(latexstring(L"(d) \ OBC, \ $(\pi/4-0.1,\ 1/2)$"), pad=8)

# axt___.annotate("(g)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)
# axl___.annotate("(h)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

data = load(dataFolder * "N=256_chi=128/ANALYZED/N=256_chi=128_lam=0.5_theta=0.785398163397448.jld2")

mX = numpy.abs(data["mags"][1, 1, :])
mY = numpy.abs(data["mags"][1, 2, :])
mZ = numpy.abs(data["mags"][1, 3, :])
# println(numpy.mean(mZ))

r = range(1, length(mZ))
mZ = mZ[r]
poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r), numpy.abs(mZ), [0.1, 0.3, -1, 0.2])
perroz = numpy.sqrt(numpy.diag(pcov))
# println(poptoz, perroz)

p, = ax.plot(range(1, length(mX)), mX, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax.plot(range(1, length(mY)), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax.plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

ax.minorticks_on()

ax.set_xlabel(L"$j$")

# ax.set_xticks([0, 10, 20, 32])

ax.set_xlim(0, 257)

ax.set_ylim(-0.15, 1.15)

ax.set_title(latexstring(L"(e) \ OBC, \ $(\pi/4,\ 1/2)$"), pad=8)

# ax.annotate("(i)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

data = load(dataFolder * "N=256_chi=128/ANALYZED/N=256_chi=128_lam=0.7_theta=0.785398163397448.jld2")

mX = numpy.abs(data["mags"][1, 1, :])
mY = numpy.abs(data["mags"][1, 2, :])
mZ = numpy.abs(data["mags"][1, 3, :])
println(numpy.mean(mZ))

r = range(1, length(mZ))
mZ = mZ[r]
poptoz, pcov = scipy.optimize.curve_fit(py"oz", collect(r), numpy.abs(mZ), [0.01, 0.325, -1, 0.3])
perroz = numpy.sqrt(numpy.diag(pcov))
println(poptoz, perroz)

p, = ax_.plot(range(1, length(mX)), mX, markersize=4, lw=0.8, markeredgewidth=2, marker=2, color=colors[1], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_.plot(range(1, length(mY)), mY, markersize=4, lw=0.8, markeredgewidth=2, marker=3, color=colors[4], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))
p, = ax_.plot(range(1, length(mZ)), mZ, markersize=3, lw=0.8, markeredgewidth=2, marker="x", color=colors[7], zorder=100)
p.set_marker(matplotlib.markers.MarkerStyle(p.get_marker(), capstyle="round"))

ax_.set(yticklabels=[])

ax_.minorticks_on()

ax_.set_xlabel(L"$j$")

# ax_.set_xticks([0, 10, 20, 32])

ax_.set_xlim(0, 257)

ax_.set_ylim(-0.15, 1.15)

ax_.set_title(latexstring(L"(f) \ OBC, \ $(\pi/4,\ 7/10)$"), pad=8)

# ax_.annotate("(j)", xy=(1.01, 0.45), xycoords="axes fraction", fontsize=16)

ax.legend(handles=ps, loc="lower center", markerscale=1.5, bbox_to_anchor=(1.04, 1.33), borderaxespad=0, ncol=6, labelspacing=0.8, columnspacing=0.8, handletextpad=0.6, handlelength=1.8).set_zorder(200)

plt.draw()

if savePlot
	py"createFolderIfMissing"(graphFolder)
	fig.savefig(graphFolder * "spins.pdf")
	fig.clf()
end

plt.close()