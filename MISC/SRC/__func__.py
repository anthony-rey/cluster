import numpy as np
import pickle
import os as os
import re as re
import math

def loadFile(folder, expression, loadEngines=False):

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

def folderNames(folder):

	return sorted([name for name in os.listdir(folder) if os.path.isdir(os.path.join(folder, name))])

def magnitude(x):

    return int(math.floor(math.log10(abs(x))))

def r2(y, fit):

	ss_res = np.sum((y-fit)**2)
	ss_tot = np.sum((y-np.mean(y))**2)

	return (1 - ss_res/ss_tot)

def findNearest(array, value):

	array = np.asarray(array)
	idx = np.abs(array - value).argmin()

	return idx

def createFolderIfMissing(path):

	if not os.path.exists(path):
		os.makedirs(path)

def saveGraph(fig, path, name):

	createFolderIfMissing(path)
	fig.savefig(os.path.join(path, name) + ".")

def anal(args):

	a = {'eigen': False, 'misc': False}

	a['eigen'] = args.overlap or args.norm or args.ratio or args.mod or args.central or args.complex or args.length or args.sqrt or args.tot or args.norm_ or args.split or args.length_ or args.corr or args.ratio_
	a['misc'] = args.energies or args.entanglement or args.en

	return a

def dictToStr(d):

	string = ""
	for item in d:
		string += item + "=" + str(d[item]) + "_"

	return string[:-1]

def saveFit(s):

	with open("../data/fits.dat", 'a') as file:
		file.write(s)