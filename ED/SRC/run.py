import numpy as np
from ncon import ncon
import scipy.sparse.linalg as ln
import argparse
from __func__ import *
import EDRoutine.lin
import EDRoutine.ed

parser = argparse.ArgumentParser()
args = parser.parse_args()

dataFolder = "../DATA/test/"

nLam = 1
lam = np.round(np.linspace(0.95, 0.95, nLam), 5)
nTheta = 1
theta = np.round(np.linspace(np.pi/4, np.pi/4, nTheta), 15)

nN = 1
N = np.linspace(12, 12, nN, dtype=int)

d = 2

PBCs = 0
arpack = 0

eigenvectors = 1
saveMax = 100

k = 10
ncv = 2*k
precEigen = 1e-15

for i in range(nLam):
	for j in range(nTheta):
		for n in N:
			folderRun = f"pbc={PBCs}_arpack={arpack}_N={n}"

			params = {'lam': lam[i], 'theta': theta[j]}

			dataFilename = f"pbc={PBCs}_arpack={arpack}_N={n}_{dictToStr(params)}.dat"
			filename = os.path.join(dataFolder, folderRun, dataFilename)

			if not os.path.exists(os.path.join(dataFolder, folderRun, dataFilename)):
				
				if not os.path.exists(os.path.join(dataFolder, folderRun)):
					os.makedirs(os.path.join(dataFolder, folderRun))
					
				engine = EDRoutine.ed.EDEngine(n, d, PBCs, params, arpack, eigenvectors, saveMax, k, ncv, precEigen)
				engine.run()

				print(f"{engine.E[1]-engine.[0]:.15g}")
				# print(f"{engine.E[-2]-engine.[-1]:.15g}")
				# print(engine.E)

				X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
				Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
				Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
				I = np.eye(2)

				# psi = engine.V[:, 2]
				# mag = []
				# for i in range(n):
				# 	legLinks = [[2, 1], [3, 1, 4], [3, 2, 4]]
				# 	s = ncon([X.reshape(2, 2), psi.reshape(2**i, 2, 2**(n-i-1)), psi.conj().reshape(2**i, 2, 2**(n-i-1))], legLinks)
				# 	mag.append(s)
				# print(mag)
				# mag = []
				# for i in range(n):
				# 	legLinks = [[2, 1], [3, 1, 4], [3, 2, 4]]
				# 	s = ncon([Y.reshape(2, 2), psi.reshape(2**i, 2, 2**(n-i-1)), psi.conj().reshape(2**i, 2, 2**(n-i-1))], legLinks)
				# 	mag.append(s)
				# print(mag)
				# mag = []
				# for i in range(n):
				# 	legLinks = [[2, 1], [3, 1, 4], [3, 2, 4]]
				# 	s = ncon([Z.reshape(2, 2), psi.reshape(2**i, 2, 2**(n-i-1)), psi.conj().reshape(2**i, 2, 2**(n-i-1))], legLinks)
				# 	mag.append(s)
				# print(mag)
						
				# with open(filename, 'wb') as file:
				# 	pickle.dump(engine, file)

				print(f"--- finished", dataFilename)

			else:

				print(f"--- already done", dataFilename)