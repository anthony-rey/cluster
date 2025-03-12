import numpy as np
from ncon import ncon
import scipy.sparse.linalg as ln
import scipy.linalg
from .lin import *

class EDEngine:

	def __init__(self, N, d, PBCs, params, arpack, eigenvectors, saveMax, k, ncv, precEigen):

		self.N = N
		self.d = d
		self.PBCs = PBCs
		self.params = params

		self.arpack = arpack
		self.eigenvectors = eigenvectors
		self.saveMax = saveMax

		self.k = k
		self.ncv = ncv
		self.precEigen = precEigen

	def run(self):
		
		H = Hamiltonian(self.N, self.d, self.PBCs, self.params, self.arpack)

		if self.arpack:
			if self.eigenvectors:
				e, v = ln.eigsh(H, k=self.k, ncv=self.ncv, which='LA', return_eigenvectors=True, tol=self.precEigen)
			else:
				e = ln.eigsh(H, k=self.k, ncv=self.ncv, which='LA', return_eigenvectors=False, tol=self.precEigen)
		
		else:
			if self.eigenvectors:
				e, v = scipy.linalg.eigh(H.getFullMatrix(), eigvals_only=False)
			else:
				e = scipy.linalg.eigh(H.getFullMatrix(), eigvals_only=True)

		ind = np.argsort(e)
		self.E = e[ind][:self.saveMax]
		if self.eigenvectors:
			self.V = v[:, ind][:, :self.saveMax]
		else:
			self.V = None