import numpy as np
from ncon import ncon
import scipy.sparse.linalg as ln

class Hamiltonian(ln.LinearOperator):

	def __init__(self, N, d, PBCs, params, arpack):

		self.N = N
		self.d = d
		self.PBCs = PBCs
		self.params = params
		self.arpack = arpack

		self.even = 1

		self.initSystem()

		self.dtype = self.s[0].dtype
		self.shape = (self.d**self.N, self.d**self.N)

	def initSystem(self):

		X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
		Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
		Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
		I = np.eye(2)

		self.s = np.array([I, X, Y, Z])

		if self.arpack:

			localH = (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(X, np.kron(X, np.kron(I, I)))/3
			localH += (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(I, np.kron(X, np.kron(X, I)))/3
			localH += (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(I, np.kron(I, np.kron(X, X)))/3
			localH += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(Y, np.kron(Y, np.kron(I, I)))/3
			localH += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(I, np.kron(Y, np.kron(Y, I)))/3
			localH += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(I, np.kron(I, np.kron(Y, Y)))/3
			localH += self.params['lam']*np.cos(self.params['theta'])*np.kron(Z, np.kron(X, np.kron(X, Z)))
			localH += self.params['lam']*np.sin(self.params['theta'])*np.kron(Z, np.kron(Y, np.kron(Y, Z)))

			self.localHs = [localH.copy() for i in range(self.N)]

			if not self.PBCs:

				self.localHs[0] = (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(X, np.kron(X, np.kron(I, I)))
				self.localHs[0] += (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(I, np.kron(X, np.kron(X, I)))/2
				self.localHs[0] += (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(I, np.kron(I, np.kron(X, X)))/3
				self.localHs[0] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(Y, np.kron(Y, np.kron(I, I)))
				self.localHs[0] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(I, np.kron(Y, np.kron(Y, I)))/2
				self.localHs[0] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(I, np.kron(I, np.kron(Y, Y)))/3
				self.localHs[0] += self.params['lam']*np.cos(self.params['theta'])*np.kron(Z, np.kron(X, np.kron(X, Z)))
				self.localHs[0] += self.params['lam']*np.sin(self.params['theta'])*np.kron(Z, np.kron(Y, np.kron(Y, Z)))

				self.localHs[1] = (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(X, np.kron(X, np.kron(I, I)))/2
				self.localHs[1] += (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(I, np.kron(X, np.kron(X, I)))/3
				self.localHs[1] += (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(I, np.kron(I, np.kron(X, X)))/3
				self.localHs[1] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(Y, np.kron(Y, np.kron(I, I)))/2
				self.localHs[1] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(I, np.kron(Y, np.kron(Y, I)))/3
				self.localHs[1] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(I, np.kron(I, np.kron(Y, Y)))/3
				self.localHs[1] += self.params['lam']*np.cos(self.params['theta'])*np.kron(Z, np.kron(X, np.kron(X, Z)))
				self.localHs[1] += self.params['lam']*np.sin(self.params['theta'])*np.kron(Z, np.kron(Y, np.kron(Y, Z)))

				self.localHs[-4] = (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(X, np.kron(X, np.kron(I, I)))/3
				self.localHs[-4] += (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(I, np.kron(X, np.kron(X, I)))/2
				self.localHs[-4] += (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(I, np.kron(I, np.kron(X, X)))
				self.localHs[-4] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(Y, np.kron(Y, np.kron(I, I)))/2
				self.localHs[-4] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(I, np.kron(Y, np.kron(Y, I)))/3
				self.localHs[-4] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(I, np.kron(I, np.kron(Y, Y)))
				self.localHs[-4] += self.params['lam']*np.cos(self.params['theta'])*np.kron(Z, np.kron(X, np.kron(X, Z)))
				self.localHs[-4] += self.params['lam']*np.sin(self.params['theta'])*np.kron(Z, np.kron(Y, np.kron(Y, Z)))

				self.localHs[-5] = (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(X, np.kron(X, np.kron(I, I)))/3
				self.localHs[-5] += (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(I, np.kron(X, np.kron(X, I)))/3
				self.localHs[-5] += (1-self.params['lam'])*np.cos(self.params['theta'])*np.kron(I, np.kron(I, np.kron(X, X)))/2
				self.localHs[-5] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(Y, np.kron(Y, np.kron(I, I)))/3
				self.localHs[-5] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(I, np.kron(Y, np.kron(Y, I)))/3
				self.localHs[-5] += (1-self.params['lam'])*np.sin(self.params['theta'])*np.kron(I, np.kron(I, np.kron(Y, Y)))/2
				self.localHs[-5] += self.params['lam']*np.cos(self.params['theta'])*np.kron(Z, np.kron(X, np.kron(X, Z)))
				self.localHs[-5] += self.params['lam']*np.sin(self.params['theta'])*np.kron(Z, np.kron(Y, np.kron(Y, Z)))

		# up =  np.array([1] + [0 for i in range(self.d-1)]).reshape(2, 1)
		# stateUp =  np.array([1] + [0 for i in range(self.d-1)]).reshape(2, 1)
		# for i in range(1, self.N):
		# 	legLinks = [[-1, 1], [-2, 1]]
		# 	stateUp = ncon([stateUp, up], legLinks).reshape(self.d**(i+1), 1)
		# stateUp = stateUp.reshape(self.d**self.N)

		# down =  np.array([0 for i in range(self.d-1)] + [1]).reshape(2, 1)
		# stateDown =  np.array([1] + [0 for i in range(self.d-1)]).reshape(2, 1)
		# for i in range(1, self.N):
		# 	legLinks = [[-1, 1], [-2, 1]]
		# 	stateDown = ncon([stateDown, down], legLinks).reshape(self.d**(i+1), 1)
		# stateDown = stateDown.reshape(self.d**self.N)
		
		# if self.even:
		# 	self.initState = 0.5*(stateUp+stateDown)
		# else:
		# 	self.initState = 0.5*(stateUp-stateDown)

		# np.random.seed(111)
		# up1 =  np.random.rand(self.d).reshape(self.d, 1)
		# up2 =  -up1.copy()
		# stateUp =  up1.copy()
		# for i in range(1, self.N):
		# 	legLinks = [[-1, 1], [-2, 1]]
		# 	if i%2==1:
		# 		stateUp = ncon([stateUp, up2], legLinks).reshape(self.d**(i+1), 1)
		# 	else:
		# 		stateUp = ncon([stateUp, up1], legLinks).reshape(self.d**(i+1), 1)
		# stateUp = stateUp.reshape(self.d**self.N)

		# np.random.seed(222)
		# down1 = np.random.rand(self.d).reshape(self.d, 1)
		# down2 =  -down1.copy()
		# stateDown = down1.copy()
		# for i in range(1, self.N):
		# 	legLinks = [[-1, 1], [-2, 1]]
		# 	if i%2==1:
		# 		stateDown = ncon([stateDown, down2], legLinks).reshape(self.d**(i+1), 1)
		# 	else:
		# 		stateDown = ncon([stateDown, down1], legLinks).reshape(self.d**(i+1), 1)
		# stateDown = stateDown.reshape(self.d**self.N)

		# if self.even:
		# 	self.initState = 0.5*(stateUp+stateDown)
		# else:
		# 	self.initState = 0.5*(stateUp-stateDown)
		
		# np.random.seed(111)
		# up1 =  np.array([1] + [0 for i in range(self.d-1)]).reshape(2, 1)
		# up2 =  np.array([0 for i in range(self.d-1)] + [1]).reshape(2, 1)
		# stateUp =  up1.copy()
		# for i in range(1, self.N):
		# 	legLinks = [[-1, 1], [-2, 1]]
		# 	if i%2==1:
		# 		stateUp = ncon([stateUp, up2], legLinks).reshape(self.d**(i+1), 1)
		# 	else:
		# 		stateUp = ncon([stateUp, up1], legLinks).reshape(self.d**(i+1), 1)
		# stateUp = stateUp.reshape(self.d**self.N)

		# np.random.seed(222)
		# down1 = np.array([0 for i in range(self.d-1)] + [1]).reshape(2, 1)
		# down2 =  np.array([1] + [0 for i in range(self.d-1)]).reshape(2, 1)
		# stateDown = down1.copy()
		# for i in range(1, self.N):
		# 	legLinks = [[-1, 1], [-2, 1]]
		# 	if i%2==1:
		# 		stateDown = ncon([stateDown, down2], legLinks).reshape(self.d**(i+1), 1)
		# 	else:
		# 		stateDown = ncon([stateDown, down1], legLinks).reshape(self.d**(i+1), 1)
		# stateDown = stateDown.reshape(self.d**self.N)

		# if self.even:
		# 	self.initState = 1/np.sqrt(2)*(stateUp+stateDown)
		# else:
		# 	self.initState = 1/np.sqrt(2)*(stateUp-stateDown)
		

	def getFullSigma(self, i, j):

		return np.kron(np.identity(self.d**(j%self.N)), np.kron(self.s[i], np.identity(self.d**((self.N-j-1)%self.N))))

	def getFullMatrix(self):

		H = np.zeros((self.d**self.N, self.d**self.N), dtype=np.complex128)

		for i in range(self.N-3+3*self.PBCs):

			H += self.params['lam']*np.cos(self.params['theta'])*self.getFullSigma(3, i)@self.getFullSigma(1, i+1)@self.getFullSigma(1, i+2)@self.getFullSigma(3, i+3)
			H += self.params['lam']*np.sin(self.params['theta'])*self.getFullSigma(3, i)@self.getFullSigma(2, i+1)@self.getFullSigma(2, i+2)@self.getFullSigma(3, i+3)

		for i in range(self.N-1+self.PBCs):

			H += (1-self.params['lam'])*np.cos(self.params['theta'])*self.getFullSigma(1, i)@self.getFullSigma(1, i+1)
			H += (1-self.params['lam'])*np.sin(self.params['theta'])*self.getFullSigma(2, i)@self.getFullSigma(2, i+1)

		return H

	def _matvec(self, stateIn):

		stateOut = np.zeros(stateIn.size, dtype=np.complex128)

		for i in range(self.N-3):
			
			legLinks = [[-2, 1], [-1, 1, -3]]
			stateOut += ncon([self.localHs[i].reshape(self.d**4, self.d**4), stateIn.reshape(self.d**i, self.d**4, self.d**(self.N-i-4))], legLinks).reshape(self.d**self.N)	

		if self.PBCs:
			
			legLinks = [[-2, 2, 1], [1, -1, 2]]
			stateOut += ncon([self.localHs[-3].reshape(self.d**4, self.d**3, self.d), stateIn.reshape(self.d, self.d**(self.N-4), self.d**3)], legLinks).reshape(self.d**self.N)
			stateOut += ncon([self.localHs[-2].reshape(self.d**4, self.d**2, self.d**2), stateIn.reshape(self.d**2, self.d**(self.N-4), self.d**2)], legLinks).reshape(self.d**self.N)
			stateOut += ncon([self.localHs[-1].reshape(self.d**4, self.d, self.d**3), stateIn.reshape(self.d**3, self.d**(self.N-4), self.d)], legLinks).reshape(self.d**self.N)

		return stateOut