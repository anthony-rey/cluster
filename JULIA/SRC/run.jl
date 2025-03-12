using ITensors, ITensorMPS
using FileIO
using Random

mutable struct Observer <: AbstractObserver
   energy_tol::Float64
   last_energy::Float64

   H::MPO

   Observer(H, energy_tol=0.0) = new(energy_tol, 0.0, H)
end

function ITensorMPS.checkdone!(o::Observer; kwargs...)
	energy = kwargs[:energy]

	if abs(energy-o.last_energy)/abs(energy) < o.energy_tol
		println("Energy difference less than $(o.energy_tol), stopping DMRG")
		return true
	end

	o.last_energy = energy
	return false
end

function doDMRG(N, chi, lambda, theta, PBCs, k)

	sites = siteinds("S=1/2", N)
	os = OpSum()

	for j=1:N-1
		os += (1-lambda)*cos(theta)*4, "Sx", j, "Sx", j+1
		os += (1-lambda)*sin(theta)*4, "Sy", j, "Sy", j+1
	end

	for j=1:N-3
		os += lambda*cos(theta)*16, "Sz", j, "Sx", j+1, "Sx", j+2, "Sz", j+3
		os += lambda*sin(theta)*16, "Sz", j, "Sy", j+1, "Sy", j+2, "Sz", j+3
	end

	if PBCs

		os += (1-lambda)*cos(theta)*4, "Sx", N, "Sx", 1
		os += (1-lambda)*sin(theta)*4, "Sy", N, "Sy", 1
		
		os += lambda*cos(theta)*16, "Sz", N-2, "Sx", N-1, "Sx", N, "Sz", 1
		os += lambda*sin(theta)*16, "Sz", N-2, "Sy", N-1, "Sy", N, "Sz", 1
		os += lambda*cos(theta)*16, "Sz", N-1, "Sx", N, "Sx", 1, "Sz", 2
		os += lambda*sin(theta)*16, "Sz", N-1, "Sy", N, "Sy", 1, "Sz", 2
		os += lambda*cos(theta)*16, "Sz", N, "Sx", 1, "Sx", 2, "Sz", 3
		os += lambda*sin(theta)*16, "Sz", N, "Sy", 1, "Sy", 2, "Sz", 3

	end
	
	H = MPO(os, sites)
	psi = randomMPS(ComplexF64, sites; linkdims=20)

	nsweeps = 50
	cutoff = [1E-9]
	level = 1
	energy_tol = 1E-12
	eigsolve_krylovdim = 10
	eigsolve_maxiter = 2

	if k>1
		println("----- get state\n \t | k : 1\n")
	end

	energies = []
	bulkEnergies = []

	obs = Observer(H, energy_tol)
	energy, psi = dmrg(H, psi; nsweeps, cutoff, maxdim=chi, observer=obs, outputlevel=level, eigsolve_krylovdim=eigsolve_krylovdim, eigsolve_maxiter=eigsolve_maxiter)

	push!(energies, energy)
	psis = [psi]

	for i=1:k-1

		println("\n----- get state\n\t | k : ", i+1, "\n")

		psi = randomMPS(ComplexF64, sites; linkdims=20)
		obs = Observer(H, energy_tol)
		energy, psi = dmrg(H, psis, psi; nsweeps, cutoff, maxdim=chi, observer=obs, weight=10, outputlevel=level, eigsolve_krylovdim=eigsolve_krylovdim, eigsolve_maxiter=eigsolve_maxiter)

		push!(energies, energy)
		push!(psis, psi)

	end
		 
	return energies, psis

end

nLam = 1
lam = range(0.95, 0.95, nLam)
nTheta = 1
theta = range(pi/4, pi/4, nTheta)

nN = 1
N = range(80, 80, nN)
nChi = 1
chi = range(128, 128, nChi)

PBCs = false
k = 4

for i=1:nLam
	for j=1:nTheta
		for n=1:nN
			for x=1:nChi

				if PBCs
					local dataFolder = "../DATA/PBC/_/N=$(trunc(Int, N[n]))_chi=$(trunc(Int, chi[x]))/"
				else
					local dataFolder = "../DATA/OBC/SCALING_CORR_END_E+DU(1)_SPT/N=$(trunc(Int, N[n]))_chi=$(trunc(Int, chi[x]))/"
				end

				println("\n\t\t start DMRG \n")
				println("\t | N : ", trunc(Int, N[n]))
				println("\t | chi : ", trunc(Int, chi[x]))
				println("\t | lambda : ", round(lam[i]; digits=5), ", theta : ", round(theta[j]; digits=15), "\n")

				dataFilename = "N=$(trunc(Int, N[n]))_chi=$(trunc(Int, chi[x]))_lam=$(round(lam[i]; digits=5))_theta=$(round(theta[j]; digits=15)).jld2"

				if !isfile(dataFolder * dataFilename)

					energies, psis = doDMRG(trunc(Int, N[n]), trunc(Int, chi[x]), round(lam[i]; digits=5), round(theta[j]; digits=15), PBCs, k)

					dict = Dict("psis" => psis, "energies" => energies)
					save(dataFolder * dataFilename, dict)

				else
					println("··············· Already converged")
				end

			end
		end
	end
end
