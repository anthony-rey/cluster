using ITensors, ITensorMPS
using FileIO
using PyCall
using ITensorCorrelators

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

PBCs = false
force = false

compute_entropies = false
compute_entanglement = false
compute_mags = true
compute_corrs = true
compute_corrsSPT = false
compute_evs_2 = false
compute_evs_3 = false
compute_evs_4 = false
compute_evs_6 = false
compute_evs_8 = false
compute_evs_10 = false
compute_opSPT = false
compute_ev_ends = true

r = 96

if PBCs
	dataFolder = "../DATA/PBC/_/"
else
	dataFolder = "../DATA/OBC/SCALING_CORR_END_E+DU(1)_SPT/"
end
folders = py"folderNames"(dataFolder)

# folders = ["N=128_chi=256/"]
for folder in folders

	engines, vals, dataFilenames = py"loadFile"(dataFolder * folder)

	for i=1:length(dataFilenames)
		
		data = 0
		data = load(dataFolder * folder * dataFilenames[i])
		
		println("··· ", dataFilenames[i])

		N = Int(vals[i]["N"])
		k = length(data["energies"])

		quantities = Dict{Any, Any}("energies" => data["energies"])

		if isfile(dataFolder * folder * "ANALYZED/" * dataFilenames[i])

			quantities = load(dataFolder * folder * "ANALYZED/" * dataFilenames[i])
			println("\t already computed : ", keys(quantities))
			println("\t k : ", k)

		end
		
		# quantities["bulkEnergies"] = data["bulkEnergies"]

		if force | !isfile(dataFolder * folder * "ANALYZED/" * dataFilenames[i])

			if compute_entropies | compute_entanglement
				println("\t | entropies/entanglement...")

				entropies = zeros((k, N-1))
				entanglement = Array{Vector{Float64}}(undef, k, N-1)
				for j=1:k
					psi = data["psis"][j]
					for b=1:N-1
						psi = ITensorMPS.orthogonalize!(psi, b)
						U,S,V = svd(psi[b], (linkinds(psi, b-1)..., siteinds(psi, b)...))
						SvN = 0.0
						diag = []
						for l=1:dim(S, 1)
							push!(diag, S[l,l])
							p = S[l,l]^2
							SvN -= p * log(p)
						end
						entropies[j, b] = SvN
						entanglement[j, b] = diag
					end
				end
				quantities["entropies"] = entropies
				quantities["entanglement"] = entanglement
			end

			if compute_mags
				println("\t | mags...")

				mags = zeros((k, 3, N))
				for j=1:k
					mags[j, 1, :] = 2*expect(data["psis"][j], "Sx")
					mags[j, 2, :] = 2*expect(data["psis"][j], "Sy")
					mags[j, 3, :] = 2*expect(data["psis"][j], "Sz")
				end
				quantities["mags"] = mags
			end

			if compute_corrs
				println("\t | corrs...")

				corrs = zeros((k, 3, N, N))
				for j=1:k
					corrs[j, 1, :, :] .= real(4*correlation_matrix(data["psis"][j], "Sx", "Sx"))
					corrs[j, 2, :, :] .= real(4*correlation_matrix(data["psis"][j], "Sy", "Sy"))
					corrs[j, 3, :, :] .= real(4*correlation_matrix(data["psis"][j], "Sz", "Sz"))
				end
				quantities["corrs"] = corrs
			end

			if compute_corrsSPT
				println("\t | corrsSPT...")

				corrsSPT = zeros((k, 3, r))	
				for j=1:k
					corrX = []
					for l=1:r
						push!(corrX, 64*correlator(data["psis"][j], ("Sz", "Sx", "Sz", "Sz", "Sx", "Sz"), [(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1))])[(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1))])
					end
					corrsSPT[j, 1, :] = real(collect(corrX))

					corrY = []
					for l=1:r
						push!(corrY, 64*correlator(data["psis"][j], ("Sz", "Sy", "Sz", "Sz", "Sy", "Sz"), [(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1))])[(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1))])
					end
					corrsSPT[j, 2, :] = real(collect(corrY))
					
					corrZ = []
					for l=1:r
						push!(corrZ, 64*correlator(data["psis"][j], ("Sz", "Sz", "Sz", "Sz", "Sz", "Sz"), [(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1))])[(Int(N/2-r/2-1), Int(N/2-r/2), Int(N/2-r/2+1), Int(N/2-r/2+l-1), Int(N/2-r/2+l), Int(N/2-r/2+l+1))])
					end
					corrsSPT[j, 3, :] = real(collect(corrZ))
				end
				quantities["corrsSPT"] = corrsSPT
			end

			if compute_evs_2
				println("\t | evs_2...")

				evs_2 = zeros((k, 3, N-1))
				for j=1:k
					ev2X = []
					for l=1:N-1
						push!(ev2X, 4*correlator(data["psis"][j], ("Sx", "Sx"), [(l, l+1)])[(l, l+1)])
					end
					evs_2[j, 1, :] = real(collect(ev2X))

					ev2Y = []
					for l=1:N-1
						push!(ev2Y, 4*correlator(data["psis"][j], ("Sy", "Sy"), [(l, l+1)])[(l, l+1)])
					end
					evs_2[j, 2, :] = real(collect(ev2Y))
					
					ev2Z = []
					for l=1:N-1
						push!(ev2Z, 4*correlator(data["psis"][j], ("Sz", "Sz"), [(l, l+1)])[(l, l+1)])
					end
					evs_2[j, 3, :] = real(collect(ev2Z))
				end
				quantities["evs_2"] = evs_2
			end

			if compute_evs_3 && !haskey(quantities, "evs_3")
				println("\t | evs_3...")

				evs_3 = zeros((k, 3, N-2))
				for j=1:k
					ev3X = []
					for l=1:N-2
						push!(ev3X, 4*correlator(data["psis"][j], ("Sz", "Sx", "Sz"), [(l, l+1, l+2)])[(l, l+1, l+2)])
					end
					evs_3[j, 1, :] = real(collect(ev3X))

					ev3Y = []
					for l=1:N-2
						push!(ev3Y, 4*correlator(data["psis"][j], ("Sz", "Sy", "Sz"), [(l, l+1, l+2)])[(l, l+1, l+2)])
					end
					evs_3[j, 2, :] = real(collect(ev3Y))
					
					ev3Z = []
					for l=1:N-2
						push!(ev3Z, 4*correlator(data["psis"][j], ("Sz", "Sz", "Sz"), [(l, l+1, l+2)])[(l, l+1, l+2)])
					end
					evs_3[j, 3, :] = real(collect(ev3Z))
				end
				quantities["evs_3"] = evs_3
			end

			if compute_evs_4
				println("\t | evs_4...")

				evs_4 = zeros((k, 3, N-3))
				for j=1:k
					ev4X = []
					for l=1:N-3
						push!(ev4X, 4*correlator(data["psis"][j], ("Sz", "Sx", "Sx", "Sz"), [(l, l+1, l+2, l+3)])[(l, l+1, l+2, l+3)])
					end
					evs_4[j, 1, :] = real(collect(ev4X))

					ev4Y = []
					for l=1:N-3
						push!(ev4Y, 4*correlator(data["psis"][j], ("Sz", "Sy", "Sy", "Sz"), [(l, l+1, l+2, l+3)])[(l, l+1, l+2, l+3)])
					end
					evs_4[j, 2, :] = real(collect(ev4Y))
					
					ev4Z = []
					for l=1:N-3
						push!(ev4Z, 4*correlator(data["psis"][j], ("Sz", "Sz", "Sz", "Sz"), [(l, l+1, l+2, l+3)])[(l, l+1, l+2, l+3)])
					end
					evs_4[j, 3, :] = real(collect(ev4Z))
				end
				quantities["evs_4"] = evs_4
			end

			if compute_evs_6
				println("\t | evs_6...")

				evs_6 = zeros((k, 3, N-5))
				for j=1:k
					ev6X = []
					for l=1:N-5
						push!(ev6X, 4*correlator(data["psis"][j], ("Sz", "Sx", "Sx", "Sx", "Sx", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5)])[(l, l+1, l+2, l+3, l+4, l+5)])
					end
					evs_6[j, 1, :] = real(collect(ev6X))

					ev6Y = []
					for l=1:N-5
						push!(ev6Y, 4*correlator(data["psis"][j], ("Sz", "Sy", "Sy", "Sy", "Sy", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5)])[(l, l+1, l+2, l+3, l+4, l+5)])
					end
					evs_6[j, 2, :] = real(collect(ev6Y))
					
					ev6Z = []
					for l=1:N-5
						push!(ev6Z, 4*correlator(data["psis"][j], ("Sz", "Sz", "Sz", "Sz", "Sz", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5)])[(l, l+1, l+2, l+3, l+4, l+5)])
					end
					evs_6[j, 3, :] = real(collect(ev6Z))
				end
				quantities["evs_6"] = evs_6
			end

			if compute_evs_8
				println("\t | evs_8...")

				evs_8 = zeros((k, 3, N-7))
				for j=1:k
					ev8X = []
					for l=1:N-7
						push!(ev8X, 4*correlator(data["psis"][j], ("Sz", "Sx", "Sx", "Sx", "Sx", "Sx", "Sx", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7)])[(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7)])
					end
					evs_8[j, 1, :] = real(collect(ev8X))

					ev8Y = []
					for l=1:N-7
						push!(ev8Y, 4*correlator(data["psis"][j], ("Sz", "Sy", "Sy", "Sy", "Sy", "Sy", "Sy", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7)])[(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7)])
					end
					evs_8[j, 2, :] = real(collect(ev8Y))
					
					ev8Z = []
					for l=1:N-7
						push!(ev8Z, 4*correlator(data["psis"][j], ("Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7)])[(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7)])
					end
					evs_8[j, 3, :] = real(collect(ev8Z))
				end
				quantities["evs_8"] = evs_8
			end

			if compute_evs_10
				println("\t | evs_10...")

				evs_10 = zeros((k, 3, N-9))
				for j=1:k
					ev10X = []
					for l=1:N-9
						push!(ev10X, 4*correlator(data["psis"][j], ("Sz", "Sx", "Sx", "Sx", "Sx", "Sx", "Sx", "Sx", "Sx", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9)])[(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9)])
					end
					evs_10[j, 1, :] = real(collect(ev10X))

					ev10Y = []
					for l=1:N-9
						push!(ev10Y, 4*correlator(data["psis"][j], ("Sz", "Sy", "Sy", "Sy", "Sy", "Sy", "Sy", "Sy", "Sy", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9)])[(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9)])
					end
					evs_10[j, 2, :] = real(collect(ev10Y))
					
					ev10Z = []
					for l=1:N-9
						push!(ev10Z, 4*correlator(data["psis"][j], ("Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz", "Sz"), [(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9)])[(l, l+1, l+2, l+3, l+4, l+5, l+6, l+7, l+8, l+9)])
					end
					evs_10[j, 3, :] = real(collect(ev10Z))
				end
				quantities["evs_10"] = evs_10
			end

			if compute_opSPT
				println("\t | opSPT...")

				if PBCs==false
					opSPT = zeros((k, 3, N-2))
					for j=1:k
						opSPTX = []
						for l=1:N-2
							push!(opSPTX, 8*correlator(data["psis"][j], ("Sz", "Sx", "Sz"), [(l, l+1, l+2)])[(l, l+1, l+2)])
						end
						opSPT[j, 1, :] = real(collect(opSPTX))

						opSPTY = []
						for l=1:N-2
							push!(opSPTY, 8*correlator(data["psis"][j], ("Sz", "Sy", "Sz"), [(l, l+1, l+2)])[(l, l+1, l+2)])
						end
						opSPT[j, 2, :] = real(collect(opSPTY))

						opSPTZ = []
						for l=1:N-2
							push!(opSPTZ, 8*correlator(data["psis"][j], ("Sz", "Sz", "Sz"), [(l, l+1, l+2)])[(l, l+1, l+2)])
						end
						opSPT[j, 3, :] = real(collect(opSPTZ))
					end
					quantities["opSPT"] = opSPT
				elseif PBCs==true
					opSPT = zeros((k, 3, N))
					for j=1:k
						opSPTX = []
						push!(opSPTX, 8*correlator(data["psis"][j], ("Sz", "Sx", "Sz"), [(N, 1, 2)])[(N, 1, 2)])
						for l=1:N-2
							push!(opSPTX, 8*correlator(data["psis"][j], ("Sz", "Sx", "Sz"), [(l, l+1, l+2)])[(l, l+1, l+2)])
						end
						push!(opSPTX, 8*correlator(data["psis"][j], ("Sz", "Sx", "Sz"), [(N-1, N, 1)])[(N-1, N, 1)])
						opSPT[j, 1, :] = real(collect(opSPTX))

						opSPTY = []
						push!(opSPTY, 8*correlator(data["psis"][j], ("Sz", "Sy", "Sz"), [(N, 1, 2)])[(N, 1, 2)])
						for l=1:N-2
							push!(opSPTY, 8*correlator(data["psis"][j], ("Sz", "Sy", "Sz"), [(l, l+1, l+2)])[(l, l+1, l+2)])
						end
						push!(opSPTY, 8*correlator(data["psis"][j], ("Sz", "Sy", "Sz"), [(N-1, N, 1)])[(N-1, N, 1)])
						opSPT[j, 2, :] = real(collect(opSPTY))

						opSPTZ = []
						push!(opSPTZ, 8*correlator(data["psis"][j], ("Sz", "Sz", "Sz"), [(N, 1, 2)])[(N, 1, 2)])
						for l=1:N-2
							push!(opSPTZ, 8*correlator(data["psis"][j], ("Sz", "Sz", "Sz"), [(l, l+1, l+2)])[(l, l+1, l+2)])
						end
						push!(opSPTZ, 8*correlator(data["psis"][j], ("Sz", "Sz", "Sz"), [(N-1, N, 1)])[(N-1, N, 1)])
						opSPT[j, 3, :] = real(collect(opSPTZ))
					end
					quantities["opSPT"] = opSPT
				end
			end

			if compute_ev_ends
				println("\t | ev_ends...")

				ev_ends = zeros((k, 2))
				for j=1:k
					ev4X = 16*correlator(data["psis"][j], ("Sx", "Sz", "Sz", "Sx"), [(1, 2, N-1, N)])[(1, 2, N-1, N)]
					ev_ends[j, 1] = real(ev4X)

					ev4Y = 16*correlator(data["psis"][j], ("Sy", "Sz", "Sz", "Sy"), [(1, 2, N-1, N)])[(1, 2, N-1, N)]
					ev_ends[j, 2] = real(ev4Y)
				end
				quantities["ev_ends"] = ev_ends
			end

			save(dataFolder * folder * "ANALYZED/" * dataFilenames[i], quantities)

		end
	end
end