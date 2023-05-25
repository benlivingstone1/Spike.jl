using Plots

Ne = 800
Ni = 200
re = rand(Ne)
ri = rand(Ni)

a = [0.02*ones(Ne); 0.02 .+ 0.08*ri]
b = [0.2*ones(Ne); 0.25 .- 0.05*ri]
c = [-65 .+ 15*re.^2; -65*ones(Ni)]
d = [8 .- 6*re.^2; 2*ones(Ni)]
S = [0.5*rand(Ne+Ni, Ne) -rand(Ne+Ni, Ni)]

v = -65*ones(Ne+Ni)  # Initial values of v
u = b.*v  # Initial values of u

firings = Tuple{Int, Int}[]  # spike timings

for t = 1:1000  # simulation of 1000 ms
    I = [5*randn(Ne); 2*randn(Ni)]  # thalamic input
    fired = findall(v .>= 30)  # indices of spikes
    append!(firings, [(t, f) for f in fired])
    v[fired] = c[fired]
    u[fired] .= u[fired] .+ d[fired]
    I .= I .+ sum(S[:, fired], dims=2)
    v .= v .+ 0.5*(0.04*v.^2 .+ 5*v .+ 140 .- u .+ I)  # step 0.5 ms
    v .= v .+ 0.5*(0.04*v.^2 .+ 5*v .+ 140 .- u .+ I)  # for numerical stability
    u .= u .+ a.*(b.*v .- u)
end

plot([spike[1] for spike in firings], [spike[2] for spike in firings], marker = ".",legend = false)
