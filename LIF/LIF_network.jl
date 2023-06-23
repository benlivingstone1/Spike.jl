using Plots

# Create an alpha function
# larger β increases rate of decay
function alpha(β)
    T = 10
    dt = 0.01
    steps = Int(T / dt)
    α = Vector{Float64}(undef, steps + 1)

    for i = 1:steps
        α[i] = β * (ℯ^(-β * (i * dt)))
    end
    
    return α
end

function alpha(β, t)
    α = β * (ℯ*(-β * t))
    return α
end

# function delta(spike_time, T, dt)
#     t = 0:dt:T
#     steps = Int(T / dt)
#     δ = zeros(steps + 1)
#     δ[Int(spike_time / dt)] = 1

#     return δ
# end

function delta(t, tl)
    ret = Vector{Int64}(undef, length(tl))
    for i = 1:size(tl)[1]
        dif = t - tl[i]
        if dif == 0
            ret[i] = 1
        else
            ret[i] = 0
        end
    end
    return ret
end

# α1 = alpha(1)
# println(length(α1))

# t = 0:0.01:10

# d1 = delta(5, 10, 0.01)
# print(length(d1))
# plot(t, d1.*α1)

# We want to model the following equation: 
#
# dv/dt = I - v + Σ(j,m)[J/N * α(t - tm)] - Σ(l)[δ(t-tl)]
# Σ(j,m)[J/N * α(t - tm)] = Σ(j)Σ(m)[J/N * α(t - tm)]
#
# We will use Euler's method to solve for Δv at each time step
# Euler's method: v(t+1) = v(t) + dt*(dv/dt)

function yPrime(index, I, v, J, N, dt, step, tl, s)
    tl = tl .* dt  # Times of all previous firings of neuron i
    t = step * dt

    # Σ_jm = sum((J/N) * alpha(1, t))

    series_jm = zeros(N, 1)

    for j in 1:size(J)[1]
        jn = J[j] / N
        j_spike = findall(s[j, :]) .* dt
        α = zeros(size(s[j,:])[1], 1)
        i = 1
        for tm in j_spike
            α[i] = alpha(0.25, t-tm)
            i +=1
        end
        series_jm[j] = sum(jn * j_spike)
    end

    Σ_jm = sum(series_jm)

    Σ_δ = sum(delta(t, tl))

    dvdt = I - v + Σ_jm - Σ_δ

    return dvdt

end



numNeurons = 10

# Define weights matrix, j
J = rand(numNeurons, numNeurons) .* 0.5
# Define neuron paramters 
Vthresh = 1.0
V0 = 0.0
I = 0.25

# Define time range
T = 10
dt = 0.1
t = [i for i = 0:dt:T]

# Initialize voltage and spike matrices
v = zeros(numNeurons, Int(T/dt)+1)  # Voltage for each neuron for each dt
s = BitArray(undef, numNeurons, Int(T/dt)+1)  # Boolean value representing spikes for each neuron at each dt

for step = 1:Int(T/dt)
    for i=1:numNeurons
        if v[i, step] >= Vthresh
            # A spike occurred
            s[i, step] = 1
            # Reset v to V0
            v[i, step + 1] = V0
        else
            # Calculate change in v 
            all_spikes = findall(s[i, :])  # indices (time step) of all spikes for neuron i
            tls = reverse(all_spikes)  # reverse list to get firings in most recent order 
            dv = dt * yPrime(i, I, v[i, step], J[i, :], numNeurons, dt, step, tls, s)
            # Apply dv to v 
            v[i, step+1] = v[i, step] + dv
            # No spike occurred
            s[i, step] = 0
        end
    end
end

plot(t, v[1, :])
plot!(t, s[1, :])


