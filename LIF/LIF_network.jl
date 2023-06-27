using Gadfly, LinearAlgebra, DataFrames

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

# ******************************
# START OF MODEL:
# ******************************

# Define number of neurons in the model 
numNeurons = 100

# Define weights matrix, J
J = rand(numNeurons, numNeurons)  # Random weights between all neurons  
J = J[:, 1:Int(numNeurons * 0.8)] .* 0.5  # excitatory neurons 
J = J[:, Int(numNeurons * 0.8)+1:end] .* -2.0  # inhibitory neurons
J[diagind(J)] .= 0.0  # no self-excitation

# Define neuron paramters 
Vthresh = 1.0  # Threshold voltage
V0 = 0.0  # Initial / Reset voltage
I = 1.3  # DC input current 

# Define time range
# 10 seconds total with time steps of 0.01s 
T = 10 
dt = 0.01  
t = [i for i = 0:dt:T]

# Initialize voltage and spike matrices
v = zeros(numNeurons, Int(T/dt)+1)  # Voltage for each neuron for each dt
s = BitArray(undef, numNeurons, Int(T/dt)+1)  # Boolean value representing spikes for each neuron at each dt
spikes = Matrix{Float64}(undef, numNeurons, Int(T/dt))  # Matrix of firing Times

for step = 1:Int(T/dt)
    for i=1:numNeurons
        if v[i, step] >= Vthresh
            # A spike occurred
            s[i, step] = 1
            spikes[i, step] = step * dt
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
            spikes[i, step] = NaN
        end
    end
end

# ******************************
# RASTER PLOT OF SIMULATION:
# ******************************

df = DataFrame(spikes, :auto)

# Initialize an empty vector to store the firing points
firing_times = []
firing_index = []

# Iterate over the columns (time steps) of the DataFrame
count = 0

for col in eachcol(df)
    time_step = count * dt  # Get the column name (time step)
    global count += 1
    
    # Iterate over the rows (neurons) of the DataFrame
    for (i, value) in enumerate(col)
        if !isnan(value)
            neuron_index = i  # Get the neuron index
            push!(firing_times, time_step)  # Add firing point to the vector
            push!(firing_index, neuron_index)
        end
    end
end

set_default_plot_size(30cm, 8cm)
raster = plot(x=firing_times, y=firing_index, Geom.point)
mem_v = plot(x=t, y=v[1, :], Geom.line)
hstack(raster, mem_v)
