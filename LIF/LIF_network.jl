#=
    This script is a simulation of a ring attractor network between Leaky Integrate-and-fire 
    (LIF) neurons. The code is adapted from Laing and Chow's 2001 paper "Stationary bumps in
    Networks of Spiking Neurons". 

    - Ben Livingstone, June '23
=#

#=
We want to model the following equation: 

dv/dt = I - v + Σ(j,m)[Jij/N * α(t - tm)] - Σ(l)[δ(t-tl)]
Σ(j,m)[J/N * α(t - tm)] = Σ(j)Σ(m)[J/N * α(t - tm)]

We will use Euler's method to solve for Δv at each time step
Euler's method: v(t+1) = v(t) + dt*(dv/dt)
=#

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

# The alpha function determines the synaptic pulse
# sent to all connected neurons
# β affects the rate at which the postsynaptic current decays. 
function alpha(β, t)
    α = β * (ℯ^(-β * t))
    return α
end

# Implementation of the Dirac delta function.
# The δ(x) = 0, except for x=0 where δ(0)=1
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

function yPrime(index, I, v, J, N, dt, step, tl, s)
    tl = tl .* dt  # Times of all previous firings of neuron i
    t = step * dt

    # Σ_jm = sum((J/N) * alpha(1, t))

    series_jm = zeros(N, 1)

    β = 0.25

    for j in 1:size(J)[1]
        jn = J[j] / N
        j_spike = findall(s[j, :]) .* dt
        α = zeros(size(s[j,:])[1], 1)
        i = 1
        for tm in j_spike
            α[i] = alpha(β, t-tm)
            i +=1
        end
        series_jm[j] = sum(jn * j_spike)
    end

    Σ_jm = sum(series_jm)
    Σ_δ = sum(delta(t, tl))
    dvdt = I - v + Σ_jm - Σ_δ

    return dvdt

end

# *********************
# Weight functions
# *********************

function w(a, z)
    w = (a * π)^-0.5 * exp(-z^2 / a)
    return w
end

function J_z(z)
    # Modifying the fractions (w(fraction, z)), will flatten the trough of the sombrero.
    # Ratios closer to 1/50, more reliably create single bumps, this is becuase every neuron on 
    # the edges stay inhibitory instead of approaching 0. 

    # The second term of this expression determine the spread of the sombrero. By multiplying it by
    # 1.15, the sombrero is widened, meaning that neurons farther away have a more inhibitory effect. 
    # This gets results in only one bump being stabilized as neurons further away are inhibited by the 
    # bump activity. 

    # Jz = 5 * (1.1 * w((1/80), z) - w((1/20), z))  # original values from paper
    # Jz = 5 * (1.1 * w((1/50), z) - w((1/50), z))  # flatter tails
    Jz =  (1.1 * w((1/80), z) - 1.15 * w((1/10), z))
    return Jz
end

function J_ij(i, numNeurons)
    weights = []
    for j = 1:numNeurons
        Jij = J_z(abs(i - j) / numNeurons)
        push!(weights, Jij)
    end
    return weights
end

function norm_mat(mat)
    min_val = minimum(mat)
    max_val = maximum(mat)
    range_val = max_val - min_val

    norm_mat = (mat .- min_val) ./ range_val
    norm_mat = (norm_mat .* 1.3) .- 0.3

    return norm_mat
end

function shift_vector(vec)
    N = length(vec)
    matrix = zeros(N, N)

    # Middle rows
    matrix[Int(N/2), :] = vec

    # Shifting vec for each row
    for i = 1:N
        shift = i - Int(N/2)
        shifted_vec = circshift(vec, shift)
        matrix[i, :] = shifted_vec
    end
    return matrix
end


#=
########################
START OF THE MODEL 
########################
=#

# Define number of neurons in the model 
numNeurons = 100

###########################
# Define weights matrix, J
###########################

# ****************
# Random weights: 
# ****************

# J = rand(numNeurons, numNeurons)  # Random weights between all neurons  
# J[:, 1:Int(numNeurons * 0.8)-1] .*= 0.5  # excitatory neurons 
# J[:, Int(numNeurons * 0.8):end] .*= -0.9  # inhibitory neurons
# J[diagind(J)] .= 0.0  # no self-excitation


# *******************************************
# Lateral inhibition weights without wrapping
# *******************************************
# J = Matrix{Float64}(undef, numNeurons, numNeurons)

# for i = 1:numNeurons
#     J[i,:] = J_ij(i, numNeurons)
# end

# *****************************************
# Lateral inhibition weights with wrapping
# *****************************************

N_2 = numNeurons / 2  # Halfway through the range, the function is perfectly centered

# Get the corresponding weight vector to the N/2th neuron
J_vec = J_ij(N_2, numNeurons)
# Create the weight matrix by shifting J_vec 
J = shift_vector(J_vec)
# Normalize the weight matrix
# J = norm_mat(J)



#################################
# Meat and potatoes of the model
#################################



# Define neuron paramters 
Vthresh = 1.0  # Threshold voltage
V0 = 0.0  # Initial / Reset voltage
I = 0.5  # DC input current 

# Define time range
# 10 seconds total with time steps of 0.01s 
T = 10 
dt = 0.1  
t = [i for i = 0:dt:T]

# DC input current with noise


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

# Convert spikes into DataFrame
df = DataFrame(spikes, :auto)

# Initialize an empty vector to store the firing points
firing_times = []  # Vector for firing times (x values)
firing_index = []  # Vector for neuron numebr (y values)

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

set_default_plot_size(30cm, 16cm)
plot_i = 50
raster = plot(x=firing_times, y=firing_index, Geom.point, 
        Guide.xlabel("Time (s)"), Guide.ylabel("Neuron number"), 
        Guide.title("Raster plot for neurons 1 to $numNeurons"), shape=[Shape.square])
mem_v = plot(x=t, y=v[plot_i, :], Geom.line, Guide.xlabel("Time (s)"), 
            Guide.ylabel("Voltage (arbitrary)"), Guide.title("Membrane voltage for neuron $plot_i"))
j_n = plot(x=1:100, y=J[:, plot_i], Geom.point, Guide.xlabel("Neuron index (j)"),
            Guide.ylabel("Connection weight"), Guide.title("Connections weights for neuron $plot_i"))
v_j = hstack(mem_v, j_n)
vj_raster = vstack(v_j, raster)


