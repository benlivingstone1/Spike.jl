include("./LIF.jl")
using .NeuronModule
import .NeuronModule: LIFNeuron, run_LIF!
using Plots, Random

function rand_generator(T, dt)
    samples = Int(T / dt)
    I = rand(samples) * 400
    
    return I
end


neuron = LIFNeuron()
T = 400.0  # Total time of simulation
dt = 0.1  # Time steps for simulation
# I = 300.0  # Constant DC current stimulation
# Random current injection
I = Vector{Float64}(undef, Int(T/dt))
I = rand_generator(T, dt)

v, rec_spikes = run_LIF!(neuron, T, dt, I)

time = [i for i in 0:dt:T]

plot(time, v)