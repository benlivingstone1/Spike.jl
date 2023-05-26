include("./LIF.jl")
using .NeuronModule
import .NeuronModule: LIFNeuron, run_LIF!
using Plots

neuron = LIFNeuron()
T = 400.0  # Total time of simulation
dt = 0.1  # Time steps for simulation
I = 300.0  # DC current stimulation
v, rec_spikes = run_LIF!(neuron, T, dt, I)

time = [i for i in 0:dt:T]

plot(time, v)