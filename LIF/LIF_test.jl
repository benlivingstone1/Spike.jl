include("./LIF.jl")
using .NeuronModule
import .NeuronModule: LIFNeuron, increment_voltage!
using Plots

neuron = LIFNeuron(1.0, 1.0, -55.0, 40.0, -70.0, -70.0)

dt = 0.5
v_mem = Vector{Float64}(undef, 1)

high::Vector{Float64} = ones(100)
low::Vector{Float64} = zeros(100)
duty_cycle = vcat(high, low)

stimulus = repeat(duty_cycle, 5) * 10

noise_stim = randn(100) * 10

for t = 1:100
    I = noise_stim[t]
    increment_voltage!(neuron, I, dt)
    append!(v_mem, neuron.V)
end

plot(v_mem, xlabel="time", ylabel="Membrane potential", legend=false, )