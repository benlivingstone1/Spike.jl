#=
The goal of this module is to allow for the modular construction of LIF neuron networks. 
This module gives the user the ability to set all parameters and easily implement weight functions. 

Ben Livingstone September '23
=#

#=
We want to model the following equation: 

dv/dt = I - v + Σ(j,m)[Jij/N * α(t - tm)] - Σ(l)[δ(t-tl)]
Σ(j,m)[J/N * α(t - tm)] = Σ(j)Σ(m)[J/N * α(t - tm)]

We will use Euler's method to solve for Δv at each time step
Euler's method: v(t+1) = v(t) + dt*(dv/dt)
=#

# HOW CAN YOU MAKE A NETWORK BASED ON THE ABOVE EQUATION WITH INDIVIDUALLY ADDRESSABLE NEURONS? 

mutable struct LIFNeuron
    I::Real  # Input current
    V0::Real  # Initial voltage
    Ji::Vector{Real}  # Vector of [INPUT?] neuron weights
    Jj::Vector{Real}  # Vector of [OUTPUT?] neuron weights 
    spikes::Vector{Real}  # Vector of previous spike times
end