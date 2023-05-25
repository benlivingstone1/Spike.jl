# This script is to create Izhikevich neurons for use in spiking neural networks (SNNs)
# This script follows Izhikevich's 2003 paper 'Simple Model of Spiking Neurons'
# Functions are largely based on this github repo:
# https://github.com/Jumaruba/SNN/tree/master

module Izhikevich
    
    using Random

    export izh

    mutable struct izh
        # Variables needed for the Izhikevich neuron model

        v::Float64  # represents the membrane potential of the neuron
        u::Float64  # represents a membrane recovery variable, which accounts for activation of K+ currents
        # 'u' also provides negative feedback to 'v'
        thrs::Int64 # 30mV is the threshold to RESET the membrane potential and set u = u + d

        # Parameters for the Izhikevich model

        # 'a' describes teh time scale of the recovery variable 'u'
        a::Float64  # Smaller values result in slower recovery. typically a=0.02

        # 'b' describes the sensitivity of the recovery variable 'u' to subthreshold fluctuations
        # in membrane potential 'v'
        # Greater values couple 'v' and 'u' more strongly resulting in possible sub-threshold oscillations
        # and low-threshold spiking dynamics
        b::Float64  # typical value of 0.2

        # 'c' describes the after-spike reset value of the membrane potential 'v' caused by
        # the fast high-threshold K+ conductances.
        c::Float64 # typical value of -65

        # 'd' describes after spike reset of the recovery variable 'u' caused by slow high-threshold
        # Na+ and K+ conductances
        d::Float64 # typcial value of 2

    end

    # functions for using Izhikevich neurons

    export RS_neuron
    function RS_neuron()
        neuron = izh(-65.0, -14.0, 30, 0.02, 0.2, -65, 8)
        return neuron
    end

    export IB_neuron
    function IB_neuron()
        neuron = izh(-65.0, -14.0, 30, 0.02, 0.2, -55, 4)
        return neuron
    end

    export CH_neuron
    function CH_neuron()
        neuron = izh(-65.0, -14.0, 30, 0.02, 0.2, -50, 2)
        return neuron
    end

    export FS_neuron
    function FS_neuron()
        neuron = izh(-65.0, -14.0, 30, 0.1, 0.2, -65, 2)
        return neuron
    end

    export LTS_neuron
    function LTS_neuron()
        neuron = izh(-65.0, -14.0, 30, 0.02, 0.25, -65, 2)
        return neuron
    end

    export RZ_neuron
    function RZ_neuron()
        neuron = izh(-65.0, -14.0, 30, 0.1, 0.26, -65, 2)
        return neuron
    end

    # The best way to achieve heterogeneity between neurons is to make slight randomizations
    # to the neuron parameters, here are functions for heterogeneous excitatory and inhibitory neurons

    export ex_neuron
    function ex_neuron()
        neuron = izh(-65.0, -14.0, 30, 0.02, 0.2, (-65 + 15)*(rand()^2), (8-6)*(rand()^2))
        return neuron
    end

    export in_neuron
    function in_neuron()
        neuron = izh(-65.0, -14.0, 30, (0.02 + 0.08)*rand(), (0.25 - 0.05)*rand(), -65, 2)
        return neuron
    end

    # Neuron encoding / decoding functions

    export step!
    function step!(neuron, dt, I)
        if neuron.v >= neuron.thrs
            neuron.v = neuron.c
            neuron.u += neuron.d
        else
            solve_euler!(neuron, dt, I)
        end
    end

    function solve_euler!(neuron, dt, I)
        dv = f_v(neuron, I, neuron.v, dt)
        du = f_u(neuron, dt)
        neuron.v += dv
        neuron.u += du
    end

    function f_v(neuron, i, v, dt)
        return (0.04 * v * v + 5 * v + 140 - neuron.u + i) * dt
    end

    function f_u(neuron, dt)
        return neuron.a * (neuron.b * neuron.v - neuron.u) * dt
    end

end

