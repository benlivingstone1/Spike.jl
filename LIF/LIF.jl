# This module creates Leaky Integrate-and-Fire (LIF) neurons
# The LIF neuron activity is dictated by the following equation
#
# 𝜏𝑚Δ𝑉𝑛/Δ𝑡=−(𝑉𝑛−𝑉𝐿)+𝐼𝑛/𝑔
# 
# Tau is the time constant, defined by tau_m = Cm/gl
# Code is adapted from: https://github.com/johanjan/MOOC-HPFEM-source/blob/master/LIF_ei_balance_irregularity.ipynb

module NeuronModule

    export LIFNeuron

    mutable struct LIFNeuron
        V_th::Int64     # Spike threshold [mV]
        V_reset::Int64  # Reset potential [mV]
        tau_m::Float64  # Membrane time constant [ms]
        g_L::Float64    # leak conductance [nS]
        V_init::Float64 # initial potential [mV]
        V_L::Float64    # leak reversal potential [mV]
        tref::Float64   # refractory time [ms]

        function LIFNeuron()
            V_th = -55
            V_reset = -75
            tau_m = 10
            g_L = 10
            V_init = -65
            V_L = -75
            tref = 2

            new(V_th, V_reset, tau_m, g_L, V_init, V_L, tref)

        end

    end

    export run_LIF!
    function run_LIF!(neuron::LIFNeuron, T::Float64, dt::Float64, I::Float64)
        # Set parameters
        range_t = [i for i = 0:dt:T]  # Vector for all time points
        Lt = length(range_t) 

        # Initialize the voltage and currents
        v = zeros(Lt)
        v[1] = neuron.V_init
        I = I * ones(Lt)
        tr = 0.0

        # Simulate the LIF dynamics
        rec_spikes = Vector{Any}(undef, 0)  # record spike times
        for i in 1:Lt-1
            if tr > 0
                v[i] = neuron.V_reset
                tr = tr-1
            elseif v[i] >= neuron.V_th
                append!(rec_spikes, i)
                v[i] = neuron.V_reset
                tr = neuron.tref / dt
            end

            # Calculate the increment of the membrane potential
            dv = (-(v[i]-neuron.V_L) + I[i]/neuron.g_L) * (dt/neuron.tau_m)

            # Update the membrane potential
            v[i+1] = v[i] + dv

        end

        rec_spikes = rec_spikes .* dt

        return v, rec_spikes
        
    end


end  # module NeuronModule
