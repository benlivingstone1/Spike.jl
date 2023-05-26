module NeuronModule

    export LIFNeuron

    mutable struct LIFNeuron
        C::Float64  # Membrane capacitance
        R::Float64  # Membrane resistance
        Vth::Float64  # Threshold voltage
        Vmax::Float64  # Peak spike voltage
        Vreset::Float64  # Reset voltage
        V::Float64  # Membrane voltage

    end

    export increment_voltage!

    function increment_voltage!(neuron::LIFNeuron, I::Float64, dt::Float64)
        dV = (I - neuron.V / neuron.R) * dt / neuron.C  # Change in voltage
        neuron.V += dV
    
        if neuron.V >= neuron.Vth
            neuron.V = neuron.Vreset  # Reset voltage if threshold is crossed
        end
    end

end  # module NeuronModule
