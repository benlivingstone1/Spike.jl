# This script uses the Izhikevich struct and supporting functions to create a network of
# neurons and run a simulation of the network firing.
# Large portions of this code are translations from this github repo
# https://github.com/Jumaruba/SNN/tree/master

module izhNetwork
    include("./Izhikevich.jl")
    using .Izhikevich

    export network
    mutable struct network
        # Set the number of neurons in the network
        numNeurons::Int64

        # initialize varialbes for number of excitatory and inhibitory neurons
        Ni::Int64   # inhibitory
        Ne::Int64   # excitatory

        # Initialize vector of neurons
        neurons::Vector

        # Initialize weight matrix 
        weights::Matrix

        function network(numNeurons)
            Ni = Int(ceil(numNeurons / 5))   # 1/5th of the neurons are inhibitory
            Ne = Int(numNeurons - Ni)        # The rest are excitatory
    
            # Create the neurons
            neurons = Vector{Any}(undef, numNeurons)
            lenNeur = length(neurons)
            for i=1:lenNeur
                if i < Ne
                    neurons[i] = ex_neuron()    # excitatory neurons
                else
                    neurons[i] = in_neuron()    # inhibitory neurons
                end
            end
    
            # Generate weight matrix 
            weights = rand(numNeurons, numNeurons) # Generate random nums between 0-1
            weights[:, 1:Ne-1] .= weights[:, 1:Ne-1] .* 0.5     # excitatory weights
            weights[:, Ne:end] .= weights[:, Ne:end] .* -1  # inhibitory weights

            new(numNeurons, Ni, Ne, neurons, weights)
            
        end

    end

    export fire
    function fire(network, time, dt) # time = total simulation time (ms); dt = time step between calculations (ms)
        firings = []

        for t = 1:time
            I = [8*randn(network.Ne); 2*randn(network.Ni)]  # thalamic input
            # fired = [i for i=1:length(network.neurons) if network.neurons[i].v >= network.neurons[i].thrs]
            fired = []
            for i = 1:network.numNeurons
                if network.neurons[i].v >= network.neurons[i].thrs
                    append!(fired, i)
                end
            end
            len_fired = length(fired)
            len_firings = length(firings)

            # if len_fired == 0
            #     return
            if len_firings == 0
                firings = [[neuronNumber, t] for neuronNumber in fired]
            else
                fire_time = [[neuronNumber, t] for neuronNumber in fired]
                append!(firings, fire_time)
            end

            # Update 'u' and 'v' for neurons that fired
            len_fired = length(fired)
            for k = 1:len_fired
                for i = 1:ceil(1/dt)
                    step!(network.neurons[fired[k]], dt, I[fired[k]])
                end
            end

            # Update I
            I .+= sum(network.weights[:, fired], dims = 2)

            for k = 1:network.numNeurons
                for i = ceil(1/dt)
                    step!(network.neurons[k], dt, I[k])
                end
            end

                
        end

        return firings

    end

end

