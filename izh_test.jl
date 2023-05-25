include("./Izhikevich.jl")
import .Izhikevich: RS_neuron, ex_neuron, FS_neuron, in_neuron
using .Izhikevich
using Plots

function main()
    neuron = in_neuron()
    dt = 0.5 # time step of 0.5ms

    v_mem = Vector{Float64}(undef, 1)
    for t=1:1000    # 1000ms simulation
        I = 5 * randn()
        step!(neuron, dt, I)
        append!(v_mem, neuron.v)
    end

    # print(length(v_mem))
    plot(v_mem, xlabel = "Time", ylabel = "Membrane potential (mV)", legend = false)

end

main()