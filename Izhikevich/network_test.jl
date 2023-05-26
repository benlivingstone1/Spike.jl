include("./izhNetwork.jl")
using .izhNetwork
using Plots

function main()
    n = network(1000)
    # run simulation on network for 1000ms and dt of 0.5ms
    t = 1000
    dt = 0.5
    all_firings = fire(n, t, dt)

    len_firings = length(all_firings)
    y = [all_firings[i][1] for i=1:len_firings]
    x = [all_firings[i][2] for i=1:len_firings]

    # scatter(x, y, label = "spikes in an SNN", xlabel = "Time (ms)", ylabel = "Neurons", markershape = :rect, legend = false, ms=1)
    plot = scatter(x, y, label = "spikes in an SNN", xlabel = "Time (ms)", ylabel = "Neurons", markershape = :rect, legend = false, ms=1)
    savefig(plot, "./plots/1000_neuron_tweaked_input_current.png")

end

main()
