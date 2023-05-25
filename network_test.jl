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

    y1 = []
    x1 = []
    for i in all_firings
        if i[1] == 1
            append!(y1, i[1])
            append!(x1, i[2])
        end
    end
        

    # scatter(x, y, label = "spikes in an SNN", xlabel = "Time (ms)", ylabel = "Neurons", legend = false, ms=4)
    # scatter(x[1:800], y[1:800], label = "spikes in an SNN", xlabel = "Time (ms)", ylabel = "Neurons", legend = false, ms=2)
    scatter(x1, y1, label = "spikes in an SNN", xlabel = "Time (ms)", ylabel = "Neurons", legend = false, ms=4)



    # CHECK TO SEE WHAT THE INPUT CURRENT IS DOING 
    # plots show that inhibitory neurons are not firing that much, but inhibitory neurons 
    # continuously active. Also check parameters of 'in_neuron()' and 'ex_neuron()'
    # See if you can get a more balanced output. 

end

main()
