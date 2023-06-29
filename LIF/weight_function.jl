using Gadfly, DataFrames, LinearAlgebra

function w(a, z)
    w = (a * π)^-0.5 * exp(-z^2 / a)
    return w
end

function J_z(z)
    Jz = 5 * (1.1 * w((1/80), z) - w((1/20), z))
    return Jz
end

function J_ij(i, numNeurons)
    weights = []
    for j = 1:numNeurons
        Jij = J_z(abs(i - j) / numNeurons)
        push!(weights, Jij)
    end
    return weights
end

function norm_mat(mat)
    min_val = minimum(mat)
    max_val = maximum(mat)
    range_val = max_val - min_val

    norm_mat = (mat .- min_val) ./ range_val
    norm_mat = (norm_mat .* 1.25) .- 0.25

    return norm_mat
end

function shift_vector(vec)
    N = length(vec)
    matrix = zeros(N, N)

    # Middle rows
    matrix[Int(N/2), :] = vec

    # Shifting vec for each row
    for i = 1:N
        shift = i - Int(N/2)
        shifted_vec = circshift(vec, shift)
        matrix[i, :] = shifted_vec
    end
    return matrix
end


numNeurons = 100

# # Initialize NxN weight matrix
# J = Matrix{Float64}(undef, numNeurons, numNeurons)

# # Create rows of weights between the ith neuron to all j neurons
# for i = 1:numNeurons
#     J[i,:] = J_ij(i, numNeurons)
# end

# # Normalize the weight matrix between two values
# J = norm_mat(J)

# Wrap the function around the indices:

N_2 = numNeurons / 2  # Halfway through the range, the function is perfectly centered

# Get the corresponding weight vector to the N/2th neuron
J_vec = J_ij(N_2, numNeurons)
# Create the weight matrix by shifting J_vec 
J = shift_vector(J_vec)

# Shift the vector indices +/- 1 to get the first / last halves of the weight matrix 
# this allows the weight function to "wrap" from the first index to the last
# This means that neuron 1 will have a strong connection with the Nth neuron

N = 1:numNeurons

set_default_plot_size(30cm, 32cm)
Q1 = plot(x=N, y=J[1,:], Geom.point)
Q2 = plot(x=N, y=J[25,:], Geom.point)
Q3 = plot(x=N, y=J[50,:], Geom.point)
Q4 = plot(x=N, y=J[75,:], Geom.point)

p = vstack(Q1, Q2, Q3, Q4)

