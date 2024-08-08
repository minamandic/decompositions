using Pkg
using LinearAlgebra

#=
This is a function that computes the singular value decomposition of
a given matrix.
Params: X (original matrix)
Returns: U (left singular vectors), sigma (matrix containing singular values),
VT (right singular vectors transposed)
=#

function SVD(X)
    XT = transpose(X)
    XTX = XT * X
    
    eigenvals, _ = eigen(XTX)
    singular_vals = []
    for i in eigenvals
        if i >= 0
            push!(singular_vals, sqrt(i))
        else
            push!(singular_vals, sqrt(abs(i)))
        end
    end
    
    sort!(singular_vals, rev=true)

    count = length(singular_vals)
    sigma = zeros(count, count)

    for i in 1:count
        sigma[i, i] = singular_vals[i]
    end

    sigma_inv = inv(sigma)
    
    eigenvalues, eigenvectors = eigen(XTX)
    sorted_indices = sortperm(eigenvalues, rev=true)
    V = -eigenvectors[:, sorted_indices]
    VT = transpose(V)
    
    U = X * (V * sigma_inv)
    
    return U, sigma, VT
end

#=
This is a function that contracts the tensor along each mode of the factor
matrices to compute the core tensor
Params: A (original tensor), M (factor matrix), mode (dimension to unfold over)
Returns: core tensor
=#

function tensor_times_matrix(A, M, mode)
    perm = [mode; setdiff(1:ndims(A), mode)...]
    B = permutedims(A, perm)
    B = reshape(B, size(A, mode), :)
    B = M * B
    newdims = (size(M, 1), size(A)[setdiff(1:ndims(A), mode)]...)
    return permutedims(reshape(B, newdims), invperm(perm))
end

#=
This is a function that computes the Tucker Decomposition of an 
order-3 tensor using the Higher-Order Orthogonal Iteration 
Algorithm (HOOI) and the SVD program above.
Params: A (original tensor)
Returns: core tensor, factor matrices
=#

function tucker_decomposition(A, ranks)
    modes = ndims(A)
    factors = []
    for mode in 1:modes
        perm = [mode; setdiff(1:modes, mode)...]
        unfolded_tensor = permutedims(A, perm)
        unfolded_tensor = reshape(unfolded_tensor, size(A, mode), :)
        U, _, _ = SVD(unfolded_tensor)
        
        factor_matrix = U[:, 1:ranks[mode]]
        push!(factors, factor_matrix)
    end
    
    core_tensor = A
    for mode in 1:modes
        core_tensor = tensor_times_matrix(core_tensor, factors[mode]', mode)
    end
    
    return core_tensor, factors
end

A = rand(5, 5, 5)
ranks = [2, 2, 1]
println("Original Tensor:")
println(A)
core_tensor, factors = tucker_decomposition(A, ranks)
println("Core Tensor:")
println(core_tensor)
println("\nFactor Matrices:")
for (i, factor) in enumerate(factors)
    println("Factor matrix $i:")
    println(factor)
end