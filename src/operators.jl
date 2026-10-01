"""
    assemble_operator(problem::LevinProblem; order=32)

Return `(nodes, matrix, rhs)` for Chebyshev collocation of the supplied equation.
`order` is the maximum polynomial degree: there are `order + 1` coefficients
and interior Chebyshev–Gauss nodes, listed from right to left. Matrix column
`k + 1` applies the operator to `Tₖ(2(x-left)/(right-left)-1)`.

The derivative includes the physical-interval scaling `2/(right-left)`.
Each callback is evaluated once per node. Endpoints are not sampled, so a
vanishing derivative coefficient there is allowed. No linear solve, boundary
constraint, rank decision, or error estimate is performed by this function.
"""
function assemble_operator(problem::LevinProblem; order::Integer=32)
    order >= 0 || throw(ArgumentError("order must be nonnegative."))
    left, right = problem.domain
    n = order + 1
    T = typeof(left)
    xi = [cospi(T(2*j + 1) / T(2*n)) for j in 0:order]
    nodes = [left + (right-left) * (t+1)/2 for t in xi]
    basis = Matrix{T}(undef, n, n)
    deriv = Matrix{T}(undef, n, n)
    for j in 1:n
        basis[j,1] = one(T)
        deriv[j,1] = zero(T)
        if order >= 1
            basis[j,2] = xi[j]
            deriv[j,2] = one(T)
        end
        for k in 2:order
            basis[j,k+1] = 2*xi[j]*basis[j,k] - basis[j,k-1]
            deriv[j,k+1] = 2*basis[j,k] + 2*xi[j]*deriv[j,k] - deriv[j,k-1]
        end
    end
    a0, a1 = problem.coefficients
    values0 = a0.(nodes)
    values1 = a1.(nodes)
    rhs = problem.rhs.(nodes)
    for values in (values0, values1, rhs)
        all(v -> v isa Number && isfinite(v), values) ||
            throw(ArgumentError("Operator and RHS callbacks must return finite scalar numbers at the nodes."))
    end
    scale = 2/(right-left)
    matrix = [values0[j]*basis[j,k] + values1[j]*scale*deriv[j,k]
              for j in 1:n, k in 1:n]
    return (; nodes, matrix, rhs)
end
