"""
    levin_solve(problem::LevinProblem; order=32)

Assemble and solve the square Chebyshev collocation system at a fixed polynomial
degree `order` (`order + 1` interior nodes). Return a callable `LevinResult`
containing the auxiliary function q and its Chebyshev coefficients.

Each nonzero matrix row and its RHS are divided by the largest absolute entry
in that row before Julia's direct linear solve. This balances equation magnitudes
without changing the mathematical problem. Zero rows are left unchanged, so
singular systems still fail in the linear solve. `assemble_operator` continues
to return the original, unscaled operator. No boundary conditions,
rank truncation, adaptive refinement, or integral reconstruction are applied.
Linear-algebra failures propagate to the caller. A successful solve alone is
not an accuracy estimate or a guarantee of a well-conditioned system.

# Example
```julia
# x*q′ + 2*q = 2 + 3*x, with polynomial solution q = 1 + x.
problem = LevinProblem(x -> 2 + 3*x, (x -> 2, x -> x), (0.0, 1.0))
solution = levin_solve(problem; order=8)
solution(0.5)  # approximately 1.5
solution.coefficients
```
"""
function levin_solve(problem::LevinProblem; order::Integer=32)
    system = assemble_operator(problem; order)
    matrix = system.matrix
    scales = vec(maximum(abs, matrix; dims=2))
    all(isfinite, scales) || throw(ArgumentError("Assembled operator must be finite."))
    # Do not turn a singular zero row into NaNs. Keep it for the linear solver.
    scales = map(s -> iszero(s) ? one(s) : s, scales)
    matrix ./= scales
    # Allocate to promote integer/real RHS values when division requires it.
    rhs = system.rhs ./ scales
    all(isfinite, rhs) || throw(ArgumentError("Row-normalized RHS must be finite."))
    coefficients = matrix \ rhs
    return LevinResult(problem, coefficients)
end
