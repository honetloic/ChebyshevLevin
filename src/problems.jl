"""
    LevinProblem(rhs, (a0, a1), (left, right))

Describe the scalar equation `a1(x)*q′(x) + a0(x)*q(x) = rhs(x)`
on a finite, strictly increasing interval. All three coefficients are callables;
real and complex scalar values are supported. The coefficient tuple is ordered
by derivative, starting with the coefficient of q itself.

Coordinate transformations, regularization, and integral reconstruction belong
to the caller. In particular, this equation alone does not specify an integral
or select a unique auxiliary solution. No endpoint constraints are imposed.
"""
struct LevinProblem{F,A0,A1,T<:Real}
    rhs::F
    coefficients::Tuple{A0,A1}
    domain::Tuple{T,T}

    function LevinProblem(rhs::F, coefficients::Tuple{A0,A1},
                          domain::Tuple{Real,Real}) where {F,A0,A1}
        left, right = promote(float(domain[1]), float(domain[2]))
        isfinite(left) && isfinite(right) && left < right ||
            throw(ArgumentError("Expected finite endpoints with left < right."))
        return new{F,A0,A1,typeof(left)}(rhs, coefficients, (left, right))
    end
end

"""
    LevinResult(problem, coefficients)

Store an auxiliary Chebyshev approximation to q, with coefficients ordered
`T₀, T₁, …` on the problem interval. Call `result(x)` to evaluate it using
Clenshaw recurrence. This container does not certify that the equation has
been solved, impose a boundary condition, or represent an integral value.
Evaluation outside the interval is rejected.
"""
struct LevinResult{P<:LevinProblem,C<:AbstractVector}
    problem::P
    coefficients::C

    function LevinResult(problem::P, coefficients::C) where
            {P<:LevinProblem,C<:AbstractVector}
        isempty(coefficients) && throw(ArgumentError("At least one coefficient is required."))
        firstindex(coefficients) == 1 || throw(ArgumentError("Coefficients must use one-based indexing."))
        all(c -> c isa Number && isfinite(c), coefficients) ||
            throw(ArgumentError("Coefficients must be finite scalar numbers."))
        return new{P,C}(problem, coefficients)
    end
end

function (result::LevinResult)(x::Real)
    left, right = result.problem.domain
    left <= x <= right || throw(DomainError(x, "Evaluation point is outside the problem interval."))
    xi = 2 * (x - left) / (right - left) - 1
    c = result.coefficients
    b1 = zero(c[1] * xi)
    b2 = b1
    for k in length(c):-1:2
        b0 = c[k] + 2 * xi * b1 - b2
        b2, b1 = b1, b0
    end
    return c[1] + xi * b1 - b2
end
