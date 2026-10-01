"""
    reconstruct_integral(solution::LevinResult; lower, upper)

Evaluate `upper(solution) - lower(solution)`. Each endpoint functional receives
the callable auxiliary solution and returns the corresponding boundary value
of the physical primitive, including any scaling and oscillatory factor.

`lower` and `upper` refer to the original integral's bounds, not necessarily
the left and right endpoints of the collocation coordinate. This supports
reversed coordinates and known singular-endpoint limits. For example,
`upper = q -> 0.0im` supplies a vanishing limit without evaluating an undefined
phase or the auxiliary solution at that endpoint.

Both functionals are evaluated once. They must be consistent with the supplied
Levin equation; this function does not infer transformations, validate endpoint
limits, or estimate the integration error.
"""
function reconstruct_integral(solution::LevinResult; lower, upper)
    lower_value = lower(solution)
    upper_value = upper(solution)
    return upper_value - lower_value
end

"""
    reconstruct_integral(solution::LevinResult, phase)

Convenience reconstruction for the ordinary equation `q′ + im*phase′*q = rhs`
on the solution interval `(left, right)`:

    q(right)*exp(im*phase(right)) - q(left)*exp(im*phase(left))

Use this method only when q is the unscaled Levin auxiliary function in the
original integration coordinate and the endpoint phases are finite. For
coordinate transformations, scaling, or singular endpoint limits, supply the
`lower` and `upper` functionals instead.
"""
function reconstruct_integral(solution::LevinResult, phase)
    left, right = solution.problem.domain
    return reconstruct_integral(solution;
        lower = q -> q(left)*exp(im*phase(left)),
        upper = q -> q(right)*exp(im*phase(right)))
end
