"""
    LevinEstimate

Result of a two-order convergence check. `value` is the higher-order integral;
`low_value` is the lower-order integral. `error_estimate` is their absolute
(complex-modulus) difference, and `tolerance` is
`max(atol, rtol*abs(value))`. `tolerance_met` reports only whether this comparison
passes: agreement between two orders is not a rigorous error bound.

`low_order`, `high_order`, `low_solution`, and `high_solution` retain both
polynomial degrees and auxiliary approximations for inspection.
"""
struct LevinEstimate{V,L,E,T,S0<:LevinResult,S1<:LevinResult}
    value::V
    low_value::L
    error_estimate::E
    tolerance::T
    low_order::Int
    high_order::Int
    low_solution::S0
    high_solution::S1
    tolerance_met::Bool
end

"""
    levin_estimate(problem; lower, upper, order=16, higher_order=2*order,
                   rtol=1e-8, atol=0)
    levin_estimate(problem, phase; order=16, higher_order=2*order,
                   rtol=1e-8, atol=0)

Independently solve at two polynomial degrees and reconstruct both integrals.
Return a `LevinEstimate` using the higher-order value and
`abs(high_value - low_value)` as an absolute error estimate.

The keyword endpoint functionals have the same meaning as in
`reconstruct_integral`; the positional phase form is for ordinary, unscaled
Levin equations with finite endpoint phases. Reconstruction must return finite
real or complex scalar values.

Require `0 <= order < higher_order` and finite, nonnegative tolerances. A zero
lower order is allowed with an explicitly positive `higher_order`. There is no
automatic refinement: unmet tolerances return `tolerance_met == false`.
Linear-solve failures propagate; no fallback or rank regularization is applied.

This estimate measures agreement of the two discretizations only. It does not
bound errors in input functions, endpoint limits, or ill-conditioned solves,
and can underestimate error when both approximations share the same error.
"""
function levin_estimate(problem::LevinProblem; lower, upper,
                        order::Integer=16, higher_order::Integer=2*order,
                        rtol::Real=1e-8, atol::Real=0)
    0 <= order < higher_order ||
        throw(ArgumentError("Expected 0 <= order < higher_order."))
    isfinite(rtol) && rtol >= 0 && isfinite(atol) && atol >= 0 ||
        throw(ArgumentError("rtol and atol must be finite and nonnegative."))
    low_order, high_order = Int(order), Int(higher_order)
    low_solution = levin_solve(problem; order=low_order)
    high_solution = levin_solve(problem; order=high_order)
    low_value = reconstruct_integral(low_solution; lower, upper)
    value = reconstruct_integral(high_solution; lower, upper)
    all(v -> v isa Number && isfinite(v), (low_value,value)) ||
        throw(ArgumentError("Endpoint reconstruction must yield finite scalar integrals."))
    error_estimate = abs(value-low_value)
    tolerance = max(atol,rtol*abs(value))
    tolerance_met = isfinite(error_estimate) && isfinite(tolerance) && error_estimate <= tolerance
    return LevinEstimate(value,low_value,error_estimate,tolerance,
                         low_order,high_order,low_solution,high_solution,tolerance_met)
end

function levin_estimate(problem::LevinProblem, phase; kwargs...)
    left,right = problem.domain
    return levin_estimate(problem;
        lower=q->q(left)*exp(im*phase(left)),
        upper=q->q(right)*exp(im*phase(right)), kwargs...)
end
