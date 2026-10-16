"""
Numerical integration against supplied Chebyshev moments. This backend depends
on the shared Chebyshev utilities and QuadGK. It knows no Kerr
geometry or radial domain and does not solve a Levin ODE. `MomentProblem` is the generic boundary;
`inverse_phase_weights` supplies one moment family. Physics and solver selection
belong to callers.
"""
module Moments

using QuadGK: quadgk_segbuf!
using ..ChebyshevLevin: chebyshev_gauss_nodes, from_chebyshev_coordinate, chebyshev_value, weighted_chebyshev_values!
export MomentProblem, MomentEstimate, moment_estimate, integrate_moments, inverse_phase_weights
const DEFAULT_ORDERS = (8,12,16,24,32,48,64)

"""
    MomentProblem(amplitude, (left,right), moments; weight_norm, orders=DEFAULT_ORDERS)

Represent `integral amplitude(x)*weight(x) dx` on a finite interval.
`moments(n)` returns `n+1` integrals of
`T_k(2*(x-left)/(right-left)-1)*weight(x)`, k=0:n, including the Jacobian.
`weight_norm` bounds `integral abs(weight(x)) dx`. Only the smooth amplitude is
sampled: the weight and its oscillations enter through the supplied moments.
Callbacks must return finite values convertible to ComplexF64. Errors in the
amplitude or moment callbacks are not estimated by this interface.
"""
struct MomentProblem{F,W,O}
    amplitude::F
    interval::Tuple{Float64,Float64}
    moments::W
    weight_norm::Float64
    orders::O
end

function MomentProblem(amplitude, interval::Tuple{Real,Real}, moments;
                       weight_norm::Real, orders=DEFAULT_ORDERS)
    left,right=Float64.(interval)
    isfinite(left) && isfinite(right) && left<right || throw(ArgumentError("Invalid moment interval."))
    isfinite(weight_norm) && 0<=weight_norm<=floatmax(Float64) ||
        throw(ArgumentError("Expected a finite nonnegative weight norm."))
    degrees=Tuple(orders)
    length(degrees)>=2 && all(n->n isa Integer && 0<=n<=256,degrees) &&
        all(diff(collect(degrees)).>0) || throw(ArgumentError("Supply at least two increasing integer degrees in 0:256."))
    return MomentProblem(amplitude,(left,right),moments,Float64(weight_norm),degrees)
end

"""
A numerical integral and its refinement state, independent of physical domains.
The error estimate is the maximum of degree change, sampled interpolation
residual times `weight_norm`, and a roundoff indicator. The sampled residual is
not a supremum bound: narrow unsampled features can escape it. Callback errors
are separate. No auxiliary endpoint solution or physical fallback is stored.
"""
struct MomentEstimate
    value::ComplexF64
    low_value::ComplexF64
    error_estimate::Float64
    tolerance::Float64
    low_order::Int
    high_order::Int
    tolerance_met::Bool
    degree_index::Int
    interpolation_error::Float64
    roundoff_error::Float64
    probes::Vector{ComplexF64}
end

# Re-sum the final quadrature panels with componentwise compensated addition.
# QuadGK's error estimate does not include its final accumulation roundoff.
function sum_moment_segments!(result, segments)
    fill!(result,0)
    correction=zeros(eltype(result),length(result))
    for segment in segments, j in eachindex(result)
        addend=segment.I[j]-correction[j]
        total=result[j]+addend
        correction[j]=(total-result[j])-addend
        result[j]=total
    end
    return result
end

"""
    inverse_phase_weights(beta, b, order; moment_atol=1e-14, maxevals=2_000_000)

Return `integral_0^b T_k(2x/b-1)*exp(-im*beta/x) dx`, `k=0:order`, using
Float64/ComplexF64 throughout. `moment_atol` requests normalized absolute error:
error in each returned weight divided by `b`. It is not a relative tolerance or
an error certificate. The previous arbitrary-precision `precision_bits` control
has been replaced by this accuracy request.

For `eta=beta/b >= 0`, put `u=b/x`. Choose a turn U>=1, keep the real segment
`1 <= u <= U` and rotate only its tail to `u=U-im*t`. No pole is crossed, and the
large closing arc vanishes because the rational amplitude is O(abs(u)^(-2)).
The exact normalized moment is

    integral_1^U T_k(2/u-1)*exp(-im*eta*u)/u^2 du
    - im*exp(-im*eta*U)*integral_0^Inf T_k(2/(U-im*t)-1)*exp(-eta*t)/(U-im*t)^2 dt.

The polynomial is never expanded into monomials. On the real segment its
modulus is <= 1. Define eta_switch=1/max(2,order^2). For
0<eta<eta_switch use U=max(2,order^2). On its tail,
zeta=2/(U-im*t)-1 satisfies
abs(zeta+1)<=2/U and abs(zeta-1)<=2. The Chebyshev ellipse identity gives
abs(T_k(zeta)) <= cosh(k*sqrt(2/U)) <= C=cosh(order*sqrt(2/U)) <= cosh(sqrt(2)).
Thus the absolute integral norm is <= 1-1/U+pi*C/(2U), less than 1.000592 at
order 64. This removes the huge artificial cancellation of the monomial
representation; it does not cure relative conditioning near moment zeros.
Rotating at U=1 without sufficient damping would instead allow
abs(T_64(im)) approximately 1.57e24.

For this weak-damping regime set t=U*exp(s). The tail's absolute integrand
is <= C/U*exp(-abs(s)); truncating
s to [-L,L] has error <= 2C/U*exp(-L), uniformly down to eta=0. Negative eta
uses conjugation. Exactly zero uses odd weights zero and even weights b/(1-k^2),
without a small-eta cutoff or an invalid Taylor expansion under the integral.

For eta>=eta_switch, use the earlier turn U=max(1,order/sqrt(eta)). Use
v=eta*t on the tail. If z0=2/U-1, abs(zeta-z0)<=2t/U^2, so the ellipse identity
bounds abs(T_k(zeta))<=exp(2k*sqrt(t)/U)<=exp(2sqrt(v)). The entire damped factor
is therefore <=exp(2sqrt(v)-v)<=exp(2-v/2). Its absolute tail integral is
<=2exp(2)/(eta*U^2); truncation after v=V costs <=2exp(2-V/2)/(eta*U^2).
This bound holds for every eta>0, not only eta>=1. For order>=2, eta_switch
is where order^2 and order/sqrt(eta) coincide. Turning earlier above that
threshold avoids an unnecessarily long oscillatory real segment, while keeping
the same growth/damping and omission bounds. Choose V from the same omission
budget. Orders 0 and 1 use eta_switch=1/2; both bounds remain valid there.

Gauss-Kronrod vector quadrature shares phases and Chebyshev values. Initial
real panels grow by at most 1.5 and limit eta*Delta(u) to pi. Half the requested
accuracy is reserved for the two quadratures (80% real, 20% tail); the tail
omission budget is min(1e-17,moment_atol/1000). Compensated panel accumulation
reduces summation roundoff. The remaining margin is empirical, not a rigorous
bound on polynomial/phase evaluation or floating-point quadrature error.

Degrees 0:256 and finite Float64 eta are accepted. The initial real mesh costs
O(log(U)+eta*(U-1)) panels before adaptive refinement. Here U=max(2,order^2)
below eta_switch and U=max(1,order/sqrt(eta)) above it. Adaptive cost, including
resolution of the polynomial near u=1, is not guaranteed.
Throw an error if the initial mesh or adaptive calculation exhausts maxevals,
if its error indicator exceeds moment_atol, or if values are nonfinite.
No arbitrary-precision fallback or silently clipped degree/detuning is used.
A tight request may fail at the Float64 roundoff floor. The supplied-moment
interpolation diagnostic does not include these callback errors; callers remain
responsible for propagating moment accuracy to their final physical observable.
"""
function inverse_phase_weights(beta::Real,b::Real,order::Integer;
                               moment_atol::Real=1e-14,maxevals::Integer=2_000_000)
    beta64,b64,tolerance=Float64(beta),Float64(b),Float64(moment_atol)
    isfinite(beta64) && isfinite(b64) && b64>0 ||
        throw(ArgumentError("Expected finite beta and positive b representable in Float64."))
    0<=order<=256 || throw(ArgumentError("Expected degree in 0:256."))
    isfinite(tolerance) && tolerance>=eps(Float64) ||
        throw(ArgumentError("moment_atol must be finite and at least eps(Float64)."))
    maxevals>=42 || throw(ArgumentError("maxevals must be at least 42."))
    eta=beta64/b64
    isfinite(eta) || throw(ArgumentError("beta/b must be finite in Float64."))
    eta==0 && return ComplexF64[isodd(k) ? 0.0 : b64/(1-k*k) for k in 0:order]
    detuning=abs(eta)
    plan=inverse_phase_contour_plan(detuning,order,tolerance,maxevals)
    quadrature=evaluate_inverse_phase_contour(plan,detuning,order,tolerance,maxevals)
    return checked_inverse_phase_weights(quadrature,plan,eta,b64,tolerance,maxevals)
end

# Log of the damped-tail omission bound 2exp(2-V/2)/(detuning*U^2).
# Keep the subtraction order explicit, including at V=0 when choosing V.
damped_tail_log_bound(detuning,U,V) = log(2.0)+2-V/2-log(detuning)-2log(U)

# Plan the turn, tail truncation and initial real mesh without evaluating moments.
function inverse_phase_contour_plan(detuning,order,tolerance,maxevals)
    # The damped bound applies for any positive detuning. For order>=2,
    # switch where the two turn lengths coincide, rather than waiting for eta=1.
    damped=detuning>=1/max(2,order*order)
    U=damped ? max(1.0,order/sqrt(detuning)) : Float64(max(2,order*order))
    tail_budget=min(1e-17,tolerance/1000)
    if damped
        # Logarithms avoid overflow for large, finite detuning.
        V=max(1.0,2*(damped_tail_log_bound(detuning,U,0.0)-log(tail_budget)))
        tail_panels=Float64[0.0,V]
        tail_omission=exp(damped_tail_log_bound(detuning,U,V))
    else
        C=cosh(order*sqrt(2/U))
        L=ceil(log(2C/(U*tail_budget)))
        tail_panels=Float64[-L,0.0,L]
        tail_omission=2C/U*exp(-L)
    end
    # Guard the initial mesh before allocation, including phase-step overflow.
    (U-1)*(detuning/pi)<maxevals/21 ||
        error("Inverse-phase moment initial mesh exceeds maxevals; reduce degree/detuning or increase the work budget.")
    panels=Float64[1.0]
    while panels[end]<U
        push!(panels,min(U,1.5panels[end],panels[end]+pi/detuning))
    end
    21*(length(panels)-1)<maxevals || error("Inverse-phase moment initial mesh exceeds maxevals.")
    return (;damped,U,tail_panels,tail_omission,panels)
end

# Execute the two vector quadratures on the planned contour. Both keep T_k in
# Chebyshev form and re-sum the accepted panels with compensated addition.
function evaluate_inverse_phase_contour(plan,detuning,order,tolerance,maxevals)
    (;damped,U,tail_panels,panels)=plan
    real_calls,tail_calls=Ref(0),Ref(0)
    function real_integrand!(values,u)
        real_calls[]+=1
        weighted_chebyshev_values!(values,2/u-1,cis(-detuning*u)/(u*u))
    end
    phase=-im*cis(-detuning*U)
    function tail_integrand!(values,s)
        tail_calls[]+=1
        if damped
            denominator=complex(U,-s/detuning)
            factor=(phase/detuning)*exp(-s)/(denominator*denominator)
            weighted_chebyshev_values!(values,2/denominator-1,factor)
        else
            t=exp(s)
            denominator=complex(1.0,-t)
            factor=(phase/U)*t/(denominator*denominator)*exp(-detuning*U*t)
            weighted_chebyshev_values!(values,2/(U*denominator)-1,factor)
        end
    end
    real_budget,tail_quad_budget=.4tolerance,.1tolerance
    norm_inf=values->maximum(abs,values)
    real_value=zeros(ComplexF64,order+1)
    real_error=0.0
    if U>1
        real_value,real_error,real_segments=quadgk_segbuf!(real_integrand!,real_value,panels;
            order=10,atol=real_budget,rtol=0,norm=norm_inf,maxevals)
        sum_moment_segments!(real_value,real_segments)
    end
    tail_limit=min(maxevals,200_000)
    tail_value,tail_error,tail_segments=quadgk_segbuf!(tail_integrand!,zeros(ComplexF64,order+1),tail_panels;
        order=10,atol=tail_quad_budget,rtol=0,norm=norm_inf,maxevals=tail_limit)
    sum_moment_segments!(tail_value,tail_segments)
    return (;real_value,tail_value,real_error,tail_error,
            real_calls=real_calls[],tail_calls=tail_calls[],tail_limit)
end

# Enforce the work/accuracy indicators, then restore the sign and physical scale.
function checked_inverse_phase_weights(quadrature,plan,eta,b,tolerance,maxevals)
    (;real_value,tail_value,real_error,tail_error,real_calls,tail_calls,tail_limit)=quadrature
    real_calls<maxevals && tail_calls<tail_limit ||
        error("Inverse-phase moment quadrature exhausted its work budget; accuracy is unresolved.")
    # Returned estimates can change when QuadGK re-sums many panels. The tighter
    # stop requests above leave margin for this; assess the final indicator
    # against the user's unchanged tolerance rather than an internal fraction.
    isfinite(real_error+tail_error) && real_error+tail_error+plan.tail_omission<=tolerance ||
        error("Inverse-phase moment quadrature did not meet moment_atol; Float64 roundoff or discretization remains unresolved.")
    weights=real_value+tail_value
    eta<0 && (weights=conj.(weights))
    weights .*= b
    all(isfinite,weights) || error("Nonfinite inverse-phase moment result.")
    return weights
end

function chebyshev_coefficients(amplitude,interval,order)
    node_count=order+1
    angles=[pi*(j+.5)/node_count for j in 0:order]
    # Interior roots avoid endpoints. This direct cosine transform interpolates
    # the amplitude only; halving coefficient 0 gives the full-constant convention
    # shared with Clenshaw. The oscillatory weight enters through the moments.
    nodes=chebyshev_gauss_nodes(order)
    values=ComplexF64[amplitude(from_chebyshev_coordinate(t,interval)) for t in nodes]
    all(isfinite,values) || throw(ArgumentError("Nonfinite smooth amplitude."))
    coefficients=[2/node_count*sum(values[j]*cos(k*angles[j]) for j in 1:node_count) for k in 0:order]
    coefficients[1]/=2
    return coefficients
end

# Eighteen fixed, symmetric interior probes are reused between degrees. The
# offset (approximately pi/10) avoids interpolation-root and midpoint coincidences
# across the default ladder. Eighteen probes are an empirical cost/coverage choice,
# not a sampling theorem; narrow features or aliasing can escape this check.
probe_nodes() = [(j+.314159)/(17+.628318) for j in 0:17]

function estimate_order(problem,order,probes)
    coefficients=chebyshev_coefficients(problem.amplitude,problem.interval,order)
    weights=problem.moments(order)
    length(weights)==order+1 && all(isfinite,weights) || throw(ArgumentError("Invalid supplied moments."))
    terms=coefficients.*weights
    value=sum(terms)
    residual=problem.weight_norm*maximum(abs(probes[j]-chebyshev_value(coefficients,2t-1))
                                        for (j,t) in enumerate(probe_nodes()))
    # Term magnitudes expose cancellation hidden by abs(value). 32 eps is the
    # retained empirical safety margin for coefficient/sum rounding at default
    # degrees 8..64, not a rigorous forward-error bound or a callback error bound.
    roundoff=32eps(Float64)*sum(abs,terms)
    isfinite(value) && isfinite(residual) && isfinite(roundoff) || throw(ArgumentError("Nonfinite moment estimate."))
    return (;value,residual,roundoff)
end

"""
    moment_estimate(problem; atol=1e-12, rtol=1e-8, previous=nothing)

Compare the first two degrees, or advance one degree with `previous` from this
same problem. Reuse the previous integral and independent probes. No physical
fallback is chosen. Use `integrate_moments` for a complete adaptive calculation.

For amplitude H, weight w, interpolant p_N = sum(c_k*T_k), supplied moments
mu_k = integral(T_k*w), and Q_N = sum(c_k*mu_k), exact moments give the bound
`abs(integral(H*w)-Q_N) <= W * norm(H-p_N, Inf)`, provided
`W = weight_norm >= integral(abs(w))`. The caller supplies W; it cannot be
inferred from the finite moment vector. This bound concerns interpolation only.

The implementation instead uses the sampled diagnostic
`max(abs(Q_N-Q_M), W*maximum(abs(H(x_j)-p_N(x_j))), 32*eps(Float64)*sum(abs(c_k*mu_k)))`.
Here M is the previous degree and N the current degree. Successive-degree
agreement is useful for a smooth amplitude whose Chebyshev coefficients decay,
but both degrees can share an error. The 18 probe coordinates are
`x_j = left + (right-left)*(j+0.314159)/(17+0.628318)`, j=0:17.
They lie strictly inside the interval, are symmetric and avoid the default
interpolation roots and midpoint. Their number and offset are empirical choices
for an inexpensive check, not a rigorous bound on the unsampled supremum.

The absolute product sum measures sensitivity to cancellation, unlike abs(Q_N).
The factor 32 is the retained empirical roundoff margin for the default degrees
8..64. It is not a degree-dependent error bound and does not certify coefficient
transforms or supplied moments, especially for custom higher degrees. Taking the
maximum combines three warning indicators; these are not separately bounded
error contributions to be added. Unresolved narrow features or aliasing can fool
both degree comparison and probes. Amplitude/phase evaluation errors and moment
callback errors are not independently bounded by any of these terms.

`tolerance_met` means only that this diagnostic is at most
`max(atol, rtol*abs(Q_N))`. It certifies neither the true quadrature error nor the
accuracy of a physical waveform assembled by a caller.
"""
function moment_estimate(problem::MomentProblem;atol=1e-12,rtol=1e-8,previous=nothing)
    isfinite(atol) && atol>=0 && isfinite(rtol) && rtol>=0 || throw(ArgumentError("Invalid moment tolerances."))
    if previous===nothing
        left,right=problem.interval
        probes=ComplexF64[problem.amplitude(left+(right-left)*t) for t in probe_nodes()]
        all(isfinite,probes) || throw(ArgumentError("Nonfinite validation probes."))
        degree_index=2
        low_value=estimate_order(problem,problem.orders[1],probes).value
    else
        previous.degree_index<length(problem.orders) || throw(ArgumentError("Moment degree ladder is exhausted."))
        probes=previous.probes
        degree_index=previous.degree_index+1
        low_value=previous.value
    end
    high=estimate_order(problem,problem.orders[degree_index],probes)
    error_estimate=max(abs(high.value-low_value),high.residual,high.roundoff)
    tolerance=max(atol,rtol*abs(high.value))
    return MomentEstimate(high.value,low_value,error_estimate,tolerance,
        problem.orders[degree_index-1],problem.orders[degree_index],error_estimate<=tolerance,
        degree_index,high.residual,high.roundoff,probes)
end

"""
    integrate_moments(problem; atol=1e-12, rtol=1e-8, max_refinements=length(problem.orders)-2)

Refine until the empirical check passes or the degree/refinement limit is reached.
Return a `MomentEstimate`; exhausted work can leave `tolerance_met=false`.
"""
function integrate_moments(problem::MomentProblem;atol=1e-12,rtol=1e-8,
                           max_refinements::Integer=length(problem.orders)-2)
    max_refinements>=0 || throw(ArgumentError("max_refinements must be nonnegative."))
    estimate=moment_estimate(problem;atol,rtol)
    for _ in 1:min(max_refinements,length(problem.orders)-2)
        estimate.tolerance_met && break
        estimate=moment_estimate(problem;atol,rtol,previous=estimate)
    end
    return estimate
end

end # module Moments
