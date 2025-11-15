module ChebyshevLevin

using LinearAlgebra
using Polynomials

export chebyshev_collocation
export chebyshev_levin_coefficients
export chebyshev_levin
export create_M, create_f, M_func

"""
    chebyshev_collocation(a, b, n)

Generate `n` Chebyshev collocation nodes in the interval [a, b].

The nodes are based on the zeros of the Chebyshev polynomial of the first kind,
transformed from the standard interval [-1, 1] to [a, b].

# Arguments
- `a`: Left endpoint of the interval
- `b`: Right endpoint of the interval
- `n`: Number of collocation points

# Returns
- Vector of `n` Chebyshev collocation nodes in [a, b]

# Example
```julia
nodes = chebyshev_collocation(0.0, 1.0, 10)
```
"""
function chebyshev_collocation(a, b, n)
    # Chebyshev nodes in [-1, 1]
    nodes_01 = cos.((2 .* (0:n-1) .+ 1) .* π ./ (2*n))
    # Transform to [a, b]
    nodes = a .+ (b - a) .* (nodes_01 .+ 1) / 2
    return nodes
end

"""
    M_func(omegadg, a, b)

Create the operator function M for Levin's method.

This function generates the differential operator M that acts on Chebyshev polynomials
for solving the equation M[p] = f in Levin's method for oscillatory integrals.

# Arguments
- `omegadg`: Function representing the derivative of the phase function ω'(r)
- `a`: Left endpoint of the interval
- `b`: Right endpoint of the interval

# Returns
- Function `(r, n) -> M[Tₙ](r)` where Tₙ is the n-th Chebyshev polynomial

# Mathematical Background
For an oscillatory integral ∫ f(r) exp(iω(r)) dr, Levin's method solves:
    M[p](r) = iω'(r)p(r) + p'(r) = f(r)
where p is represented as a sum of Chebyshev polynomials.
"""
function M_func(omegadg::F, a, b) where F
    return (r, n) -> ChebyshevT(1, n)(-1 + 2 * (r - a) / (b - a)) * im * omegadg(r) + 
                     2 / (b - a) * derivative(ChebyshevT(1, n))(-1 + 2 * (r - a) / (b - a))
end

"""
    create_M(omegadg, n, a, b)

Construct the matrix M for the Chebyshev-Levin collocation system.

# Arguments
- `omegadg`: Function representing the derivative of the phase function ω'(r)
- `n`: Number of Chebyshev basis functions (and collocation points)
- `a`: Left endpoint of the interval
- `b`: Right endpoint of the interval

# Returns
- Matrix of size `n × n` where element (i,j) is M[Tⱼ](rᵢ), evaluated at collocation point rᵢ

# Details
The matrix M is used in the linear system M·c = f, where c are the Chebyshev coefficients
of the antiderivative function p.
"""
function create_M(omegadg::F, n, a, b) where F
    nodes = chebyshev_collocation(a, b, n)
    M_f = M_func(omegadg, a, b)
    M_return = [M_f(nodes[i], j) for i in 1:n, j in 0:n-1]
    return M_return
end

"""
    create_f(f, n, a, b)

Evaluate the integrand function f at Chebyshev collocation points.

# Arguments
- `f`: The amplitude function in the oscillatory integral
- `n`: Number of collocation points
- `a`: Left endpoint of the interval
- `b`: Right endpoint of the interval

# Returns
- Vector of length `n` containing f evaluated at each collocation point
"""
function create_f(f::F, n, a, b) where F
    nodes = chebyshev_collocation(a, b, n)
    f_return = f.(nodes)
    return f_return
end

"""
    chebyshev_levin_coefficients(f, omegadg, a, b, points=10)

Compute the Chebyshev coefficients of the antiderivative function using Levin's method.

# Arguments
- `f`: The amplitude function in the oscillatory integral ∫ f(r) exp(iω(r)) dr
- `omegadg`: Function representing the derivative of the phase ω'(r)
- `a`: Left endpoint of integration
- `b`: Right endpoint of integration
- `points`: Number of collocation points (default: 10); total basis functions = 2*points + 1

# Returns
- Vector of Chebyshev coefficients for the antiderivative function p(r) as well as the chebyshev coefficients for half of the points to estimate the error.

# Mathematical Background
Solves the linear system M·c = f to find coefficients c such that:
    p(r) = Σᵢ cᵢ Tᵢ(r)
where M[p] = f and M is the differential operator from Levin's method.
"""
function chebyshev_levin_coefficients(f::F, omegadg::G, a, b, points=10) where {F,G}
    n = 2 * points + 1
    M_tab = create_M(omegadg, n, a, b)
    f_tab = create_f(f, n, a, b)

    M_err = @view M_tab[2:2:end, 1:points]
    f_err = @view f_tab[2:2:end]
    
    cheb_coeffs = M_tab \ f_tab
    cheb_errs = M_err \ f_err

    return cheb_coeffs, cheb_errs
end

"""
    chebyshev_levin(f, omegadg, omegag, a, b, points=10)

Compute an oscillatory integral using the Chebyshev-Levin method.

Evaluates the integral:
    I = ∫ₐᵇ f(r) exp(iω(r)) dr

# Arguments
- `f`: The amplitude function
- `omegadg`: Function representing the derivative of the phase ω'(r)
- `omegag`: Function representing the phase ω(r)
- `a`: Left endpoint of integration
- `b`: Right endpoint of integration
- `points`: Number of collocation points (default: 10)

# Returns
- Tuple `(integral_value, cheb_coeffs)` where:
  - `integral_value`: The computed value of the oscillatory integral
  - `cheb_coeffs`: The Chebyshev coefficients of the antiderivative

# Mathematical Background
Uses Levin's method which transforms the oscillatory integral to:
    I = [p(r) exp(iω(r))]ₐᵇ
where p is the antiderivative satisfying M[p] = f.

# Example
```julia
f(r) = r^2
omega(r) = 10*r
omega_prime(r) = 10
result, coeffs = chebyshev_levin(f, omega_prime, omega, 0.0, 1.0, 20)
```
"""
function chebyshev_levin(f::F, omegadg::G, omegag::H, a, b, points=10) where {F,G,H}
    n = 2 * points + 1
    cheb_coeffs, cheb_errs = chebyshev_levin_coefficients(f, omegadg, a, b, points)
    
    # Construct the primitive (antiderivative) as a sum of Chebyshev polynomials
    primitive_integral = sum(cheb_coeffs[i+1] * ChebyshevT(1, i) for i in 0:n-1)
    primitive_integral_err = sum(cheb_errs[i+1] * ChebyshevT(1, i) for i in 0:points-1)
    
    # Oscillating factor
    #oscillating_factor(r) = exp(im * omegag(r))
    
    # Evaluate at boundaries: [p(r)·exp(iω(r))]ₐᵇ
    # Note: Chebyshev polynomials are defined on [-1, 1], so we evaluate at ±1
    #integral_value = primitive_integral(1) * oscillating_factor(b) - 
    #                primitive_integral(-1) * oscillating_factor(a)
    #integral_err = primitive_integral_err(1) * oscillating_factor(b) - 
    #                primitive_integral_err(-1) * oscillating_factor(a)
    #rel_err = abs(integral_value - integral_err) / abs(integral_value)
    rel_err_b = abs(primitive_integral(1) - primitive_integral_err(1)) / abs(primitive_integral(1))
    rel_err_a = abs(primitive_integral(-1) - primitive_integral_err(-1)) / abs(primitive_integral(-1))
    #return (integral_value, rel_err)
    return (primitive_integral(1), rel_err_b, primitive_integral(-1), rel_err_a)
end

end # module
