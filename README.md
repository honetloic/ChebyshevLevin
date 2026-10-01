# ChebyshevLevin.jl

**Alpha v0.1.0-alpha.1 · Julia 1.11+ · MIT**

Chebyshev collocation for scalar first-order Levin equations, including
user-supplied endpoint regularizations:

```math
a_1(x)q'(x) + a_0(x)q(x) = f(x).
```

The package separates equation assembly, solution, integral reconstruction,
and an estimate based on two approximation orders. It has no application-specific
geometry or source model and depends only on Julia's `LinearAlgebra` standard library.

## Installation

After the alpha tag is published:

```julia
using Pkg
Pkg.add(url="https://github.com/honetloic/ChebyshevLevin.git", rev="v0.1.0-alpha.1")
using ChebyshevLevin
```

For development use `Pkg.develop(path="/path/to/ChebyshevLevin")`. Run
`julia scripts/ci.jl` from a checkout to resolve and test in a fresh environment.
Local manifests are not distributed. CI targets Julia 1.11 on Linux, macOS and
Windows; inspect actual CI results before publishing the alpha tag.

## Ordinary oscillatory integral

For `∫ f(x) exp(im*phase(x)) dx`, solve
`q′ + im*phase′*q = f`, then evaluate the primitive at the endpoints.

```julia
using ChebyshevLevin

phase(x) = 25*x
problem = LevinProblem(
    x -> exp(x),                  # right-hand side
    (x -> 25im, x -> 1.0),        # (a0, a1), in derivative order
    (0.0, 1.0),
)

system = assemble_operator(problem; order=12)
# system.nodes, system.matrix, system.rhs

solution = levin_solve(problem; order=12)
solution(0.5)                    # evaluate the auxiliary function q
solution.coefficients            # coefficients of T₀, T₁, ...

value = reconstruct_integral(solution, phase)
exact = expm1(1 + 25im)/(1 + 25im)
```

`order=N` means maximum polynomial degree N and N+1 interior Chebyshev–Gauss
nodes. Derivatives include the scaling from the supplied interval to `[-1,1]`.
`solution(x)` uses Clenshaw recurrence; no polynomial objects are constructed.

## Regularization and endpoint reconstruction

The coefficient functions define only the differential equation. They do not
encode how its solution reconstructs the original integral. After a coordinate
change `r=χ(x)` or scaling `p=s(x)q(x)`, supply the boundary values of the physical
primitive through two functionals:

```julia
value = reconstruct_integral(solution;
    lower = q -> ...,  # primitive at the original lower integration bound
    upper = q -> ...,  # primitive at the original upper integration bound
)
```

The result is `upper(solution) - lower(solution)`. These labels refer to the
original integration orientation, even if the transformed coordinate runs in
reverse. Each functional includes the required scaling and oscillatory factor.
A known endpoint limit can be supplied directly without evaluating a divergent
phase.

For example, let `x=sqrt(1-r)`, `phase(r)=C/x`, and `p=x^2*q`. A manufactured
problem with the smooth solution `q=exp(x)` is:

```julia
C = 10.0
problem = LevinProblem(
    x -> exp(x)*(-x^2/2 - x + im*C/2),
    (x -> -x + im*C/2, x -> -x^2/2),
    (0.0, 1.0),
)

lower = q -> q(1.0)*exp(im*C)  # r=0 corresponds to x=1
upper = q -> 0.0im            # limit as r→1, x→0
solution = levin_solve(problem; order=16)
value = reconstruct_integral(solution; lower, upper)
exact = -exp(1.0)*exp(im*C)
```

The corresponding original integrand has both an inverse-square-root amplitude
singularity and infinitely rapid endpoint oscillations. The transformed
coefficient functions stay finite. The package does not perform this algebraic
regularization automatically.

The positional `phase` reconstruction method is only for an unscaled auxiliary
function in the original coordinate, with finite endpoint phases.

## Error estimate

```julia
estimate = levin_estimate(problem;
    lower, upper,
    order=8,
    higher_order=16,             # defaults to 2*order
    rtol=1e-9,
    atol=1e-12,
)

estimate.value                  # higher-order integral
estimate.low_value
estimate.error_estimate         # abs(value - low_value)
estimate.tolerance              # max(atol, rtol*abs(value))
estimate.tolerance_met
estimate.low_order
estimate.high_order
estimate.low_solution
estimate.high_solution
```

For an ordinary equation, use `levin_estimate(problem, phase; ...)` instead.
Both linear systems are assembled and solved independently. This is a single
comparison, not automatic adaptive refinement. The estimate is not a rigorous
error bound: agreement can miss shared discretization errors, inaccurate input
functions, incorrect endpoint limits, or ill-conditioning. An absolute
tolerance is useful when cancellation makes the integral small.

## Numerical scope

- The equation is scalar; coefficient and RHS callbacks may return real or
  complex numbers. BigFloat calculations are supported.
- The collocation interval must be finite and strictly increasing. Coordinate
  orientation is handled during reconstruction.
- Collocation does not sample endpoints or impose boundary conditions.
  A vanishing derivative coefficient at an endpoint is allowed; selecting a
  suitable regular solution remains part of the problem formulation.
- The solver uses a direct linear solve. It does not apply rank truncation or
  recover automatically from singular or poorly conditioned systems.
- Integral contributions from separately constructed domains can be summed by
  the caller. There is no automatic domain decomposition.

## Files

| File | Responsibility |
|---|---|
| `src/ChebyshevLevin.jl` | Module imports, includes, and exports |
| `src/problems.jl` | Problem/result types and auxiliary-function evaluation |
| `src/operators.jl` | Collocation nodes, basis recurrence, and matrix assembly |
| `src/solve.jl` | Fixed-order linear solve |
| `src/reconstruction.jl` | Endpoint reconstruction |
| `src/estimates.jl` | Two-order comparison |

## Examples and tests

Run from the project directory:

```sh
julia --project=. examples/generalized_levin.jl
julia --project=. -e 'using Pkg; Pkg.test()'
```

The example covers an ordinary integral and a regularized singular endpoint,
prints estimated and actual errors, and checks the results against exact answers.
The test suite covers assembly, solution, reconstruction, and error estimates.

## Migration from the original API

The old `chebyshev_collocation`, `M_func`, `create_M`, `create_f`,
`chebyshev_levin_coefficients`, and `chebyshev_levin` functions have been removed.
Use `assemble_operator`, `levin_solve`, and `reconstruct_integral` instead.
For the old ordinary equation, supply `(x -> im*phase_prime(x), x -> 1)` as the
operator coefficients. The new `order` is the actual polynomial degree, not the
old `points` parameter. The `Polynomials` dependency and legacy example were
also removed.
