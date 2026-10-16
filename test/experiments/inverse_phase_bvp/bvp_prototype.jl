# Isolated recurrence investigation, 16 October 2026. This file is not included
# by ChebyshevLevin or runtests.jl; it does not replace the contour algorithm.
#
# M_n(eta) = integral_0^1 T_n(2y-1) exp(-i*eta/y) dy. For cutoff K>N, use
# the three lower rows in assemble(), the five-term recurrence at n=2:K-2,
# and one terminal datum m_K=g_K. The exact finite-system error is
#   m_n-M_n = (g_K-M_K)*h_n(i*eta)/h_K(i*eta),
# where h_n(z) = L_z[T_n(2y-1)] and L_z[y^j]=(-z)^j/(j+1)!.
# This polynomial formula is for analysis only, never Float64 evaluation.
#
# Uniqueness for real eta: h_n(z)=(-1)^n Q_n(z), where Q_n is obtained by
# dividing each coefficient of T_n(1+2z) by (j+1)!. This preserves negative
# real roots. One proof approximates the coefficient map by applying
# product_(ell=2)^(m+1) (1+z*d/dz/ell) to p(z/(m+1)); each factor preserves
# nonpositive real roots by Rolle's theorem. Its jth multiplier tends to
# 1/(j+1)!. Q_n(0)=1, so Q_n has no zero root. Thus |h_n(i*eta)|>=1.
# This establishes nonsingularity, not a uniform Float64 conditioning bound.
#
# Terminal choices: zero; or first/second-order small-detuning expansions
# including z*log(z). terminal_bound() bounds their analytic truncation only.
# A second right side measures propagation of terminal error. Its computed
# Float64 value, and a small matrix residual, are not certified error bounds.
#
# Reproduce manually with Julia 1.11.9, from the repository root:
#   julia --startup-file=no test/experiments/inverse_phase_bvp/final_audit.jl
#   julia --startup-file=no test/experiments/inverse_phase_bvp/zero_audit.jl
# Only standard libraries are needed. Runs overwrite the adjacent CSV outputs.
# The independent moment reference is the existing test/moment_reference.jl.
# Final audit: reference at >=1536 bits and +256 bits; finite BVP at 256/384
# bits, compared with Float64; exact-zero audit uses closed parity values.
# The BigFloat BVP is diagnostic, NOT an independent moment reference.
# All final Float64 runs explicitly use scaling=:none; :max preserves the
# exploratory row-scaling option for investigation, not a production default.
#
# Recorded outcome: 14/16 nonzero cases and all 4 exact-zero cases meet 1e-14
# normalized absolute error over n=0:N. Two eta=1e-6 cases fail from terminal
# truncation, including K=8192 with zero terminal data. Failures remain in CSV;
# these runners report research outcomes rather than constitute passing CI.
# The boundary order and K are manually chosen. No uniform accuracy guarantee,
# automatic stopping rule, or speed advantage over contour quadrature is
# established. Production package validation remains a separate review step.
module MomentBVPPrototype
using LinearAlgebra
export bvp_solve, moment_zero, small_terminal, terminal_bound, reference_vectors

moment_zero(::Type{T},n) where T = isodd(n) ? zero(T) : inv(T(1-n*n))

# Exact integral D_n = integral_0^1 (T_n(2y-1)-(-1)^n)/y dy.
# The positive harmonic sum avoids a cancellation-prone degree recurrence.
function endpoint_difference(::Type{T},n) where T
    value=zero(T)
    for j in 1:div(n,2)
        value+=inv(T(2j-1))
    end
    magnitude=4value+(isodd(n) ? 2/T(n) : zero(T))
    return isodd(n) ? magnitude : -magnitude
end

function second_endpoint_difference(::Type{T},n) where T
    previous,current=zero(T),zero(T)
    for j in 1:n-1
        previous,current=current,-2current-previous+4endpoint_difference(T,j)
    end
    return n==0 ? zero(T) : current
end

function small_terminal(::Type{T},eta,K;degree=1) where T
    eta=T(eta)
    eta==0 && return complex(moment_zero(T,K))
    z=complex(zero(T),eta)
    endpoint=isodd(K) ? -one(T) : one(T)
    logarithm=log(z)+T(Base.MathConstants.eulergamma)
    value=moment_zero(T,K)+z*(endpoint*(logarithm-1)-endpoint_difference(T,K))
    if degree==2
        slope=-2T(K)^2*endpoint
        value+=z*z/2*(-endpoint-slope*(logarithm-T(3)/2)+second_endpoint_difference(T,K))
    end
    return value
end

# Absolute analytic terminal-defect bounds for real eta. These bound truncation,
# not floating-point evaluation of the boundary datum or solution.
function terminal_bound(::Type{T},eta,K,condition) where T
    a=abs(T(eta))
    condition===:zero && return one(T) # |M_K|<=1 on the real defining integral.
    condition in (:small1,:small2) || error("Unknown terminal condition")
    a==0 && return zero(T)
    a<=1 || return T(Inf)
    condition===:small1 && return a*a*(exp(a)/2+T(K)^2*(4+log(inv(a))))
    quadratic_bound=T(2)/3*T(K)^2*(T(K)^2-1)
    return a^3*(exp(a)*(T(1)/12+T(K)^2/3)+quadratic_bound*(T(5)/3+log(inv(a))/6))
end

function assemble(::Type{T},eta,K,condition;scaling=:max) where T
    K>=4 || throw(ArgumentError("K must be at least four"))
    z=complex(zero(T),T(eta)); C=Complex{T}
    rows=[Dict{Int,C}() for _ in 1:K+1]
    # Two RHS columns: physical normalization/terminal datum, and sensitivity
    # to adding one to the terminal datum.
    rhs=zeros(C,K+1,2)
    function row!(i,entries)
        scale=scaling===:max ? maximum(abs(last(entry)) for entry in entries) : one(T)
        for (j,value) in entries
            rows[i+1][j+1]=C(value)/scale
        end
        return scale
    end
    scale=row!(0,[(0,1+z),(1,one(C))])
    rhs[1,1]=exp(-z)/scale
    row!(1,[(0,-1+4z),(1,-4-4z),(2,C(-3))])
    row!(2,[(0,-2-4z),(1,-4+8z),(2,-6-4z),(3,C(-4))])
    for n in 2:K-2
        row!(n+1,[(n-2,C(n-3)),(n-1,2n-4-4z),(n,-2+8z),
                  (n+1,-2n-4-4z),(n+2,C(-n-3))])
    end
    row!(K,[(K,one(C))])
    rhs[K+1,1]=condition===:zero ? zero(C) : small_terminal(T,eta,K;degree=condition===:small2 ? 2 : 1)
    rhs[K+1,2]=one(C)
    return rows,rhs
end

function lapack_solve(rows,rhs)
    kl,ku=3,1
    count=length(rows)
    bands=zeros(ComplexF64,2kl+ku+1,count)
    for i in eachindex(rows), (j,value) in rows[i]
        @assert -ku<=i-j<=kl
        bands[kl+ku+1+i-j,j]=value
    end
    factors,pivots=LinearAlgebra.LAPACK.gbtrf!(kl,ku,count,bands)
    return LinearAlgebra.LAPACK.gbtrs!('N',kl,ku,count,factors,pivots,copy(rhs))
end

# Audit solver for BigFloat: independent generic Gaussian elimination with
# partial pivoting on the same banded equations, using sparse row dictionaries.
# This is not used by the Float64 prototype and is not an independent moment reference.
function generic_solve(original_rows,original_rhs)
    rows=deepcopy(original_rows); rhs=copy(original_rhs); count=length(rows)
    C=eltype(rhs)
    for k in 1:count
        lastrow=min(count,k+3)
        pivot=k
        for i in k+1:lastrow
            abs(get(rows[i],k,zero(C)))>abs(get(rows[pivot],k,zero(C))) && (pivot=i)
        end
        diagonal=get(rows[pivot],k,zero(C))
        iszero(diagonal) && error("Singular finite boundary system at column $k")
        if pivot!=k
            rows[k],rows[pivot]=rows[pivot],rows[k]
            rhs[k,:],rhs[pivot,:]=copy(rhs[pivot,:]),copy(rhs[k,:])
        end
        for i in k+1:lastrow
            multiplier=get(rows[i],k,zero(C))/diagonal
            delete!(rows[i],k)
            iszero(multiplier) && continue
            for j in sort!(collect(keys(rows[k])))
                j>k || continue
                rows[i][j]=get(rows[i],j,zero(C))-multiplier*rows[k][j]
            end
            for j in axes(rhs,2)
                rhs[i,j]-=multiplier*rhs[k,j]
            end
        end
    end
    solution=zeros(C,size(rhs))
    for k in count:-1:1
        for column in axes(rhs,2)
            value=rhs[k,column]
            for j in sort!(collect(keys(rows[k])))
                j>k || continue
                value-=rows[k][j]*solution[j,column]
            end
            solution[k,column]=value/rows[k][k]
        end
    end
    return solution
end

function residual_norm(rows,rhs,solution)
    result=zero(real(zero(eltype(rhs))))
    for i in eachindex(rows), c in axes(rhs,2)
        value=-rhs[i,c]
        for j in sort!(collect(keys(rows[i])))
            value+=rows[i][j]*solution[j,c]
        end
        result=max(result,abs(value))
    end
    return result
end

function bvp_solve(eta,N,K;condition=:zero,T=Float64,solver=:default,scaling=:max)
    0<=N<K || throw(ArgumentError("Require 0<=N<K"))
    rows,rhs=assemble(T,eta,K,condition;scaling)
    solution=(T===Float64 && solver!==:generic) ? lapack_solve(rows,rhs) : generic_solve(rows,rhs)
    sigma=maximum(abs,@view solution[1:N+1,2])
    return (;values=solution[1:N+1,1],sensitivity=solution[1:N+1,2],sigma,
            truncation_bound=sigma*terminal_bound(T,eta,K,condition),
            residual=residual_norm(rows,rhs,solution),full_values=solution[:,1])
end

function reference_vectors(path)
    references=Dict{Tuple{Int,Float64},Vector{Complex{BigFloat}}}()
    for line in readlines(path)[2:end]
        fields=split(line,',')
        fields[3]=="contour_candidate" || continue
        key=(parse(Int,fields[1]),parse(Float64,fields[2]))
        push!(get!(references,key,Complex{BigFloat}[]),complex(parse(BigFloat,fields[5]),parse(BigFloat,fields[6])))
    end
    return references
end
end
