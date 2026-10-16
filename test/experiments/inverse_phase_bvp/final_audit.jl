using LinearAlgebra,Printf
include("bvp_prototype.jl")
include(joinpath(@__DIR__, "..", "..", "moment_reference.jl"))
using .MomentBVPPrototype
BLAS.set_num_threads(1);setprecision(BigFloat,256)
cases=((0,.1,256,:zero),(1,-.1,256,:zero),
       (64,1e-16,80,:small1),(64,-1e-16,80,:small1),
       (64,1e-8,80,:small2),(64,-1e-8,80,:small2),
       (256,1e-12,272,:small1),(256,1e-8,257,:small2),
       (64,1e-6,80,:small2),(64,1e-6,8192,:zero),
       (64,1e-4,8192,:zero),(64,-1e-4,8192,:zero),
       (64,.1,384,:zero),(128,.5,384,:zero),(256,.5,512,:zero),
       (64,510.61945248368005,80,:zero))

function hompoly(eta,n)
    z=complex(BigFloat(0),BigFloat(eta));term=BigFloat(1)+zero(z);value=term
    for j in 1:n
        term*=2BigFloat(n*n-(j-1)^2)/(BigFloat(j)*(2j-1)*(j+1))*z
        value+=term
    end
    return isodd(n) ? -value : value
end

# Verify the exact terminal-sensitivity formula with an independently evaluated
# polynomial, at a size where 768-bit evaluation easily covers cancellation.
setprecision(BigFloat,768) do
    solve=bvp_solve(.1,16,32;T=BigFloat,scaling=:none)
    expected=[hompoly(.1,n)/hompoly(.1,32) for n in 0:16]
    @assert maximum(abs,solve.sensitivity-expected)<big"1e-180"
    @assert maximum(abs,(bvp_solve(.1,16,32;T=BigFloat,condition=:small2,scaling=:none).values-solve.values)-
        solve.sensitivity*small_terminal(BigFloat,.1,32;degree=2))<big"1e-180"
end
println("PASS: exact boundary-sensitivity identity and perturbation response")

open(joinpath(@__DIR__,"final_audit.csv"),"w") do io
    println(io,"N,eta,K,terminal,error_float,error_big256,float_minus_big256,big256_minus_big384,reference_precision_gap,terminal_error_bound,residual_float,accuracy_met,bound_below_target")
    for (N,eta,K,condition) in cases
        bits=max(1536,4N+512+ceil(Int,2abs(eta)/log(2)))
        reference=series_reference(eta,N;bits)
        refined_reference=series_reference(eta,N;bits=bits+256)
        reference_gap=maximum(abs,reference-refined_reference)
        @assert reference_gap<big"1e-50"
        ordinary=bvp_solve(eta,N,K;condition,scaling=:none)
        precise=setprecision(BigFloat,256) do
            bvp_solve(eta,N,K;condition,scaling=:none,T=BigFloat)
        end
        refined=setprecision(BigFloat,384) do
            bvp_solve(eta,N,K;condition,scaling=:none,T=BigFloat)
        end
        ef=maximum(abs,ordinary.values-refined_reference)
        eb=maximum(abs,precise.values-refined_reference)
        gap=maximum(abs,ordinary.values-precise.values)
        precision_gap=maximum(abs,precise.values-refined.values)
        @assert precision_gap<big"1e-50"
        # An analytic bound evaluated in 384-bit arithmetic is an audit of
        # truncation. It is NOT a certified Float64 stopping criterion.
        @assert eb<=refined.truncation_bound+big"1e-50"
        met=ef<1e-14; bounded=refined.truncation_bound<1e-14
        @printf(io,"%d,%.17g,%d,%s,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g,%s,%s\n",N,eta,K,string(condition),ef,eb,gap,precision_gap,reference_gap,refined.truncation_bound,ordinary.residual,string(met),string(bounded))
        flush(io)
        @printf("N=%d eta=%.5g K=%d %s: Float=%.3e trunc=%.3e roundoff=%.3e bound=%.3e accuracy=%s\n",N,eta,K,string(condition),ef,eb,gap,refined.truncation_bound,string(met))
        flush(stdout)
    end
end
println("Completed bounded pilot; failed accuracy cases are retained, not asserted away.")
