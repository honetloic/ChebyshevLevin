using LinearAlgebra,Printf
include("bvp_prototype.jl")
using .MomentBVPPrototype
BLAS.set_num_threads(1)

# Exact zero-detuning moments provide a separate analytic reference and check
# that the logarithmic-terminal branch handles eta=0 without evaluating log(0).
open(joinpath(@__DIR__,"zero_audit.csv"),"w") do io
    println(io,"N,K,error_float,float_minus_big,error_big,accuracy_met")
    for N in (0,1,64,256)
        K=max(N+16,4)
        ordinary=bvp_solve(0.0,N,K;condition=:small1,scaling=:none)
        precise,reference=setprecision(BigFloat,384) do
            bvp_solve(0.0,N,K;condition=:small1,scaling=:none,T=BigFloat),
            [complex(moment_zero(BigFloat,n)) for n in 0:N]
        end
        ef=maximum(abs,ordinary.values-reference)
        gap=maximum(abs,ordinary.values-precise.values)
        eb=maximum(abs,precise.values-reference)
        met=ef<1e-14
        @assert eb<big"1e-100"
        @printf(io,"%d,%d,%.17g,%.17g,%.17g,%s\n",N,K,ef,gap,eb,string(met))
        @printf("N=%d K=%d exact zero: Float=%.3e Big=%.3e accuracy=%s\n",N,K,ef,eb,string(met))
    end
end
