using Test
using ChebyshevLevin

@testset "ChebyshevLevin" begin
    include("operator_assembly.jl")
    include("fixed_order_solver.jl")
    include("reconstruction.jl")
    include("error_estimates.jl")
end
