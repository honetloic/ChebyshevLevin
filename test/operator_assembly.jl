using Test
using ChebyshevLevin

@testset "Generalized operator assembly" begin
    # A known cubic in the Chebyshev coordinate on a non-unit interval.
    a0(x) = 1 + im*x
    a1(x) = x - 2
    q(x) = begin t=(x-4)/2; 1 + 2*t + 3*(2*t^2-1) + 4*(4*t^3-3*t) end
    dq(x) = begin t=(x-4)/2; (2 + 12*t + 4*(12*t^2-3))/2 end
    f(x) = a1(x)*dq(x) + a0(x)*q(x)
    problem = LevinProblem(f, (a0,a1), (2,6.0))
    system = assemble_operator(problem; order=3)
    @test size(system.matrix) == (4,4)
    @test system.matrix * [1,2,3,4] ≈ system.rhs
    @test all(2 .< system.nodes .< 6)
    @test issorted(system.nodes; rev=true)
    @test system.matrix[:,1] ≈ a0.(system.nodes)
    result = LevinResult(problem, [1,2,3,4])
    for x in range(2,6; length=11)
        @test result(x) ≈ q(x) atol=1e-13
    end
    @test_throws DomainError result(1.0)
    @test_throws ArgumentError LevinResult(problem, Float64[])
    @test_throws ArgumentError LevinResult(problem, [NaN])
    @test_throws ArgumentError LevinProblem(f,(a0,a1),(2.,2.))
    @test_throws ArgumentError LevinProblem(f,(a0,a1),(6.,2.))
    @test_throws ArgumentError LevinProblem(f,(a0,a1),(0.,Inf))
    @test_throws ArgumentError assemble_operator(problem; order=-1)

    # Endpoint-degenerate derivative coefficient: no endpoint sampling.
    counts = zeros(Int,3)
    wrap(f,k) = x -> begin
        @test 0 < x < 1
        counts[k] += 1
        f(x)
    end
    p = LevinProblem(wrap(x->1+im*x,1),
                     (wrap(x->im,2),wrap(x->x,3)), (0.,1.))
    s = assemble_operator(p; order=5)
    @test counts == [6,6,6]
    c = ComplexF64[-im+0.5im/(1+im), 0.5im/(1+im),0,0,0,0]
    @test s.matrix*c ≈ s.rhs
    r = LevinResult(p,c)
    @test r(0.) ≈ -im
    @test r(1.) ≈ -im+im/(1+im)

    constant = LevinProblem(x->6., (x->2.,x->9.), (0.,1.))
    s0=assemble_operator(constant;order=0)
    @test s0.matrix == reshape([2.],1,1)
    @test s0.rhs == [6.]
    @test LevinResult(constant,[3.])(0.2) == 3.
    bad = LevinProblem(x->Inf,(x->1.,x->1.),(0.,1.))
    @test_throws ArgumentError assemble_operator(bad)

    bp=LevinProblem(x->big"1",(x->big"2",x->big"3"),(big"0",big"1"))
    @test eltype(assemble_operator(bp;order=4).matrix) == BigFloat
end
