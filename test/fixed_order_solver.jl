using Test
using LinearAlgebra
using ChebyshevLevin

@testset "Fixed-order Levin solver" begin
    # Non-unit interval, complex coefficients, and an independently specified cubic.
    q(x) = (1 + 2im) + (2 - im)*x + x^3
    dq(x) = (2 - im) + 3*x^2
    a0(x) = 3im
    a1(x) = 2.0
    rhs(x) = a1(x)*dq(x) + a0(x)*q(x)
    problem = LevinProblem(rhs, (a0,a1), (2.,5.))
    solution = levin_solve(problem; order=3)
    @test solution isa LevinResult
    @test solution.problem === problem
    @test length(solution.coefficients) == 4
    for x in range(2.,5.; length=13)
        @test solution(x) ≈ q(x) rtol=1e-12
    end

    # A derivative coefficient that vanishes at the left endpoint.
    p = LevinProblem(x -> (1+2im)*(1+x+x^2) + x*(1+2*x),
                     (x -> 1+2im, x -> x), (0.,1.))
    s = levin_solve(p; order=8)
    for x in range(0.,1.; length=11)
        @test s(x) ≈ 1+x+x^2 atol=1e-12
    end

    # Ordinary Levin equation: constant amplitude and linear phase.
    omega = 7.
    ordinary = LevinProblem(x -> 1., (x -> im*omega, x -> 1.), (0.,2.))
    s = levin_solve(ordinary; order=0)
    @test s(0.7) ≈ 1/(im*omega)
    @test s(2.)*exp(2im*omega)-s(0.) ≈ (exp(2im*omega)-1)/(im*omega)

    @test_throws ArgumentError levin_solve(p; order=-1)
    singular = LevinProblem(x -> 1., (x -> 0.,x -> 1.), (0.,1.))
    @test_throws SingularException levin_solve(singular; order=3)

    setprecision(128) do
        bp = LevinProblem(x -> 2+3*x, (x -> big"2",x -> x), (big"0",big"1"))
        bs = levin_solve(bp; order=3)
        @test eltype(bs.coefficients) == BigFloat
        @test isapprox(bs(big"0.3"), big"1.3"; atol=big"1e-30")
    end
end

@testset "Equation scaling invariance" begin
    q(x) = 1 + (2-im)*x + x^3
    dq(x) = 2-im + 3*x^2
    # Same polynomial equation over a 300-decade range of row magnitudes.
    factor(x) = 10.0^(150*x)
    p = LevinProblem(x -> factor(x)*(x*dq(x)+3im*q(x)),
                     (x -> factor(x)*3im, x -> factor(x)*x), (-1.,1.))
    solution = levin_solve(p; order=12)
    for x in range(-1.,1.; length=21)
        @test solution(x) ≈ q(x) atol=1e-11 rtol=1e-11
    end
    # Integer-valued RHS must be promoted during normalization.
    integer_rhs = LevinProblem(x -> 1, (x -> 2., x -> 0.), (0.,1.))
    @test levin_solve(integer_rhs; order=3)(0.3) ≈ 0.5
    zero_operator = LevinProblem(x -> 1., (x -> 0., x -> 0.), (0.,1.))
    @test_throws SingularException levin_solve(zero_operator; order=3)
end
