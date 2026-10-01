using Test
using LinearAlgebra
using ChebyshevLevin

@testset "Two-order integral estimates" begin
    # q=exp(x), primitive x^2*q(x): exact integral e, degenerate endpoint.
    p=LevinProblem(x->(x+2)*exp(x),(x->2.,x->x),(0.,1.))
    lower=q->0.
    upper=q->q(1.)
    coarse=levin_estimate(p;lower,upper,order=1,rtol=1e-12)
    fine=levin_estimate(p;lower,upper,order=8,rtol=1e-8)
    @test coarse isa LevinEstimate
    @test !coarse.tolerance_met
    @test fine.tolerance_met
    @test abs(fine.value-exp(1.)) < abs(coarse.value-exp(1.))
    @test fine.value ≈ exp(1.) rtol=1e-12
    @test fine.error_estimate < coarse.error_estimate
    @test (fine.low_order,fine.high_order) == (8,16)
    @test length(fine.low_solution.coefficients) == 9
    @test length(fine.high_solution.coefficients) == 17
    @test fine.error_estimate == abs(fine.value-fine.low_value)
    @test fine.value == reconstruct_integral(fine.high_solution;lower,upper)
    @test fine.low_value == reconstruct_integral(fine.low_solution;lower,upper)

    # Ordinary oscillatory reconstruction, including a degree-zero lower solve.
    omega=7.
    ordinary=LevinProblem(x->1.,(x->im*omega,x->1.),(0.,2.))
    est=levin_estimate(ordinary,x->omega*x;order=0,higher_order=2,rtol=1e-12)
    @test est.value ≈ (exp(2im*omega)-1)/(im*omega)
    @test est.tolerance_met

    # Reversed-coordinate endpoint functionals, zero singular-endpoint limit.
    C=3.
    reversed=LevinProblem(x->-x+im*C/2,(x->-x+im*C/2,x->-x^2/2),(0.,1.))
    est=levin_estimate(reversed;lower=q->q(1.)*exp(im*C),upper=q->0im,
                       order=2,higher_order=5,rtol=1e-12)
    @test est.value ≈ -exp(im*C)
    @test est.tolerance_met
    @test (est.low_order,est.high_order) == (2,5)

    # Absolute tolerance permits comparison of a cancellation-dominated integral.
    cancel=levin_estimate(p;lower=q->q(0.),upper=q->q(1.)-expm1(1.),
                          order=8,rtol=0.,atol=1e-8)
    @test abs(cancel.value) < 1e-12
    @test cancel.tolerance_met
    @test cancel.tolerance == 1e-8

    @test_throws ArgumentError levin_estimate(p;lower,upper,order=-1)
    @test_throws ArgumentError levin_estimate(p;lower,upper,order=2,higher_order=2)
    @test_throws ArgumentError levin_estimate(p;lower,upper,rtol=-1)
    @test_throws ArgumentError levin_estimate(p;lower,upper,atol=NaN)
    @test_throws ArgumentError levin_estimate(p;lower,upper,rtol=Inf)
    @test_throws ArgumentError levin_estimate(p;lower,upper=q->NaN,order=1)
    singular=LevinProblem(x->1.,(x->0.,x->1.),(0.,1.))
    @test_throws SingularException levin_estimate(singular;lower,upper,order=2)

    setprecision(128) do
        bp=LevinProblem(x->big"2"+3*x,(x->big"2",x->x),(big"0",big"1"))
        e=levin_estimate(bp;lower=q->big"0",upper=q->q(big"1"),
                         order=2,rtol=big"1e-30",atol=big"0")
        @test e.value isa BigFloat
        @test e.error_estimate isa BigFloat
        @test e.tolerance isa BigFloat
        @test isapprox(e.value,big"2";atol=big"1e-30")
        @test e.tolerance_met
    end
end
