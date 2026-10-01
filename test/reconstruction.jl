using Test
using ChebyshevLevin

@testset "Integral reconstruction" begin
    # Constant amplitude and linear phase: exact elementary integral.
    omega=7.
    problem=LevinProblem(x->1.,(x->im*omega,x->1.),(0.,2.))
    solution=levin_solve(problem;order=0)
    expected=(exp(2im*omega)-1)/(im*omega)
    @test reconstruct_integral(solution,x->omega*x) ≈ expected
    calls=zeros(Int,2)
    lower=q->begin calls[1]+=1; q(0.) end
    upper=q->begin calls[2]+=1; q(2.)*exp(2im*omega) end
    @test reconstruct_integral(solution;lower,upper) ≈ expected
    @test calls == [1,1]

    # Logarithmic phase, scaled primitive p=y^2*q and q=1+y.
    # The integrand is the derivative of y^(2+im*kappa)*(1+y).
    kappa=.4
    problem=LevinProblem(y->2+im*kappa+(3+im*kappa)*y,
                         (y->2+im*kappa,y->y),(0.,1.))
    solution=levin_solve(problem;order=4)
    phase_calls=Ref(0)
    phase(y)=begin
        y>0 || error("The singular endpoint phase must not be evaluated")
        phase_calls[]+=1
        kappa*log(y)
    end
    value=reconstruct_integral(solution;
        lower=q->0.0im,
        upper=q->1.0^2*q(1.)*exp(im*phase(1.)))
    @test value ≈ 2 atol=1e-12
    @test phase_calls[] == 1

    # Reversed coordinate r=1-x^2 and primitive p=x^2*q, q=1.
    # Phase is C/x; the physical upper-endpoint primitive tends to zero.
    C=3.
    problem=LevinProblem(x->-x+im*C/2,
                         (x->-x+im*C/2,x->-x^2/2),(0.,1.))
    solution=levin_solve(problem;order=4)
    value=reconstruct_integral(solution;
        lower=q->q(1.)*exp(im*C), upper=q->0.0im)
    @test value ≈ -exp(im*C) atol=1e-12
    # Reversing the original integration bounds changes the sign.
    @test reconstruct_integral(solution;
        lower=q->0.0im, upper=q->q(1.)*exp(im*C)) ≈ -value

    setprecision(128) do
        omega=big"3"
        p=LevinProblem(x->big"1",(x->im*omega,x->big"1"),(big"0",big"1"))
        s=levin_solve(p;order=0)
        z=reconstruct_integral(s,x->omega*x)
        @test z isa Complex{BigFloat}
        @test isapprox(z,(exp(im*omega)-1)/(im*omega);atol=big"1e-30")
    end
end
