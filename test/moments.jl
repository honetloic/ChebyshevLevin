using Test

# A different weight and shifted interval exercise the supplied-moment
# interface without Kerr physics or a Levin equation.
using ChebyshevLevin.Moments

@testset "Generic supplied-moment integration" begin
    left,right=-0.3,0.7
    midpoint,half_width=(left+right)/2,(right-left)/2
    plain_moment(k)=isodd(k) ? 0. : (right-left)/(1-k*k)
    # Weight 1+x: x=midpoint+half_width*t and t*T_k=(T_(k+1)+T_|k-1|)/2.
    moments(order)=[(1+midpoint)*plain_moment(k)+
        half_width*(plain_moment(k+1)+plain_moment(abs(k-1)))/2 for k in 0:order]
    rate=2+3im
    problem=MomentProblem(x->exp(rate*x),(left,right),moments;
        weight_norm=(right-left)*(1+midpoint),orders=(2,4,8,16,24))
    primitive(x)=exp(rate*x)*((1+x)/rate-1/rate^2)
    first_estimate=integrate_moments(problem;atol=1e-12,rtol=0.,max_refinements=0)
    result=integrate_moments(problem;atol=1e-12,rtol=0.)
    @test !first_estimate.tolerance_met
    @test result.tolerance_met
    @test result.value ≈ primitive(right)-primitive(left) atol=1e-12 rtol=0
    refined=moment_estimate(problem;atol=1e-12,rtol=0.,previous=first_estimate)
    @test refined.probes === first_estimate.probes
    @test refined.low_value == first_estimate.value
    @test refined.high_order == 8
    @test_throws ArgumentError integrate_moments(problem;max_refinements=-1)
    @test_throws ArgumentError moment_estimate(problem;atol=-1.)
    @test_throws ArgumentError MomentProblem(identity,(1.,0.),moments;weight_norm=1.)
    @test_throws ArgumentError MomentProblem(identity,(0.,1.),moments;weight_norm=-1.)
    bad=MomentProblem(identity,(0.,1.),order->fill(NaN,order+1);weight_norm=1.)
    @test_throws ArgumentError integrate_moments(bad)
end

include("moment_reference.jl")

@testset "Float64 inverse-phase moments against independent series" begin
    # One negative-detuning reference here and both signs in the exact-primitive
    # tests below independently check the phase sign; conjugacy alone would not.
    for (N,eta) in ((8,1e-16),(16,1e-8),(32,-.1),(64,1e-12),
                    (64,.1),(64,1.),(128,1e-8),(128,.1),
                    (128,1.),(256,.5),(256,1e-8),(256,.1),(256,1.),(64,30.),(8,510.61945248368005),
                    (24,510.61945248368005),(64,510.61945248368005))
        reference=series_reference(eta,N)
        weights=inverse_phase_weights(eta,1.,N)
        @test eltype(weights) === ComplexF64
        @test maximum(abs,weights-reference)<1e-14
        @test inverse_phase_weights(-eta,1.,N)==conj.(weights)
    end
    # Roundoff/conditioning in the test reference itself is checked separately.
    a=series_reference(1.,256;bits=1280)
    b=series_reference(1.,256;bits=1536)
    @test maximum(abs,a-b)<big"1e-30"
    for eta in (1e-16,1e-8,.1,1.)
        weights=inverse_phase_weights(eta,1.,64)
        refined=inverse_phase_weights(eta,1.,64;moment_atol=5e-15)
        @test maximum(abs,weights-refined)<1e-14
    end
end

@testset "Degree-dependent contour switch" begin
    # Compare each branch and the boundary against the independent defining
    # moment reference, including the low degrees and maximum accepted degree.
    for N in (0,1,2,8,16,64,256)
        threshold=1/max(2,N*N)
        for eta in (prevfloat(threshold),threshold,nextfloat(threshold))
            weights=inverse_phase_weights(eta,1.,N)
            @test maximum(abs,weights-series_reference(eta,N))<1e-14
        end
    end
    # A long real prefix exhausted this budget before the earlier-turn change.
    # Check bounded numerical work instead of machine-dependent wall-clock time.
    weights=inverse_phase_weights(.5,1.,256;maxevals=20_000)
    @test maximum(abs,weights-series_reference(.5,256))<1e-14
end

@testset "Inverse-phase moments: exact primitives and scaling" begin
    b=.31
    for beta in (-30.,-.3,-1e-6,0.,1e-6,.3,30.)
        weights=inverse_phase_weights(beta,b,16)
        reference=b*series_reference(beta/b,16)
        @test maximum(abs,weights-reference)/b<1e-14
        for q in (2,3,6,12)
            H(x)=q*x^(q-1)+im*beta*x^(q-2)
            coeffs=Moments.chebyshev_coefficients(H,(0.,b),16)
            @test sum(coeffs.*weights) ≈ b^q*cis(-beta/b) atol=2e-13 rtol=2e-13
        end
    end
    weights=inverse_phase_weights(0.,b,256)
    for k in 0:256
        @test weights[k+1] == (isodd(k) ? 0. : b/(1-k*k))
    end
end

@testset "Moment accuracy requests and bounded failure" begin
    @test_throws ArgumentError inverse_phase_weights(Inf,1.,4)
    @test_throws ArgumentError inverse_phase_weights(1.,0.,4)
    @test_throws ArgumentError inverse_phase_weights(1.,1.,257)
    @test_throws ArgumentError inverse_phase_weights(1.,1.,-1)
    @test_throws ArgumentError inverse_phase_weights(1.,1.,4;moment_atol=0.)
    @test_throws ArgumentError inverse_phase_weights(1.,1.,4;moment_atol=Inf)
    @test_throws ArgumentError inverse_phase_weights(1.,1.,4;moment_atol=eps()/2)
    @test_throws ArgumentError inverse_phase_weights(1.,1.,4;maxevals=1)
    @test_throws ArgumentError inverse_phase_weights(floatmax(Float64),floatmin(Float64),4)
    @test_throws ErrorException inverse_phase_weights(1.,1.,256;maxevals=42)
    @test_throws ErrorException inverse_phase_weights(.1,1.,64;moment_atol=eps(),maxevals=4000)
end
