# Independent reference only. Series for E1 followed by its integration-by-parts identity.
# Guard precision covers both the series and the subsequent monomial cancellation.
function series_reference(eta::Float64,N::Int;bits=max(768,4N+256+ceil(Int,2abs(eta)/log(2))))
    setprecision(BigFloat,bits) do
        eta==0 && return Complex{BigFloat}[isodd(n) ? 0 : inv(BigFloat(1-n*n)) for n in 0:N]
        z=complex(BigFloat(0),BigFloat(eta))
        term=-z
        series=term
        converged=false
        for k in 2:10_000
            term *= -z/k
            contribution=term/k
            series += contribution
            if k>2abs(z) && abs(contribution)<eps(BigFloat)*max(1,abs(series))/8
                converged=true;break
            end
        end
        converged || error("Reference series work budget exceeded")
        current=-BigFloat(Base.MathConstants.eulergamma)-log(z)-series
        e=Complex{BigFloat}[]
        ex=exp(-z)
        for p in 1:N+1
            current=(ex-z*current)/p
            push!(e,current)
        end
        weights=Complex{BigFloat}[e[1]]
        previous,poly=BigFloat[1],BigFloat[-1,2]
        for n in 1:N
            push!(weights,sum(poly.*e[1:n+1]))
            n==N && break # No T_(N+1) is needed by this reference vector.
            following=zeros(BigFloat,n+2)
            for j in eachindex(poly)
                following[j]-=2poly[j];following[j+1]+=4poly[j]
            end
            following[1:length(previous)].-=previous
            previous,poly=poly,following
        end
        weights
    end
end
