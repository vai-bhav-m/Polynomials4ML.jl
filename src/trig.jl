export CTrigBasis


"""
Complex trigonometric polynomials up to degree `N` (inclusive). The basis is 
constructed in the order 
```
[1, exp(im*θ), exp(-im*θ), exp(2im*θ), exp(-2im*θ), ..., 
                                exp(N*im*θ), exp(-N*im*θ) ]
```
where `θ` is input variable. 
"""
struct CTrigBasis <: AbstractPoly4MLBasis
   N::Int
   @reqfields
end

CTrigBasis(N::Integer) = CTrigBasis(N, _make_reqfields()...)

natural_indices(basis::CTrigBasis) = -basis.N:basis.N 

index(basis::CTrigBasis, m::Integer) = 
         2 * abs(m) + (sign(m) <= 0 ? 1 : 0)


Base.length(basis::CTrigBasis) = 2 * basis.N + 1 

_valtype(basis::CTrigBasis, T::Type{<: Real}) = complex(T)

            
# ----------------- main evaluation code 

function evaluate!(P::AbstractArray, basis::CTrigBasis, x) 
   N = basis.N 
   @assert length(P) >= length(basis) # 2N+1
   @inbounds P[1] = 1 
   @inbounds if N > 0 
      s, c = sincos(x)
      z = Complex(c, s)
      zi = Complex(c, -s)
      P[2] = z
      P[3] = zi
      for n = 2:N 
         P[2*n] = z * P[2*n-2]
         P[2*n+1] = zi * P[2*n-1]
      end
   end
   return P 
end

function evaluate_ed!(P::AbstractArray, dP::AbstractArray, basis::CTrigBasis, x) 
   N = basis.N 
   @assert length(P) >= length(basis) # 2N+1
   @assert length(dP) >= length(basis) 
   @inbounds P[1] = 1 
   @inbounds dP[1] = 0 
   @inbounds if N > 0 
      s, c = sincos(x)
      z = Complex(c, s)
      zi = Complex(c, -s)
      P[2] = z
      dP[2] = im * z 
      P[3] = zi
      dP[3] = -im * zi 
      for n = 2:N 
         P[2*n] = z * P[2*n-2]
         dP[2*n] = im * n * P[2*n]
         P[2*n+1] = zi * P[2*n-1]
         dP[2*n+1] = -im * n * P[2*n+1]
      end
   end
   return P, dP 
end

function evaluate_ed2!(P::AbstractArray, dP::AbstractArray, ddP::AbstractArray, basis::CTrigBasis, x) 
   N = basis.N 
   @assert length(P) >= length(basis) # 2N+1
   @assert length(dP) >= length(basis) 
   @assert length(ddP) >= length(basis) 
   @inbounds P[1] = 1 
   @inbounds dP[1] = 0 
   @inbounds ddP[1] = 0 
   @inbounds if N > 0 
      s, c = sincos(x)
      z = Complex(c, s)
      zi = Complex(c, -s)
      P[2] = z
      dP[2] = im * z 
      ddP[2] = - z 
      P[3] = zi
      dP[3] = -im * zi 
      ddP[3] = - zi 
      for n = 2:N 
         P[2*n] = z * P[2*n-2]
         dP[2*n] = im * n * P[2*n]
         ddP[2*n] = - n^2 * P[2*n]
         P[2*n+1] = zi * P[2*n-1]
         dP[2*n+1] = -im * n * P[2*n+1]
         ddP[2*n+1] = - n^2 * P[2*n+1]
      end
   end
   return P, dP, ddP 
end



function evaluate!(P::AbstractArray, basis::CTrigBasis, X::AbstractVector)
   N = basis.N 
   nX = length(X) 
   @assert size(P, 1) >= length(X) 
   @assert size(P, 2) >= length(basis)  # 2N+1

   @inbounds begin 
      for i = 1:nX 
         P[i, 1] = 1 
         s, c = sincos(X[i])
         P[i, 2] = Complex(c, s)
         P[i, 3] = Complex(c, -s)
      end 
      for n = 2:N 
         for i = 1:nX 
            P[i, 2n] = P[i, 2] * P[i, 2n-2]
            P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
         end
      end
   end
   return P 
end



function evaluate_ed!(P::AbstractArray, dP::AbstractArray, basis::CTrigBasis, X::AbstractVector)
   N = basis.N 
   nX = length(X) 
   @assert size(P, 1) >= length(X) 
   @assert size(P, 2) >= length(basis)  # 2N+1
   @assert size(dP, 1) >= length(X) 
   @assert size(dP, 2) >= length(basis)

   @inbounds begin 
      for i = 1:nX 
         P[i, 1] = 1 
         dP[i, 1] = 0
         s, c = sincos(X[i])
         P[i, 2] = Complex(c, s)
         dP[i, 2] = Complex(-s, c)
         P[i, 3] = Complex(c, -s)
         dP[i, 3] = Complex(-s, -c)
      end 
      for n = 2:N 
         for i = 1:nX 
            P[i, 2n] = P[i, 2] * P[i, 2n-2]
            dP[i, 2n] = im * n * P[i, 2n]
            P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
            dP[i, 2n+1] = - im * n * P[i, 2n+1] 
         end
      end
   end
   return P, dP 
end



function evaluate_ed2!(P::AbstractArray, dP::AbstractArray, ddP::AbstractArray, basis::CTrigBasis, X::AbstractVector)
   N = basis.N 
   nX = length(X) 
   @assert size(P, 1) >= length(X) 
   @assert size(P, 2) >= length(basis)  # 2N+1
   @assert size(dP, 1) >= length(X) 
   @assert size(dP, 2) >= length(basis)
   @assert size(ddP, 1) >= length(X) 
   @assert size(ddP, 2) >= length(basis)

   @inbounds begin 
      for i = 1:nX 
         P[i, 1] = 1 
         dP[i, 1] = 0 
         ddP[i, 1] = 0
         s, c = sincos(X[i])
         P[i, 2] = Complex(c, s)
         dP[i, 2] = Complex(-s, c)
         ddP[i, 2] = Complex(-c, -s)
         P[i, 3] = Complex(c, -s)
         dP[i, 3] = Complex(-s, -c)
         ddP[i, 3] = Complex(-c, s)
      end 
      for n = 2:N 
         for i = 1:nX 
            P[i, 2n] = P[i, 2] * P[i, 2n-2]
            dP[i, 2n] = im * n * P[i, 2n]
            ddP[i, 2n] = im * n * dP[i, 2n]
            P[i, 2n+1] = P[i, 3] * P[i, 2n-1]
            dP[i, 2n+1] = - im * n * P[i, 2n+1] 
            ddP[i, 2n+1] = - im * n * dP[i, 2n+1] 
         end
      end
   end
   return P, dP, ddP 
end

