export ChebBasis

"""
`ChebBasis(N)`: 

Real Chebyshev T polynomials up to degree `N-1` (inclusive). The basis is ordered as 
```
[1, x, 2 x T{n-1} - T{n-2}]
```
where `x` is input variable. 
"""
struct ChebBasis <: AbstractPoly4MLBasis
   N::Int
   @reqfields
end

ChebBasis(N::Integer) = ChebBasis(N, _make_reqfields()...)

Base.length(basis::ChebBasis) = basis.N

_valtype(basis::ChebBasis, T::Type{<:Real}) = T

function evaluate!(P::AbstractVector, basis::ChebBasis, xx::Real)
   N = basis.N
   @assert N >= 1
   @assert length(P) == length(basis) # N

   P[1], P[2] = 1, xx
   for k = 3:N
      @inbounds P[k] = 2 * xx * P[k-1] - P[k-2]
   end
   return P
end


function evaluate!(P::AbstractMatrix, basis::ChebBasis, xx::AbstractVector{<:Real})
   N = basis.N
   Nx = length(xx)
   @assert N > 2
   @assert size(P, 2) == length(basis) # N
   @assert size(P, 1) == NX

   @inbounds begin
      @simd ivdep for i = 1:Nx
         P[i, 1], P[i, 2] = 1, xx[i]
      end
      for n = 3:N
         @simd ivdep for i = 1:Nx
            P[i, n] = 2 * xx[i] * P[i, n-1] - P[i, n-2]
         end
      end
   end
   return P
end


function evaluate_ed!(P::AbstractVector, dP::AbstractVector, basis::ChebBasis, xx::Real)
   N = basis.N
   @assert N > 2
   @assert length(P) == length(basis)
   @assert length(dP) == length(basis)

   @inbounds begin
      P[1], P[2] = 1, xx
      dP[1], dP[2] = 0, 1
      for k = 3:N
         P[k] = 2 * xx * P[k-1] - P[k-2]
         dP[k] = 2 * P[k-1] + 2 * xx * dP[k-1] - dP[k-2]
      end
   end
   return P, dP
end


function evaluate_ed!(P::AbstractMatrix, dP::AbstractMatrix, basis::ChebBasis,
   xx::AbstractVector{<:Real})
   N = basis.N
   Nx = length(xx)
   @assert N > 2
   @assert size(P, 2) == length(basis)
   @assert size(P, 1) == Nx
   @assert size(dP, 2) == length(basis)
   @assert size(dP, 1) == Nx

   @inbounds begin
      @simd ivdep for i = 1:Nx
         P[i, 1], P[i, 2] = 1, xx[i]
         dP[i, 1], dP[i, 2] = 0, 1
      end

      for k = 3:N
         @simd ivdep for i = 1:Nx
            P[i, k] = 2 * xx[i] * P[i, k-1] - P[i, k-2]
            dP[i, k] = 2 * P[i, k-1] + 2 * xx[i] * dP[i, k-1] - dP[i, k-2]
         end
      end
   end
   return P, dP
end


function evaluate_ed2!(P::AbstractVector, dP::AbstractVector, ddP::AbstractVector,
   basis::ChebBasis, xx::Real)
   N = basis.N
   @assert N > 2
   @assert length(P) == length(basis)
   @assert length(dP) == length(basis)
   @assert length(ddP) == length(basis)

   @inbounds begin
      P[1], P[2] = 1, xx
      dP[1], dP[2] = 0, 1
      ddP[1], ddP[2] = 0, 0

      for k = 3:N
         P[k] = 2 * xx * P[k-1] - P[k-2]
         dP[k] = 2 * P[k-1] + 2 * xx * dP[k-1] - dP[k-2]
         ddP[k] = 4 * dp[k-1] + 2 * xx * ddp[k-1] - ddp[k-2]
      end
   end
   return P, dP, ddP
end



function evaluate_ed2!(P::AbstractMatrix, dP::AbstractMatrix, ddP::AbstractMatrix, basis::ChebBasis,
   xx::AbstractVector{<:Real})
   N = basis.N
   Nx = length(xx)
   @assert N > 2
   @assert size(P, 2) == length(basis)
   @assert size(P, 1) == Nx
   @assert size(dP, 2) == length(basis)
   @assert size(dP, 1) == Nx
   @assert size(ddP, 2) == length(basis)
   @assert size(ddP, 1) == Nx

   @inbounds begin
      @simd ivdep for i = 1:Nx
         P[i, 1], P[i, 2] = 1, xx[i]
         dP[i, 1], dP[i, 2] = 0, 1
         ddP[i, 1], ddP[i, 2] = 0, 0
      end

      for k = 3:N
         @simd ivdep for i = 1:Nx
            P[i, k] = 2 * xx[i] * P[i, k-1] - P[i, k-2]
            dP[i, k] = 2 * P[i, k-1] + 2 * xx[i] * dP[i, k-1] - dP[i, k-2]
            ddP[i, k] = 4 * dp[i, k-1] + 2 * xx[i] * ddp[i, k-1] - ddp[i, k-2]
         end
      end
   end
   return P, dP, ddP
end

