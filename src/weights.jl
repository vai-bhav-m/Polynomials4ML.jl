
struct DiscreteWeights{T} 
   X::T 
   W::T 
end

function DiscreteWeights(X, W, normalize=false)
   T = promote_type(eltype(X), eltype(W))
   @assert all(isreal, W)
   if minimum(W) < 0 
      @warn("Found negative weights.")
   end
   if normalize in [true, :normalize] 
      W = W ./ sum(W)
   end
   return DiscreteWeights{T}(T.(X), T.(W))
end


function orthpolys(N::Integer, W::DiscreteWeights{TW}; TX = Float64) where {TW} 
   @assert N > 0 

   T = promote_type(TW, TX)
   A = zeros(T, N)
   B = zeros(T, N)
   C = zeros(T, N)

   dotw(F, G) = sum((F .* G) .* W.W)

   # ---------------- degree 0 (n = 1)
   _J1 = ones(T, length(W.X))
   a = sqrt( dotw(_J1, _J1) )
   A[1] = 1/a 
   if N == 1
      return OrthPolyBasis1d3T(A, B, C)
   end

   # ---------------- degree 1 (n = 2)
   # a J2 = (t - b) J1
   J1 = A[1] * _J1
   b = dotw(J1, J1)
   _J2 = (W.X .- b) .* J1
   a = sqrt( dotw(_J2, _J2) )
   A[2] = 1 / a
   B[2] = -b / a
   if N == 2 
      return OrthPolyBasis1d3T(A, B, C)
   end 

   # ---------------- degree 2 and up, n = 3, 4, 5, ...
   # keep the last two for the 3-term recursion
   J2 = (A[2] * W.X .+ B[2]) .* J1
   Jprev = J2
   Jpprev = J1
   
   for n = 3:N
      # a Jn = (t - b) J_{n-1} - c J_{n-2}
      b = dotw(W.X .* Jprev, Jprev)
      c = dotw(W.X .* Jprev, Jpprev)
      _J = (W.X .- b) .* Jprev - c * Jpprev
      a = sqrt( dotw(_J, _J) )
      A[n] = 1/a
      B[n] = - b / a
      C[n] = - c / a
      Jprev, Jpprev = _J / a, Jprev
   end
   
   return OrthPolyBasis1d3T(A, B, C)
end




# --------------------- Jacobi Weights 
# this includes in particular Legendre, Gegenbauere, Chebyshev 

struct JacobiWeights{T}
   α::T
   β::T
   a::T 
   b::T 
end

chebyshev_weights(; a = -1.0, b = 1.0) = JacobiWeights(-0.5, -0.5, a, b)

legendre_weights(; a = -1.0, b = 1.0) = JacobiWeights(0.0, 0.0, a, b)

# function orthpolys(space::WeightedL2{<: JacobiWeights})

# end
