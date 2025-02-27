using Polynomials4ML, Test, ObjectPools, StaticArrays
using Polynomials4ML.Testing: println_slim, print_tf

@info(" ----- Testing the FlexArray Interface -------")

tests = [
   (chebyshev_basis(10), () -> rand()),
   (RTrigBasis(10), () -> rand()),
   (CTrigBasis(10), () -> rand()),
   (MonoBasis(10), ()-> rand()),
   (legendre_basis(10), () -> rand()),
   (CYlmBasis(5), () -> randn(SVector{3, Float64})),
   (RYlmBasis(5), () -> randn(SVector{3, Float64}))
]

for (basis, rnd) in tests   
   for ntest = 1:5 
      x = rnd()
      B0 = zeros(Polynomials4ML._valtype(basis, x), length(basis))
      evaluate!(B0, basis, x)
      B1 = evaluate(basis, x)
      B2 = evaluate!(FlexArray(), basis, x)
      print_tf(@test B0 == B1 == B2)
   end
end
println()
