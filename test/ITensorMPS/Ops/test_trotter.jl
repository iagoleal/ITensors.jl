using Test
using ITensors
using ITensors.Ops

function heisenberg(N)
  os = Sum{Op}()
  for j in 1:(N-1)
    os += "Sz", j, "Sz", j + 1
    os += 0.5, "S+", j, "S-", j + 1
    os += 0.5, "S-", j, "S+", j + 1
  end
  return os
end

@testset "Heisenberg Trotter" begin
  N = 4
  ℋ = heisenberg(N)
  s = siteinds("S=1/2", N)
  ψ₀ = MPS(s, n -> isodd(n) ? "↑" : "↓")
  t = 1.0
  for nsteps in [10, 100]
    for order in [1, 2] #, 4]
      𝒰 = exp(im * t * ℋ; alg=Trotter{order}(nsteps))
      U = Prod{ITensor}(𝒰, s)
      ∑H = Sum{ITensor}(ℋ, s)
      # XXX: Define this, filling out identities.
      # ITensor(ℋ, s)
      I = contract(MPO(s, "Id"))
      H = 0.0 * contract(MPO(s, "Id"))
      for h in ∑H
        H += apply(h, I)
      end
      Uʳᵉᶠψ₀ = replaceprime(exp(im * t * H) * prod(ψ₀), 1 => 0)
      atol = max(1e-6, 1 / nsteps^order)
      @test prod(U(ψ₀)) ≈ Uʳᵉᶠψ₀ atol = atol
    end
  end
end
