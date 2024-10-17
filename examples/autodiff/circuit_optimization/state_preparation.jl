using ITensors
using OptimKit
using Random
using Zygote

nsites = 20 # Number of sites
nlayers = 3 # Layers of gates in the ansatz
gradtol = 1e-4 # Tolerance for stopping gradient descent

# A layer of the circuit we want to optimize
function layer(nsites, θ⃗)
  RY_layer = [("Ry", (n,), (θ=θ⃗[n],)) for n in 1:nsites]
  CX_layer = [("CX", (n, n + 1)) for n in 1:2:(nsites-1)]
  return [RY_layer; CX_layer]
end

# The variational circuit we want to optimize
function variational_circuit(nsites, nlayers, θ⃗)
  range = 1:nsites
  circuit = layer(nsites, θ⃗[range])
  for n in 1:(nlayers-1)
    circuit = [circuit; layer(nsites, θ⃗[range .+ n * nsites])]
  end
  return circuit
end

Random.seed!(1234)

θ⃗ᵗᵃʳᵍᵉᵗ = 2π * rand(nsites * nlayers)
𝒰ᵗᵃʳᵍᵉᵗ = variational_circuit(nsites, nlayers, θ⃗ᵗᵃʳᵍᵉᵗ)

s = siteinds("Qubit", nsites)
Uᵗᵃʳᵍᵉᵗ = ops(𝒰ᵗᵃʳᵍᵉᵗ, s)

ψ0 = MPS(s, "0")

# Create the random target state
ψᵗᵃʳᵍᵉᵗ = apply(Uᵗᵃʳᵍᵉᵗ, ψ0; cutoff=1e-8)

#
# The loss function, a function of the gate parameters
# and implicitly depending on the target state:
#
# loss(θ⃗) = -|⟨θ⃗ᵗᵃʳᵍᵉᵗ|U(θ⃗)|0⟩|² = -|⟨θ⃗ᵗᵃʳᵍᵉᵗ|θ⃗⟩|²
#
function loss(θ⃗)
  nsites = length(ψ0)
  s = siteinds(ψ0)
  𝒰θ⃗ = variational_circuit(nsites, nlayers, θ⃗)
  Uθ⃗ = ops(𝒰θ⃗, s)
  ψθ⃗ = apply(Uθ⃗, ψ0)
  return -abs(inner(ψᵗᵃʳᵍᵉᵗ, ψθ⃗))^2
end

θ⃗₀ = randn!(copy(θ⃗ᵗᵃʳᵍᵉᵗ))

@show loss(θ⃗₀), loss(θ⃗ᵗᵃʳᵍᵉᵗ)

loss_∇loss(x) = (loss(x), convert(Vector, loss'(x)))
algorithm = LBFGS(; gradtol=gradtol, verbosity=2)
θ⃗ₒₚₜ, lossₒₚₜ, ∇lossₒₚₜ, numfg, normgradhistory = optimize(loss_∇loss, θ⃗₀, algorithm)

@show loss(θ⃗ₒₚₜ), loss(θ⃗ᵗᵃʳᵍᵉᵗ)

nothing
