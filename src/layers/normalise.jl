istraining() = false

@adjoint istraining() = true, _ -> nothing

"""
    Dropout(p)

A Dropout layer. For each input, either sets that input to `0` (with probability
`p`) or scales it by `1/(1-p)`. This is used as a regularisation, i.e. it
reduces overfitting during training.

Does nothing to the input once in [`testmode!`](@ref).
"""
mutable struct Dropout{F}
  p::F
  function Dropout(p)
    @assert 0 ≤ p ≤ 1
    new{typeof(p)}(p)
  end
end

_dropout_kernel(y::T, p, q) where {T} = y > p ? T(1 / q) : T(0)

function (a::Dropout)(x)
  istraining() || return x
  y = similar(x)
  rand!(y)
  y .= _dropout_kernel.(y, a.p, 1 - a.p)
  return x .* y
end

"""
    AlphaDropout(p)
A dropout layer. It is used in Self-Normalizing Neural Networks.
(https://papers.nips.cc/paper/6698-self-normalizing-neural-networks.pdf)
The AlphaDropout layer ensures that mean and variance of activations remains the same as before.
"""
mutable struct AlphaDropout{F}
  p::F
  function AlphaDropout(p)
    @assert 0 ≤ p ≤ 1
    new{typeof(p)}(p)
  end
end

function (a::AlphaDropout)(x)
  istraining() || return x
  λ = eltype(x)(1.0507009873554804934193349852946)
  α = eltype(x)(1.6732632423543772848170429916717)
  α1 = eltype(x)(-λ*α)
  noise = randn(eltype(x), size(x))
  x = @. x*(noise > (1 - a.p)) + α1 * (noise <= (1 - a.p))
  A = (a.p + a.p * (1 - a.p) * α1 ^ 2)^0.5
  B = -A * α1 * (1 - a.p)
  x = @. A * x + B
  return x
end

"""
    LayerNorm(h::Integer)

A [normalisation layer](https://arxiv.org/pdf/1607.06450.pdf) designed to be
used with recurrent hidden states of size `h`. Normalises the mean/stddev of
each input before applying a per-neuron gain/bias.
"""
struct LayerNorm{T}
  diag::Diagonal{T}
end

LayerNorm(h::Integer) =
  LayerNorm(Diagonal(h))

@treelike LayerNorm

(a::LayerNorm)(x) = a.diag(normalise(x))

function Base.show(io::IO, l::LayerNorm)
  print(io, "LayerNorm(", length(l.diag.α), ")")
end

"""
    BatchNorm(channels::Integer, σ = identity;
              initβ = zeros, initγ = ones,
              ϵ = 1e-8, momentum = .1)

Batch Normalization layer. The `channels` input should be the size of the
channel dimension in your data (see below).

Given an array with `N` dimensions, call the `N-1`th the channel dimension. (For
a batch of feature vectors this is just the data dimension, for `WHCN` images
it's the usual channel dimension.)

`BatchNorm` computes the mean and variance for each each `W×H×1×N` slice and
shifts them to have a new mean and variance (corresponding to the learnable,
per-channel `bias` and `scale` parameters).

See [Batch Normalization: Accelerating Deep Network Training by Reducing
Internal Covariate Shift](https://arxiv.org/pdf/1502.03167.pdf).

Example:
```julia
m = Chain(
  Dense(28^2, 64),
  BatchNorm(64, relu),
  Dense(64, 10),
  BatchNorm(10),
  softmax)
```
"""
mutable struct BatchNorm{F,V,W,N}
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::W  # moving mean
  σ²::W  # moving std
  ϵ::N
  momentum::N
end

BatchNorm(chs::Integer, λ = identity;
          initβ = (i) -> zeros(Float32, i), initγ = (i) -> ones(Float32, i), ϵ = 1f-5, momentum = 0.1f0) =
  BatchNorm(λ, initβ(chs), initγ(chs),
            zeros(chs), ones(chs), ϵ, momentum)

function (BN::BatchNorm)(x)
  size(x, ndims(x)-1) == length(BN.β) ||
    error("BatchNorm expected $(length(BN.β)) channels, got $(size(x, ndims(x)-1))")
  dims = length(size(x))
  channels = size(x, dims-1)
  affine_shape = ones(Int, dims)
  affine_shape[end-1] = channels
  m = prod(size(x)[1:end-2]) * size(x)[end]
  γ = reshape(BN.γ, affine_shape...)
  β = reshape(BN.β, affine_shape...)
  if !istraining()
    μ = reshape(BN.μ, affine_shape...)
    σ² = reshape(BN.σ², affine_shape...)
    ϵ = BN.ϵ
  else
    T = eltype(x)
    axes = [1:dims-2; dims] # axes to reduce along (all but channels axis)
    μ = mean(x, dims = axes)
    σ² = sum((x .- μ) .^ 2, dims = axes) ./ m
    ϵ = convert(T, BN.ϵ)
    # update moving mean/std
    mtm = convert(T, BN.momentum)
    BN.μ = (1 - mtm) .* BN.μ .+ mtm .* reshape(μ, :)
    BN.σ² = (1 - mtm) .* BN.σ² .+ (mtm * m / (m - 1)) .* reshape(σ², :)
  end

  let λ = BN.λ
    x̂ = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    λ.(γ .* x̂ .+ β)
  end
end

children(BN::BatchNorm) =
  (BN.λ, BN.β, BN.γ, BN.μ, BN.σ², BN.ϵ, BN.momentum)

mapchildren(f, BN::BatchNorm) =  # e.g. mapchildren(cu, BN)
  BatchNorm(BN.λ, f(BN.β), f(BN.γ), f(BN.μ), f(BN.σ²), BN.ϵ, BN.momentum)

function Base.show(io::IO, l::BatchNorm)
  print(io, "BatchNorm($(join(size(l.β), ", "))")
  (l.λ == identity) || print(io, ", λ = $(l.λ)")
  print(io, ")")
end


"""
    InstanceNorm(channels::Integer, σ = identity;
                 initβ = zeros, initγ = ones,
                 ϵ = 1e-8, momentum = .1)

Instance Normalization layer. The `channels` input should be the size of the
channel dimension in your data (see below).

Given an array with `N` dimensions, call the `N-1`th the channel dimension. (For
a batch of feature vectors this is just the data dimension, for `WHCN` images
it's the usual channel dimension.)

`InstanceNorm` computes the mean and variance for each each `W×H×1×1` slice and
shifts them to have a new mean and variance (corresponding to the learnable,
per-channel `bias` and `scale` parameters).

See [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022).

Example:
```julia
m = Chain(
  Dense(28^2, 64),
  InstanceNorm(64, relu),
  Dense(64, 10),
  InstanceNorm(10),
  softmax)
```
"""
expand_inst = (x, as) -> reshape(repeat(x, outer=[1, as[length(as)]]), as...)

mutable struct InstanceNorm{F,V,W,N}
  λ::F  # activation function
  β::V  # bias
  γ::V  # scale
  μ::W  # moving mean
  σ²::W  # moving std
  ϵ::N
  momentum::N
end

InstanceNorm(chs::Integer, λ = identity;
          initβ = (i) -> zeros(Float32, i), initγ = (i) -> ones(Float32, i), ϵ = 1f-5, momentum = 0.1f0) =
  InstanceNorm(λ, initβ(chs), initγ(chs),
            zeros(chs), ones(chs), ϵ, momentum)

function (in::InstanceNorm)(x)
  size(x, ndims(x)-1) == length(in.β) ||
    error("InstanceNorm expected $(length(in.β)) channels, got $(size(x, ndims(x)-1))")
  ndims(x) > 2 ||
    error("InstanceNorm requires at least 3 dimensions. With 2 dimensions an array of zeros would be returned")
  # these are repeated later on depending on the batch size
  dims = length(size(x))
  c = size(x, dims-1)
  bs = size(x, dims)
  affine_shape = ones(Int, dims)
  affine_shape[end-1] = c
  affine_shape[end] = bs
  m = prod(size(x)[1:end-2])
  γ, β = expand_inst(in.γ, affine_shape), expand_inst(in.β, affine_shape)

  if !istraining()
    μ = expand_inst(in.μ, affine_shape)
    σ² = expand_inst(in.σ², affine_shape)
    ϵ = in.ϵ
  else
    T = eltype(x)

    ϵ = convert(T, in.ϵ)
    axes = 1:dims-2 # axes to reduce along (all but channels and batch size axes)
    μ = mean(x, dims = axes)
    σ² = mean((x .- μ) .^ 2, dims = axes)

    # update moving mean/std
    mtm = convert(T, in.momentum)
    in.μ = dropdims(mean(repeat((1 - mtm) .* in.μ, outer=[1, bs]) .+ mtm .* reshape(μ, (c, bs)), dims = 2), dims=2)
    in.σ² = dropdims(mean((repeat((1 - mtm) .* in.σ², outer=[1, bs]) .+ (mtm * m / (m - 1)) .* reshape(σ², (c, bs))), dims = 2), dims=2)
  end

  let λ = in.λ
    x̂ = (x .- μ) ./ sqrt.(σ² .+ ϵ)
    λ.(γ .* x̂ .+ β)
  end
end

children(in::InstanceNorm) =
  (in.λ, in.β, in.γ, in.μ, in.σ², in.ϵ, in.momentum)

mapchildren(f, in::InstanceNorm) =  # e.g. mapchildren(cu, in)
  InstanceNorm(in.λ, f(in.β), f(in.γ), f(in.μ), f(in.σ²), in.ϵ, in.momentum)

function Base.show(io::IO, l::InstanceNorm)
  print(io, "InstanceNorm($(join(size(l.β), ", "))")
  (l.λ == identity) || print(io, ", λ = $(l.λ)")
  print(io, ")")
end
