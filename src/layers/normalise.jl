"""
    testmode!(m, val=true)

Put layers like [`Dropout`](@ref) and [`BatchNorm`](@ref) into testing mode
(or back to training mode with `false`).
"""
function testmode!(m, val::Bool=true)
  prefor(x -> _testmode!(x, val), m)
  return m
end

_testmode!(m, test) = nothing

"""
    Dropout(p)

A Dropout layer. For each input, either sets that input to `0` (with probability
`p`) or scales it by `1/(1-p)`. This is used as a regularisation, i.e. it
reduces overfitting during training.

Does nothing to the input once in [`testmode!`](@ref).
"""
mutable struct Dropout{F}
  p::F
  active::Bool
end

function Dropout(p)
  @assert 0 ≤ p ≤ 1
  Dropout{typeof(p)}(p, true)
end

_dropout_kernel(y::T, p, q) where {T} = y > p ? T(1 / q) : T(0)

function (a::Dropout)(x)
  a.active || return x
  y = similar(x)
  rand!(y)
  y .= _dropout_kernel.(y, a.p, 1 - a.p)
  return x .* y
end

_testmode!(a::Dropout, test) = (a.active = !test)

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

treelike(LayerNorm)

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

y = m(rand(28^2, 10))
```

To use the layer at test time set [`testmode!(m, true)`](@ref).
"""
mutable struct BatchNorm
  λ  # activation function
  β  # bias
  γ  # scale
  μ  # moving mean
  σ²  # moving var
  ϵ
  momentum
  active::Bool
end

# NOTE: Keeping the ϵ smaller than 1e-5 is not supported by CUDNN
function BatchNorm(chs::Integer, λ = identity;
          initβ = x->zeros(Float32,x),
          initγ = x->ones(Float32,x),
          ϵ = 1f-5,
          momentum = 0.1f0)
  BatchNorm(λ, param(initβ(chs)), param(initγ(chs)),
            zeros(Float32, chs), ones(Float32, chs), ϵ, momentum, true)
end

function (BN::BatchNorm)(x)
  size(x, ndims(x)-1) == length(BN.β) ||
    error("BatchNorm expected $(length(BN.β)) channels, got $(size(x, ndims(x)-1))")
  γ, β = BN.γ, BN.β
  dims = ndims(x)
  affine_shape = ones(Int, dims)
  affine_shape[end-1] = size(x, dims-1)
  T = eltype(x)

  if !BN.active
    μ = reshape(BN.μ, affine_shape...)
    σ² = reshape(BN.σ², affine_shape...)
  else

    axes = [1:dims-2; dims] # axes to reduce along (all but channels axis)
    m = prod(size(x, axes...))
    μ = mean(x, axes)
    σ² = sum((x.-μ).^2, axes) ./ m

    # update moving mean/std
    mtm = convert(T, BN.momentum)

    BN.μ = ((1 - mtm) .* BN.μ .+ mtm .* squeeze(data(μ), (axes...))) |> data
    BN.σ² = ((1 - mtm) .* BN.σ² .+ mtm .* squeeze(data(σ²), (axes...))*m/(m-1)) |> data
  end

  ϵ = convert(T, BN.ϵ)
  BN.λ.(reshape(γ, affine_shape...) .* ((x .- μ) ./ sqrt.(σ² .+ ϵ)) .+ reshape(β, affine_shape...))
end

treelike(BatchNorm)

_testmode!(BN::BatchNorm, test) = (BN.active = !test)

function Base.show(io::IO, l::BatchNorm)
  print(io, "BatchNorm($(join(size(l.β), ", "))")
  (l.λ == identity) || print(io, ", λ = $(l.λ)")
  print(io, ")")
end
