init_grad(x) = zero(x)
zero_grad!(x) = zero(x)
zero_grad!(x::AbstractArray) = (x .= 0)

function _walk(queue, seen, c::Call)
  foreach(c.args) do x
    x === nothing && return
    id = objectid(x)
    if id ∉ seen
      push!(seen, id)
      pushfirst!(queue, x)
    end
    return
  end
end

function walk(f, x::Tracked; once = true)
  queue = Tracked[x]
  seen = Set{UInt64}()
  while !isempty(queue)
    x = pop!(queue)
    f(x)
    _walk(queue, seen, x.f)
    once && !x.isleaf && (x.f = Call(missing, ()))
  end
end

accum!(x, Δ) = x .+ Δ
accum!(x::AbstractArray, Δ) = (x .+= Δ)

function _back(x::Tracked, Δ)
  if isdefined(x, :grad)
    x.grad = accum!(x.grad, Δ)
  else
    x.grad = Δ
  end
  return
end

_back(::Nothing, Δ) = return

function _back(c::Call, Δ)
  Δs = c.func(Δ)
  (Δs isa Tuple && length(Δs) >= length(c.args)) ||
    error("Gradient is not a tuple of length $(length(c.args))")
  foreach((x, d) -> _back(x, d), c.args, data.(Δs))
end

_back(::Call{Nothing}, Δ) = nothing
_back(::Call{Missing}, Δ) = error("`back!` was already used")

function back(x::Tracked, Δ, once)
  _back(x, Δ)
  walk(x, once = once) do x
    _back(x.f, x.grad)
  end
end

# Interface methods

function back!(x, Δ; once = true)
  istracked(x) || return
  back(tracker(x), Δ, once)
  return
end

function gradient_(f, xs...)
  xs = param.(xs)
  l = f(xs...)
  losscheck(l)
  back!(l)
  nobacksies("Use `gradient(...; nest = true)` for nested derivatives",
             grad.(xs))
end

# Out-of-place gradients

struct Params
  order::Vector{Any}
  params::IdSet{Any}
  Params() = new([], IdSet())
end

@forward Params.order Base.iterate, Base.length

function Base.push!(ps::Params, x)
  if !(x in ps.params)
    push!(ps.order, x)
    push!(ps.params, x)
  end
  return ps
end

Base.push!(ps::Params, x...) = (foreach(x -> push!(ps, x), x); ps)

Params(xs) = push!(Params(), xs...)

function Base.show(io::IO, ps::Params)
  print(io, "Params([")
  join(io, ps.order, ", ")
  print(io, "])")
end

struct Grads
  grads::IdDict{Any,Any}
end

Base.show(io::IO, ps::Grads) = println(io, "Grads(...)")

Grads() = Grads(IdDict())

@forward Grads.grads Base.setindex!, Base.haskey, Base.length, Base.iterate

Grads(ps::Params) = Grads(IdDict(tracker(p) => init_grad(data(p)) for p in ps))

Base.getindex(g::Grads, x::Tracked) = g.grads[x]

function Base.getindex(g::Grads, x)
  istracked(x) || error("Object not tracked: $x")
  g[tracker(x)]
end

accum!(g::Grads, x, Δ) = g[x] = haskey(g, x) ? g[x] .+ Δ : Δ

function _back(g::Grads, c::Call, Δ)
  Δs = c.func(Δ)
  (Δs isa Tuple && length(Δs) >= length(c.args)) ||
    error("Gradient is not a tuple of length $(length(c.args))")
  foreach((x, Δ) -> _back(g, x, Δ), c.args, Δs)
end

_back(g::Grads, ::Call{Nothing}, Δ) = nothing

function _back(g::Grads, x::Tracked, Δ)
  x.isleaf && (accum!(g, x, Δ); return)
  accum!(g, x, Δ)
  return
end

_back(g::Grads, ::Nothing, Δ) = return

function back(g::Grads, x::Tracked, Δ)
  _back(g, x, Δ)
  walk(x, once = false) do x
    _back(g, x.f, g[x])
  end
end

function forward(f, ps::Params)
  y = f()
  y, function (Δ)
    g = Grads(ps)
    if istracked(y)
      back(g, tracker(y), Δ)
    end
    return g
  end
end

function forward(f, args...)
  args = param.(args)
  y, back = forward(() -> f(args...), Params(args))
  y, Δ -> getindex.(Ref(back(Δ)), args)
end

function losscheck(x)
  x isa Real || error("Function output is not scalar")
  isinf(x) && error("Loss is infinite")
  isnan(x) && error("Loss is NaN")
end

function gradient_nested(f, args...)
  y, back = forward(f, args...)
  losscheck(y)
  return back(1)
end

gradient(f, xs...; nest = false) =
  nest ? gradient_nested(f, xs...) : gradient_(f, xs...)
