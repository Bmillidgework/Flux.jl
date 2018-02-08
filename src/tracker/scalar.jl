struct TrackedNumber{T<:Number} <: Number
  tracker::Tracked{T}
end

TrackedNumber(x::Number) = TrackedNumber(Tracked(Call(nothing), x, zero(x)))

tracker(x::TrackedNumber) = x.tracker

track(f::Call, x::Number) = TrackedNumber(Tracked(f, x, zero(x)))

back!(x::TrackedNumber) = back!(x, 1)

function Base.show(io::IO, x::TrackedNumber)
  show(io, data(x))
  print(io, " (tracked)")
end

Base.convert(::Type{TrackedNumber{T}}, x::TrackedNumber{T}) where T = x

Base.convert(::Type{TrackedNumber{T}}, x::TrackedNumber) where T =
  TrackedNumber(Tracked(x.tracker.f, convert(T, x.tracker.data)))

Base.convert(::Type{TrackedNumber{T}}, x::Number) where T = TrackedNumber(convert(T, x))

Base.isless(x::TrackedNumber, y::Number) = isless(data(x), y)
Base.isless(x::Number, y::TrackedNumber) = isless(x, data(y))
Base.isless(x::TrackedNumber, y::TrackedNumber) = isless(data(x), data(y))

Base.:(==)(x::TrackedNumber, y::Number) = data(x) == y
Base.:(==)(x::Number, y::TrackedNumber) = x == data(y)
Base.:(==)(x::TrackedNumber, y::TrackedNumber) = data(x) == data(y)

for f in :[isinf, isnan, isfinite].args
  @eval Base.$f(x::TrackedNumber) = Base.$f(data(x))
end

Base.Printf.fix_dec(x::TrackedNumber, n::Int) = Base.Printf.fix_dec(data(x), n)

Base.promote_rule(::Type{TrackedNumber{S}},::Type{T}) where {S,T} =
  TrackedNumber{promote_type(S,T)}

using DiffRules, SpecialFunctions, NaNMath

for (M, f, arity) in DiffRules.diffrules()
  arity == 1 || continue
  @eval begin
    $M.$f(a::TrackedNumber) = track($M.$f, a)
    back(::typeof($M.$f), Δ::Number, a::TrackedNumber) =
      back(a, Δ * $(DiffRules.diffrule(M, f, :(data(a)))))
  end
end

for (M, f, arity) in DiffRules.diffrules()
  arity == 2 || continue
  da, db = DiffRules.diffrule(M, f, :(data(a)), :(data(b)))
  @eval begin
    $M.$f(a::TrackedNumber, b::TrackedNumber)  = track($M.$f, a, b)
    $M.$f(a::TrackedNumber, b::Number) = track($M.$f, a, b)
    $M.$f(a::Number, b::TrackedNumber) = track($M.$f, a, b)
    function back(::typeof($M.$f), Δ::Number, a::Number, b::Number)
      @back(a, Δ * $da)
      @back(b, Δ * $db)
    end
  end
end

# Tuples

struct TrackedTuple{T<:Tuple}
  tracker::Tracked{T}
end

tracker(xs::TrackedTuple) = xs.tracker

accum!(x::Tuple, Δ::Tuple) = accum!.(x, Δ)
init_grad(x::Tuple) = init_grad.(x)

track(f::Call, xs::Tuple) = TrackedTuple(Tracked(f, xs))

function Base.show(io::IO, xs::TrackedTuple)
  show(io, data(xs))
  print(io, " (tracked)")
end

Base.length(x::TrackedTuple) = length(data(x))

Base.getindex(xs::TrackedTuple, i::Integer) = track(getindex, xs, i)

back(::typeof(getindex), Δ, t, i) =
  back(t, ntuple(j -> i == j ? Δ : 0, length(t)))
