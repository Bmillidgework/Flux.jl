using Zygote: Context, globals

const Param{T<:Number} = Union{AbstractArray{T},T}

struct Globals{T}
  gs::T
end

Globals(cx::Context) = Globals(globals(cx))

_apply(opt, x, x̄, state) = apply(opt, x, x̄, state)
_apply(opt, x, x̄, ::Nothing) = apply(opt, x, x̄)

# Immutable updates

function update(opt, x::Param, x̄::Param, state = nothing)
  Δ, state = _apply(opt, x, x̄, state)
  return x .- Δ, state
end

# Mutable updates

# Figure out if we can do in-place
inplace(x, y) = false
inplace(x, y::Nothing) = true
inplace(x::AbstractArray, x̄::AbstractArray) = true
inplace(x, x̄::NamedTuple) = all(inplace(getfield(x, f), getfield(x̄, f)) for f in fieldnames(typeof(x̄)))

function update!(opt, x::AbstractArray{<:Number}, x̄::AbstractArray, state = nothing)
  Δ, state = _apply(opt, x, x̄, state)
  x .-= Δ
  return state
end

function update!(opt, x, x̄::NamedTuple)
  for f in fieldnames(typeof(x̄))
    f̄ = getfield(x̄, f)
    f̄ === nothing || update!(opt, getfield(x, f), f̄)
  end
end

setglobal!(mod::Module, name::Symbol, x) =
  ccall(:jl_set_global, Cvoid, (Any, Any, Any), mod, name, x)

function update!(opt, ::Nothing, gs::Globals)
  for (id, x̄) in gs.gs
    x = getfield(id.mod, id.name)
    if inplace(x, x̄)
      update!(opt, x, x̄)
    else
      isconst(id.mod, id.name) && error("Can't update constant $id")
      x′, state = update(opt, x, x̄)
      setglobal!(id.mod, id.name, x′)
    end
  end
end
