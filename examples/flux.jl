using Flux

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
#            Logistic Regression from scratch             #
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––– #

W = param(zeros(10,784))
b = param(zeros(10))

pred(x) = softmax(W*x .+ b)

cost(x, y) = mean(sum(log.(pred(x)).*y, 1))

# See an example predction
pred(rand(784))

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
#                  Custom Layer: Dense                    #
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––– #

struct Dense
  σ
  W
  b
end

function Dense(in::Integer, out::Integer, σ = σ)
  W = param(randn(out, in))
  b = param(zeros(out))
  return Dense(σ, W, b)
end

# Note that Julia compiles:
#   * Specialised code for the forward pass (wrt activation function,
#     number types, ...)
#   * Including a single GPU/CPU kernel for the broadcast call
function (m::Dense)(x)
  σ, W, b = m.σ, m.W, m.b
  σ.(W*x .+ b)
end

d = Dense(10, 5, relu)
d(rand(10))

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
#                  RNN from scratch                       #
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––– #

in = 10
out = 5

Wi = param(randn(out, in))
Wh = param(randn(out, out))
b = param(zeros(out))

function rnn(h, x)
  h = tanh.(Wi*x .+ Wh*h .+ b)
  return h, h
end

h = rand(out)
xs = [rand(in) for i = 1:13] # Sequence of length 13
ys = []

for x in xs
  h, y = rnn(h, x)
  push!(ys, y)
end

# Output hidden state and sequence
h, ys

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––– #
#                  Recursive Net                          #
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––– #

N = 10

# Generate dummy data
tree() = rand() < 0.5 ? rand(N) : (tree(), tree())

# Model

shrink = Dense(2N, N)
combine(a, b) = shrink([a; b])

model(x) = x
model(x::Tuple) = combine(model(x[1]), model(x[2]))

# The model is able to compress an arbitrary tree into
# a single length N representation.
model(tree())
