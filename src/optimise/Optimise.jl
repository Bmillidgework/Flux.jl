module Optimise

export train!, step!,
	SGD, Descent, ADAM, Momentum, Nesterov, RMSProp,
	ADAGrad, AdaMax, ADADelta, AMSGrad, NADAM, ADAMW,
	InvDecay, ExpDecay, WeightDecay, stop, Optimiser

include("optimisers.jl")
include("update.jl")
include("train.jl")

end
