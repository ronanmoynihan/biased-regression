require 'nn'
require 'BiasedMSECriterion'
autograd = require 'autograd'
local create_model = require 'create_model'


--------------------------------------------------------------
-- SETTINGS
local opt = { nonlinearity_type = 'requ' }

-- function that numerically checks gradient of the loss:
-- f is the scalar-valued function
-- g returns the true gradient (assumes input to f is a 1d tensor)
-- returns difference, true gradient, and estimated gradient
local function checkgrad(f, g, x, eps)

  -- compute true gradient
  local grad = g(x)
  
  -- compute numeric approximations to gradient
  local eps = eps or 1e-4

  -- grad_est is the numerical gradient.
  local grad_est = torch.DoubleTensor(grad:size())

  for i = 1, grad:size(1) do
    -- TODO: do something with x[i] and evaluate f twice, and put your estimate of df/dx_i into grad_est[i]
    x[i] = x[i] + eps
    local C1 = f(x)

    x[i] = x[i] - 2 * eps

    local C2 = f(x)
    x[i] = x[i] + eps
    --...something(s) here
    grad_est[i] = (C1 - C2) / (2 * eps)
  end

  -- computes (symmetric) relative error of gradient
  local diff = torch.norm(grad - grad_est) / torch.norm(grad + grad_est)

  if diff > eps then
       print "Gradient check failed."
    else
      print "Gradient check Passed."
    end 

  return diff, grad, grad_est
end

function fakedata(n,d)
    local data = {}

    data.inputs = torch.Tensor({{1,2,1,2}, 
                                {6,6,7,7}})
    data.targets = torch.Tensor({{-2}, {7}})

    -- data.inputs = torch.randn(n, d)                     -- random standard normal distribution for inputs
    -- data.targets = torch.rand(n):mul(30):add(1):floor()  -- random integers from {1,2,3}
    return data
end

---------------------------------------------------------
-- generate fake data, then do the gradient check
--
dim = 4
torch.manualSeed(1)
local data = fakedata(30,dim)
print(data.inputs)
print(data.targets)

-- Use Biased MSE criterion if specified.

-- The lower the bias weight the more the model will under / over predict.
-- Define function for criterion
-- MSE
local mse = function(input, target)

  local delta = input - target 
  local loss = delta

   -- model has over predicted.
  loss.value[torch.lt(loss.value,0)] = 0
  -- loss.value:apply(function(x)
                 
  --                 if x < 0 then
  --                   return 0
                  
                  
  --                 end  
  --                 -- if x < 0 then
  --                 --   return 0.5 * x
  --                 -- else
  --                 --   return x - 0.5
  --                 -- end 
  --                end)

   loss = torch.cmul(loss,loss)
  
  return torch.sum(loss)



   -- buffer = input-target
   -- print(buffer.value)
   -- print('target')
   -- print(target)
   -- print('\ninput')
   -- print(input.value)
   -- -- x:apply(function() i = i + 1; return i end)
   -- return buffer.value:apply(function(x)
   --                x = x * 2
   --                return x
   --              end):sum()

   
   -- return torch.sum( torch.cmul(buffer, buffer) ) / (input:dim() == 2 and input:size(1)*input:size(2) or input:size(1))
end


local autoModel = create_model(dim)
local autoMseCriterion = autograd.nn.AutoCriterion('AutoMSE')(mse)
 
local parameters, gradParameters = autoModel:getParameters()
print(autoModel)

-- returns loss(params)
local f = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  -- return criterion:forward(model:forward(data.inputs), data.targets)
  return autoMseCriterion:forward(autoModel:forward(data.inputs), data.targets)
end
-- returns dloss(params)/dparams
local g = function(x)
  if x ~= parameters then
    parameters:copy(x)
  end
  gradParameters:zero()


  local outputs = autoModel:forward(data.inputs)
  -- criterion:forward(outputs, data.targets)
  -- model:backward(data.inputs, criterion:backward(outputs, data.targets))

  local mseOut = autoMseCriterion:forward(outputs, data.targets)
  local gradOutput = autoMseCriterion:backward(outputs, data.targets)
  -- model:backward(data.inputs, gradOutput)
  local gradInput = autoModel:backward(data.inputs, gradOutput)


  return gradParameters
end

local diff = checkgrad(f, g, parameters)
print(diff)
