local BiasedMSECriterion, parent = torch.class('nn.BiasedMSECriterion', 'nn.Criterion')

function BiasedMSECriterion:__init(opt)
  parent.__init(self)

  if opt.under and opt.over then
    error('Criterion only supports one option (over or under).')
  end  

  -- Defaults.
  self.over_weight = 1
  self.under_weight = 1
  self.sizeAverage = true

  if opt.underPredict then
    self.under_weight = opt.biasWeight   
  elseif opt.overPredict then
    self.over_weight = opt.biasWeight
  else
    error('Over or Under option has not been set.')
  end  
end

function BiasedMSECriterion:updateOutput(input, target)
  local delta = target - input

   -- model has over predicted.
   delta[torch.lt(delta,0)] = delta[torch.lt(delta,0)]:cmul(delta[torch.le(delta,0)]):mul(self.under_weight)
  
   -- model has under predicted.
   delta[torch.ge(delta,0)] = delta[torch.ge(delta,0)]:cmul(delta[torch.ge(delta,0)]):mul(self.over_weight)

   local loss = delta:sum()

  if self.sizeAverage then
    loss = loss / input:nElement()
  end

  return loss
end

function BiasedMSECriterion:updateGradInput(input, target)
  local norm = self.sizeAverage and 1.0 / input:nElement() or 1.0
  self.gradInput:resizeAs(input):copy(input):add(-1, target)

  -- model has over predicted.
  self.gradInput[torch.lt(self.gradInput,0)] = self.gradInput[torch.lt(self.gradInput,0)]:mul(norm):mul(self.under_weight)

  --- model has under predicted.
  self.gradInput[torch.ge(self.gradInput,0)] = self.gradInput[torch.ge(self.gradInput,0)]:mul(norm):mul(self.over_weight)

  return self.gradInput 
end