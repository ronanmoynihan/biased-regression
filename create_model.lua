require 'nn'

function create_model(input_dim,criterion)

  	------------------------------------------------------------------------------
   	-- MODEL
    ------------------------------------------------------------------------------

    local n_inputs = input_dim
    local numhid1 = 9
    local n_outputs = 1

    local model = nn.Sequential()           

  	model:add(nn.Linear(n_inputs, numhid1)) 
  	model:add(nn.Sigmoid())
  	model:add(nn.Linear(numhid1, n_outputs))

  	return model
end

return create_model