----------------------------------------------------------------------
-- Create model and calulate loss to optimize for decoder.
--
-- Adam Paszke,
-- May 2016.
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
print '==> define parameters'

local histClasses = opt.datahistClasses
local classes = opt.dataClasses

----------------------------------------------------------------------
print '==> construct model'

-- encoder CNN:

nn.DataParallelTable.deserializeNGPUs = 1
--model = torch.load(opt.CNNModel .. '/' .. 'model-' .. opt.year .. '_enc_pt.net')
model = torch.load(opt.CNNModel .. '/' .. 'model-' .. opt.year .. '.net')
if torch.typename(model) == 'nn.DataParallelTable' then model = model:get(1) end
model:remove(#model.modules) -- remove the classifier

-- SpatialMaxUnpooling requires nn modules...
model:apply(function(module)
   if module.modules then
      for i,submodule in ipairs(module.modules) do
         if torch.typename(submodule):match('cudnn.SpatialMaxPooling') then
            module.modules[i] = nn.SpatialMaxPooling(2, 2, 2, 2) -- TODO: make more flexible
         end
      end
   end
end)

model:cuda()
local wEn,_ = model:getParameters()

-- find pooling modules
local pooling_modules = {}
model:apply(function(module)
   if torch.typename(module):match('nn.SpatialMaxPooling') then
      table.insert(pooling_modules, module)
   end
end)
assert(#pooling_modules == 3, 'There should be 3 pooling modules')

-- kill gradient
-- local grad_killer = nn.Identity()
-- function grad_killer:updateGradInput(input, gradOutput)
--    return self.gradInput:resizeAs(gradOutput):zero()
-- end
-- model:add(grad_killer)

-- decoder:

print(pooling_modules)

function bottleneck(input, output, upsample, reverse_module)
   local internal = output / 4
   local input_stride = upsample and 2 or 1

   local module = nn.Sequential()
   local sum = nn.ConcatTable()
   local main = nn.Sequential()
   local other = nn.Sequential()
   sum:add(main):add(other)

   main:add(cudnn.SpatialConvolution(input, internal, 1, 1, 1, 1, 0, 0):noBias())
   main:add(nn.SpatialBatchNormalization(internal, 1e-3))
   main:add(cudnn.ReLU(true))
   if not upsample then
      main:add(cudnn.SpatialConvolution(internal, internal, 3, 3, 1, 1, 1, 1))
   else
      main:add(nn.SpatialFullConvolution(internal, internal, 3, 3, 2, 2, 1, 1, 1, 1))
   end
   main:add(nn.SpatialBatchNormalization(internal, 1e-3))
   main:add(cudnn.ReLU(true))
   main:add(cudnn.SpatialConvolution(internal, output, 1, 1, 1, 1, 0, 0):noBias())
   main:add(nn.SpatialBatchNormalization(output, 1e-3))

   other:add(nn.Identity())
   if input ~= output or upsample then
      other:add(cudnn.SpatialConvolution(input, output, 1, 1, 1, 1, 0, 0):noBias())
      other:add(nn.SpatialBatchNormalization(output, 1e-3))
      if upsample and reverse_module then
         other:add(nn.SpatialMaxUnpooling(reverse_module))
      end
   end

   if upsample and not reverse_module then
      main:remove(#main.modules) -- remove BN
      return main
   end
   return module:add(sum):add(nn.CAddTable()):add(cudnn.ReLU(true))
end

local nEncoderModeles = #model.modules

--model:add(bottleneck(128, 128))
model:add(bottleneck(128, 64, true, pooling_modules[3]))         -- 32x64
model:add(bottleneck(64, 64))
model:add(bottleneck(64, 64))
model:add(bottleneck(64, 16, true, pooling_modules[2]))          -- 64x128
model:add(bottleneck(16, 16))
model:add(nn.SpatialFullConvolution(16, #classes, 2, 2, 2, 2))

function pause()
   io.stdin:read'*l'
end

--local wDe,_ = (model:cuda()):getParameters()
model:cuda()
local wDe,_ = model:getParameters()
--print('weigths in encoder')
--print(#wEn)
--print('weigths in decoder')
--print(#wDe)
--pause()

if(paths.filep(opt.ptModel) == true) then -- pre-trained decoder CNN:
  print(string.format('\27[31m\27[4m\nLoading pre-trained decoder\27[0m'))
  --print(string.format('\27[33m\27[4m\n # %d encoder layers \27[0m', nEncoderModeles))
  nn.DataParallelTable.deserializeNGPUs = 1
  local modelDec = torch.load(opt.ptModel)
    
  if torch.typename(modelDec) == 'nn.DataParallelTable' then modelDec = modelDec:get(1) end
  
  --print(string.format('\27[33m\27[4m\n # %d decoder layers \27[0m', #modelDec.modules))
  --for i,submodule in ipairs(modelDec.modules) do
  --  if(i > nEncoderModeles) then
  --    model.modules[i] = modelDec.modules[i]
  --  end
  --end
  modelDec:cuda()
  local wDePt,_ = modelDec:getParameters()
  
  local nDecWgts = wDePt:size()[1]
  local nEnWgts  = wEn:size()[1]
  
  --print(nDecWgts)
  --print(nEnWgts)
  for i = 1, nDecWgts do
    if(i > nEnWgts) then
      wDe[i] = wDePt[i]
    end 
  end
end
--pause()

if cutorch.getDeviceCount() > 1 then
   local gpu_list = {}
   for i = 1,cutorch.getDeviceCount() do gpu_list[i] = i end
   model = nn.DataParallelTable(1):add(model:cuda(), gpu_list)
   print(opt.nGPU .. " GPUs being used")
end


-- Loss: NLL
print('defining loss function:')
local normHist = histClasses / histClasses:sum()
print('=============================')
print('normHist: ', normHist)
print('=============================')

local classWeights = torch.Tensor(#classes):fill(1)
for i = 1, #classes do
   -- Ignore unlabeled and egoVehicle
   if i == #classes then
      classWeights[i] = 0
   end
   if histClasses[i] < 1 then
      print("Class " .. tostring(i) .. " not found")
      classWeights[i] = 0
   else
      --classWeights[i] = 1 / (torch.log(1.2 + normHist[i]))
      classWeights[i] = torch.pow(torch.log(1.02 + normHist[i]), -1)
   end
end

loss = cudnn.SpatialCrossEntropyCriterion(classWeights)

model:cuda()
loss:cuda()

----------------------------------------------------------------------
--print '==> here is the model:'
--print(model)

--io.output('model.txt')
--io.write(print(model))

print('=============================')
print('classWeights: ', classWeights)
print('=============================')

-- return package:
return {
   model = model,
   loss = loss,
}

