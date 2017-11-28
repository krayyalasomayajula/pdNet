----------------------------------------------------------------------
-- Main script for training a model for semantic segmentation
--
-- Abhishek Chaurasia, Eugenio Culurciello
-- Sangpil Kim, Adam Paszke
----------------------------------------------------------------------

require 'pl'
require 'nn'

----------------------------------------------------------------------
-- Local repo files
local opts = require 'opts'

-- Get the input arguments parsed and stored in opt
opt = opts.parse(arg)

torch.setdefaulttensortype('torch.FloatTensor')

-- print('==> switching to CUDA')
require 'cudnn'
require 'cunn'
--print(cutorch.getDeviceProperties(opt.devid))
cutorch.setDevice(opt.devid)

--opt.save = '../resources/trained/model'
print('\n\27[32mModels will be saved in \27[0m\27[4m' .. opt.save .. '\27[0m')
os.execute('mkdir -p ' .. opt.save)
print('\n\27[32mCached data is available in \27[0m\27[4m' .. opt.cachepath .. '\27[0m')
os.execute('mkdir -p ' .. opt.cachepath)
if(opt.dataset == 'db') then
  os.execute('mkdir -p ' .. opt.cachepath)
end
  
print(opt)

----------------------------------------------------------------------
print '==> load modules'
local data, chunks, ft
if opt.dataset == 'db' then
   data = require 'data/loadDibco'
elseif opt.dataset == 'sn' then
   data = require 'data/loadSynthetic'
else
   error ("Dataset loader not found. (Available options are: cv/cs/su/db")
end

print 'saving opt as txt and t7'
local filename = paths.concat(opt.save,'opt.txt')
local file = io.open(filename, 'w')
for i,v in pairs(opt) do
    file:write(tostring(i)..' : '..tostring(v)..'\n')
end
file:close()
torch.save(path.join(opt.save,'opt.t7'),opt)

----------------------------------------------------------------------
local epoch = 1

t = paths.dofile(opt.model)

local train = require 'train'
local test  = require 'test'

print('\27[31m\27[4m\nTraining and testing started\27[0m')
print('[batchSize = ' ..  opt.batchSize .. ']')
while epoch < opt.maxepoch do
   print(string.format('\27[31m\27[4m\nEpoch # %d\27[0m', epoch))
   print('==> Training:')
   local trainConf, model, loss = train(data.trainData, opt.dataClasses, epoch)
   print('==> Testing:')
   test(data.valdData, opt.dataClasses, epoch, trainConf, model, loss )
   trainConf = nil
   collectgarbage()
   epoch = epoch + 1
end
