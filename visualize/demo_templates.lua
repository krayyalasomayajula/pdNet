#!/usr/bin/env qlua
--------------------------------------------------------------------------------
-- View demos based on the trained network output
--
-- e-Lab
-- Written by: Abhishek Chaurasia
-- Dated: 24th March, 2016
--------------------------------------------------------------------------------

-- Torch packages
require 'image'
require 'imgraph'
require 'cunn'
require 'cudnn'
require 'nn'
require 'nngraph'
package.path = "../train_pdNet/common/?/init.lua;" .. package.path -- this path should be w.r.t the startup lua script 
require('icgnn');
require('icgcunn');

-- Local repo files
local opts = require 'opts'
local colorMap = assert(require('colorMap'))

-- Get the input arguments parsed and stored in opt
local opt = opts.parse(arg)

print(opt)

torch.setdefaulttensortype('torch.FloatTensor')
if opt.dev:lower() == 'cuda' then
   cutorch.setDevice(opt.devID)
   print("GPU # " .. cutorch.getDevice() .. " selected")
end

----------------------------------------
-- Network
local network = {}
network.path = opt.dmodel .. opt.model .. '/model-' .. opt.net .. '.net'
assert(paths.filep(network.path), 'Model not present at ' .. network.path)
print("Loading model from: " .. network.path)

network.model = torch.load(network.path)

-- Convert all the modules in nn from cudnn
--if #network.model:findModules('cudnn.SpatialConvolution') > 0 then
--   if network.model.__typename == 'nn.DataParallelTable' then
--      network.model = network.model:get(1)
--   end
--end

-- Change model type based on device being used for demonstration
if opt.dev:lower() == 'cpu' then
   cudnn.convert(network.model, nn)
   network.model:float()
else
   network.model:cuda()
end

-- Set the module mode 'train = false'
network.model:cuda()
network.model:evaluate()
--network.model:clearState()
print('network.model')
print(network.model)
-- Get mean and std of the dataset used while training
local stat_file = opt.dmodel .. opt.model .. '/' .. 'stat.t7'
if paths.filep(stat_file) then
   network.stat = torch.load(stat_file)
elseif paths.filep(stat_file .. 'ascii') then
   network.stat = torch.load(stat_file .. '.ascii', 'ascii')
else
   print('No stat file found in directory: ' .. opt.dmodel .. opt.model)
   network.stat = {}
   network.stat.mean = torch.Tensor{0, 0, 0}
   network.stat.std = torch.Tensor{1, 1, 1}
end

-- classes and color based on neural net model used:
local classes

---------------------------------------
-- Split string at a given separator
---------------------------------------

function split(inputstr, sep)
  if sep == nil then
    sep = "%s"
  end
  local t={} ; i=1
  for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
    t[i] = str
    i = i + 1
  end
  return t
end

---------------------------------------
-- Read a CSV file
---------------------------------------

--change target based on categories csv file:
function readCatCSV(filepath)
   print(filepath)
   local file = io.open(filepath, 'r')
   local classes = {}
   local targets = {}
   file:read()    -- throw away first line
   local fline = file:read()
   while fline ~= nil do
      local col1, col2 = fline:match("([^,]+),([^,]+)")
      table.insert(classes, col1)
      table.insert(targets, ('1' == col2))
      fline = file:read()
   end
   return classes, targets
end

-- Load categories from the list of categories generated during training
local newcatdir = opt.dmodel .. opt.model .. '/categories.txt'
if paths.filep(newcatdir) then
   print('Loading categories file from: ' .. newcatdir)
   network.classes, network.targets = readCatCSV(newcatdir)
end

if #network.classes == 0 then
   error('Categories file contains no categories')
end

print('Network has this list of categories, targets:')
for i=1,#network.classes do
   if opt.allcat then network.targets[i] = true end
   print(i..'\t'..network.classes[i]..'\t'..tostring(network.targets[i]))
end

classes = network.classes

print("classes",classes)
-- ++++++++++++++++++++++++++++++++++
colorMap:init(opt, classes)
local colors = colorMap.getColors()

-- generating the <colormap> out of the <colors> table
local colormap = imgraph.colormap(colors)
-- ++++++++++++++++++++++++++++++++++
colormap = torch.Tensor({{1.0, 0.0, 0.0},
            {0.0, 1.0, 0.0},
            {0.0, 0.0, 1.0}
           })
print("colormap",colormap)

cnt =1

local savedir = opt.output
print("savedir",savedir)
os.execute('mkdir -p ' .. savedir)

local matio = require 'matio'

for fil in paths.files(opt.input) do
  --print('========> Processing:',paths.concat(opt.input,fil))
  if(paths.filep(paths.concat(opt.input,fil))) then
    local tabl_cst_ed = matio.load(paths.concat(opt.input,fil))
    --[[local scaledCost = tabl_cst_ed.cost
    local scaledEdges = tabl_cst_ed.edges--]]
    
    local scaledCost = torch.reshape(tabl_cst_ed.cost,1,tabl_cst_ed.cost:size(1),tabl_cst_ed.cost:size(2),tabl_cst_ed.cost:size(3))
    local scaledEdges = torch.reshape(tabl_cst_ed.edges,1,tabl_cst_ed.edges:size(1),tabl_cst_ed.edges:size(2),tabl_cst_ed.edges:size(3))
    --]]
    
    if opt.dev == 'cuda' then
       print('CUDA enabled')
       scaledCostGPU = torch.CudaTensor(scaledCost:size())
       scaledCostGPU:copy(scaledCost)
       scaledCost = scaledCostGPU
       
       scaledEdgesGPU = torch.CudaTensor(scaledEdges:size())
       scaledEdgesGPU:copy(scaledEdges)
       scaledEdges = scaledEdgesGPU
           
    end
   
    --print('cost:size()',scaledCost:size())
    --print('edges:size()',scaledEdges:size())
    
    -- compute network on frame:
    distributions = network.model:forward({scaledCost,scaledEdges})
    
    _, winners = distributions[{1}]:max(1)
    --print(winners:size())
    print('winners:min()',winners:min())
    print('winners:max()',winners:max())
    
    local colored = torch.zeros(3,winners:size(2),winners:size(3))
    --print((winners:eq(1)):size())
    --print(colored[{{1},{},{}}])
    colored[{{1},{},{}}] = winners:eq(1):float()
    colored[{{2},{},{}}] = winners:eq(2):float()
    colored[{{3},{},{}}] = winners:eq(3):float()
    image.save(paths.concat(opt.output,fil..'.png'), colored)
        
  end
end
