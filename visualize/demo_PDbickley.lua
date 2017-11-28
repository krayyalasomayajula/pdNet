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
package.path = "../train/common/?/init.lua;" .. package.path -- this path should be w.r.t the startup lua script
require('icgnn');
require('icgcunn');
--require '../train/models/pdNet'
require 'nn';
require 'nngraph';
require 'torch'
 

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
network.path = opt.dmodel .. opt.model .. '/model-' .. opt.year .. '.net'
assert(paths.filep(network.path), 'Model not present at ' .. network.path)
print("Loading model from: " .. network.path)

network.model = torch.load(network.path)
--print(network.model)
-- Convert all the modules in nn from cudnn
--[[if #network.model:findModules('cudnn.SpatialConvolution') > 0 then
   if network.model.__typename == 'nn.DataParallelTable' then
      network.model = network.model:get(1)
   end
end

-- Change model type based on device being used for demonstration
if opt.dev:lower() == 'cpu' then
   cudnn.convert(network.model, nn)
   network.model:float()
else
   network.model:cuda()
end
--]]

-- Set the module mode 'train = false'
network.model:evaluate()
network.model:clearState()

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
print('classes',classes)
-- ++++++++++++++++++++++++++++++++++
colorMap:init(opt, classes)
local colors = colorMap.getColors()

-- generating the <colormap> out of the <colors> table
local colormap = imgraph.colormap(colors)
-- ++++++++++++++++++++++++++++++++++

cnt =1
local savedir = opt.output
os.execute('mkdir -p ' .. savedir)

local matio = require 'matio'

for fil in paths.files(opt.input) do
  --print('========> Processing:',paths.concat(opt.input,fil))
  if(paths.filep(paths.concat(opt.input,fil))) then
    img = image.load(paths.concat(opt.input,fil),3,'float')
    
    if img:dim() == 3 then
         img = img:view(1, img:size(1), img:size(2), img:size(3))
    end
    local scaledImg = torch.Tensor(1, 3, opt.ratio * img:size(3), opt.ratio * img:size(4))

    if opt.ratio == 1 then
       scaledImg[1] = img[1]
    else
       scaledImg[1] = image.scale(img[1],
                                  opt.ratio * source.w,
                                  opt.ratio * source.h,
                                  'bilinear')
    end

    if opt.dev == 'cuda' then
       scaledImgGPU = scaleImgGPU or torch.CudaTensor(scaledImg:size())
       scaledImgGPU:copy(scaledImg)
       scaledImg = scaledImgGPU
    end
    
    -- compute network on frame:
    distributions = network.model:forward(scaledImg):squeeze()
    
    --print('distributions:size()',distributions:size())
    _, winners = distributions:max(1)
    if opt.dev == 'cuda' then
       cutorch.synchronize()
       winner = winners:squeeze():float()
    else
       winner = winners:squeeze()
    end
    colored, colormap = imgraph.colorize(winner, colormap)
    
    image.save(paths.concat(opt.output,fil), colored)
  end
end
