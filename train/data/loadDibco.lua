----------------------------------------------------------------------
-- Dibco data loader,
-- Kalyan Ram,
-- August 2017
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- to visualize the dataset

torch.setdefaulttensortype('torch.FloatTensor')
----------------------------------------------------------------------
-- Dibco dataset:

local MAX_VALD_SIZE = 3000
local trsize, trlsize
local vasize, valsize
local tesize, telsize

local classes     = {'FG', 'BG','Unlabled'}
local conClasses  = {'FG', 'BG'} -- 2 classes

local nClasses    = #classes
opt.channels = 3
--------------------------------------------------------------------------------

-- Ignoring unnecessary classes
print '==> remapping classes'
local classMap = {[0]   =  {1}, -- Text/ink/Foreground
                  [255] =  {2}, -- Paper/Parchment/Background
                  }

-- From here #class will give number of classes even after shortening the list
-- nClasses should be used to get number of classes in original list

-- saving training histogram of classes
local histClasses = torch.Tensor(#classes):zero()

print('==> number of classes: ' .. #classes)
print('classes are:')
print(classes)
--------------------------------------------------------------------------------
function pause()
  io.stdin:read'*l'
end

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

function getNumberOfFiles(filepath,type)
  local nFiles = 0
  if(type == 'train' or type == 'valid') then  
    for file in paths.iterfiles(filepath) do
      local year = split(file, '_')
      if(year[1] ~= tostring(opt.year)) then nFiles = nFiles+1 end
    end
  else
    for file in paths.iterfiles(filepath) do
      local year = split(file, '_')
      if(year[1] == tostring(opt.year)) then nFiles = nFiles+1 end
    end
  end
  if(type == 'valid') then nFiles = math.min(nFiles,MAX_VALD_SIZE) end
  return nFiles
end
--------------------------------------------------------------------------------
print '==> loading dibco dataset'
local trainData, valdData, testData
local loadedFromCache = false
local dibcoCachePath = paths.concat(opt.cachepath, opt.year .. '_data.t7')

local INCLUDE_TEST_DATA = false
local dataCache

if opt.cachepath ~= "none" and paths.filep(dibcoCachePath) then
   dataCache  = torch.load(dibcoCachePath)
   trainData  = dataCache.trainData
   valdData   = dataCache.valdData
   
   if(INCLUDE_TEST_DATA == true) then
    testData         = dataCache.testData
   end
   
   histClasses      = dataCache.histClasses
   loadedFromCache  = true
   dataCache        = nil
   collectgarbage()
else
   local function check_year_fileExtensions(filename,type)
      local year = split(filename, '_')
      local ext = string.lower(path.extension(filename))

      -- compare with list of image extensions
      if(type == 'train' or type == 'valid') then
        if(year[1] ~= tostring(opt.year)) then
          --print('Type: '.. type .. ' using file: ' .. filename)
          local img_extensions = {'.jpeg', '.jpg', '.png', '.ppm', '.pgm'}
          for i = 1, #img_extensions do
             if ext == img_extensions[i] then
                return true
             end
          end
        end
        return false
      else
        if(year[1] == tostring(opt.year)) then
          --print('Type: '.. type .. ' using file: ' .. filename)
          local img_extensions = {'.jpeg', '.jpg', '.png', '.ppm', '.pgm'}
          for i = 1, #img_extensions do
             if ext == img_extensions[i] then
                return true
             end
          end
        end
        return false
      end
   end

   -- initialize data structures:
   print('==> loading training files');

   local dpathRoot = opt.datapath .. '/train/'

   assert(paths.dirp(dpathRoot), 'No training folder found at: ' .. opt.datapath)
   
   trsize  = getNumberOfFiles(opt.datapath .. '/train/', 'train')
   trlsize = getNumberOfFiles(opt.datapath .. '/trainannot/', 'train')
   assert(trsize==trlsize, 'Training samples size mismatch.')
   
   --print('trsize', trsize)
   --print('trlsize', trlsize)
   
   trainData = {
      data = torch.FloatTensor(trsize, opt.channels, opt.imHeight, opt.imWidth),
      labels = torch.FloatTensor(trsize, opt.labelHeight, opt.labelWidth),
      preverror = 1e10, -- a really huge number
      size = function() return trsize end
   }

   --load training images and labels:
   local c = 1
   for file in paths.iterfiles(dpathRoot) do
       -- process each image
       if check_year_fileExtensions(file, 'train') and c <= trsize then
          local imgPath = path.join(dpathRoot, file)
    
          --load training images:
          local dataTemp = image.load(imgPath)
          trainData.data[c] = image.scale(dataTemp,opt.imWidth, opt.imHeight)
    
          -- Load training labels:
          -- Load labels with same filename as input image.
          imgPath = string.gsub(imgPath, "train", "trainannot")

          -- label image data are resized to be [1,nClasses] in [0 255] scale:
          local labelIn = image.load(imgPath, 1, 'byte'):squeeze():float()
          
          if (opt.labelHeight == labelIn:size(1)) and (opt.labelWidth == labelIn:size(2)) then
             labelFile = labelIn
          else
             labelFile = image.scale(labelIn, opt.labelWidth, opt.labelHeight, 'simple'):float()
          end
          
          labelFile:apply(function(x) return classMap[x][1] end)
          
          --print(labelFile)
    
          -- Syntax: histc(data, bins, min, max)
          histClasses = histClasses + torch.histc(labelFile, #classes, 1, #classes)
    
          -- convert to int and write to data structure:
          trainData.labels[c] = labelFile
    
          c = c + 1
          if c % 20 == 0 then
             xlua.progress(c, trsize)
          end
          collectgarbage()
       end
   end
   --end
   print('')

   print('==> loading validation files');
   dpathRoot = opt.datapath .. '/valid/'
   assert(paths.dirp(dpathRoot), 'No validation folder found at: ' .. opt.datapath)
   
   vasize  = getNumberOfFiles(opt.datapath .. '/valid/', 'valid')
   valsize = getNumberOfFiles(opt.datapath .. '/validannot/', 'valid')
   assert(vasize == valsize, 'Validation samples size mismatch.')
   
   valdData = {
      data = torch.FloatTensor(vasize, opt.channels, opt.imHeight, opt.imWidth),
      labels = torch.FloatTensor(vasize, opt.labelHeight, opt.labelWidth),
      preverror = 1e10, -- a really huge number
      size = function() return vasize end
   }
   
   -- load test images and labels:
   local c = 1
   for file in paths.iterfiles(dpathRoot) do
       -- process each image
       if check_year_fileExtensions(file, 'valid') and c <= vasize then
          local imgPath = path.join(dpathRoot, file)
  
          --load training images:
          local dataTemp = image.load(imgPath)
          valdData.data[c] = image.scale(dataTemp, opt.imWidth, opt.imHeight)
  
          -- Load validation labels:
          -- Load labels with same filename as input image.
          imgPath = string.gsub(imgPath, "valid", "validannot")

          -- load test labels:
          -- label image data are resized to be [1,nClasses] in in [0 255] scale:
          local labelIn = image.load(imgPath, 1, 'byte'):squeeze():float()
          if (opt.labelHeight == labelIn:size(1)) and (opt.labelWidth == labelIn:size(2)) then
             labelFile = labelIn
          else
             labelFile = image.scale(labelIn, opt.labelWidth, opt.labelHeight, 'simple'):float()
          end
          
          labelFile:apply(function(x) return classMap[x][1] end)
  
          -- convert to int and write to data structure:
          valdData.labels[c] = labelFile
  
          c = c + 1
          if c % 20 == 0 then
             xlua.progress(c, vasize)
          end
          collectgarbage()
       end
   end
   
   if(INCLUDE_TEST_DATA == true) then
     -- Test data is the relevent year that is skipped in validation data
     print('==> loading testing files');
     dpathRoot = opt.datapath .. '/valid/'
     assert(paths.dirp(dpathRoot), 'No test folder found at: ' .. opt.datapath)
     
     tesize  = getNumberOfFiles(opt.datapath .. '/valid/', 'test')
     telsize = getNumberOfFiles(opt.datapath .. '/validannot/', 'test')
     assert(tesize == telsize, 'Validation samples size mismatch.')
     
     testData = {
        data = torch.FloatTensor(tesize, opt.channels, opt.imHeight, opt.imWidth),
        labels = torch.FloatTensor(tesize, opt.labelHeight, opt.labelWidth),
        preverror = 1e10, -- a really huge number
        size = function() return tesize end
     }
     
     -- load test images and labels:
     local c = 1
     for file in paths.iterfiles(dpathRoot) do
         -- process each image
         if check_year_fileExtensions(file,'test') and c <= tesize then
            local imgPath = path.join(dpathRoot, file)
    
            --load training images:
            local dataTemp = image.load(imgPath)
            testData.data[c] = image.scale(dataTemp, opt.imWidth, opt.imHeight)
    
            -- Load validation labels:
            -- Load labels with same filename as input image.
            imgPath = string.gsub(imgPath, "valid", "validannot")
  
            -- load test labels:
            -- label image data are resized to be [1,nClasses] in in [0 255] scale:
            local labelIn = image.load(imgPath, 1, 'byte'):squeeze():float()
            if (opt.labelHeight == labelIn:size(1)) and (opt.labelWidth == labelIn:size(2)) then
               labelFile = labelIn
            else
               labelFile = image.scale(labelIn, opt.labelWidth, opt.labelHeight, 'simple'):float()
            end
            
            labelFile:apply(function(x) return classMap[x][1] end)
    
            -- convert to int and write to data structure:
            testData.labels[c] = labelFile
    
            c = c + 1
            if c % 20 == 0 then
               xlua.progress(c, tesize)
            end
            collectgarbage()
         end
     end
   end
  if(INCLUDE_TEST_DATA == true) then
    if opt.cachepath ~= "none" and not loadedFromCache then
       print('==> saving data to cache: ' .. dibcoCachePath)
       dataCache = {
          trainData = trainData,
          valdData = valdData,
          testData = testData,
          histClasses = histClasses
       }
     end
  else
    if opt.cachepath ~= "none" and not loadedFromCache then
       print('==> saving data to cache: ' .. dibcoCachePath)
       dataCache = {
          trainData = trainData,
          valdData = valdData,
          histClasses = histClasses
       }
    end   
  end
  torch.save(dibcoCachePath, dataCache)
  dataCache = nil
  collectgarbage()
end

----------------------------------------------------------------------
print '==> verify statistics'

-- It's always good practice to verify that data is properly
-- normalized.

local trainMean, trainStd
local valdMean,  valdStd
local testMean,  testStd

for i = 1, opt.channels do
   trainMean  = trainData.data[{ {},i }]:mean()
   trainStd   = trainData.data[{ {},i }]:std()

   valdMean   = valdData.data[{ {},i }]:mean()
   valdStd    = valdData.data[{ {},i }]:std()

   print('training data, channel-'.. i ..', mean: ' .. trainMean)
   print('training data, channel-'.. i ..', standard deviation: ' .. trainStd)

   print('validation data, channel-'.. i ..', mean: ' .. valdMean)
   print('validation data, channel-'.. i ..', standard deviation: ' .. valdStd)
   
   if(INCLUDE_TEST_DATA == true) then   
     testMean   = testData.data[{ {},i }]:mean()
     testStd    = testData.data[{ {},i }]:std()
        
     print('testing data, channel-'.. i ..', mean: ' .. testMean)
     print('testing data, channel-'.. i ..', standard deviation: ' .. testStd)
   end
end

----------------------------------------------------------------------

local classes_td = {[1] = 'classes,targets\n'}
for _,cat in pairs(classes) do
   table.insert(classes_td, cat .. ',1\n')
end

local file = io.open(paths.concat(opt.save, 'categories.txt'), 'w')
file:write(table.concat(classes_td))
file:close()

-- Exports
opt.dataClasses = classes
opt.dataconClasses  = conClasses
opt.datahistClasses = histClasses

if(INCLUDE_TEST_DATA == true) then
  return {
     trainData = trainData,
     valdData = valdData,
     testData = testData,
     mean = trainMean,
     std = trainStd
  }
else
  return {
     trainData = trainData,
     valdData = valdData,
     mean = trainMean,
     std = trainStd
  }
end
