require 'nn';
require 'nngraph';

local nBatch   = 2
local nCh      = 3
local nHt      = 4
local nWd      = 5

xx = torch.Tensor(nBatch,nCh, nHt, nWd)

--========== First batch 
xx[{1,1,{},{}}] = 1*torch.ones(nHt, nWd) -- bluCh
xx[{1,2,{},{}}] = 2*torch.ones(nHt, nWd) -- grnCh
xx[{1,3,{},{}}] = 3*torch.ones(nHt, nWd) -- redCh

--========== Second batch 
xx[{2,1,{},{}}] = 0.5*torch.ones(nHt, nWd) -- bluCh
xx[{2,2,{},{}}] = 0.2*torch.ones(nHt, nWd) -- grnCh
xx[{2,3,{},{}}] = 0.3*torch.ones(nHt, nWd) -- redCh

local augNetInitBlock = nn.ConcatTable()
local allCh  = nn.Identity()

local sepCh  = nn.ConcatTable()
local bluCh  = nn.Sequential():add(nn.Narrow(2, 1, 1)):add(nn.MulConstant(0.114, false))
local grnCh  = nn.Sequential():add(nn.Narrow(2, 2, 1)):add(nn.MulConstant(0.587, false))
local redCh  = nn.Sequential():add(nn.Narrow(2, 3, 1)):add(nn.MulConstant(0.299, false))
sepCh:add(bluCh)
sepCh:add(grnCh)
sepCh:add(redCh)

local gryCh  = nn.Sequential()
gryCh:add(sepCh):add(nn.CAddTable())

augNetInitBlock:add(allCh)
augNetInitBlock:add(gryCh)

local augNet = nn.Sequential()
augNet:add(augNetInitBlock):add(nn.JoinTable(2))

yy = augNet:forward(xx)

print(yy:size())
print(yy)