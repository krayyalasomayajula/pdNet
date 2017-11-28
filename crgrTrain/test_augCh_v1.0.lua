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

local augIn  = nn.Identity()()
local bluCh  = nn.Narrow(2, 1, 1)(augIn)
local grnCh  = nn.Narrow(2, 2, 1)(augIn)
local redCh  = nn.Narrow(2, 3, 1)(augIn)
local gryCh  = nn.CAddTable()({
                        nn.CAddTable()({nn.MulConstant(0.114, false)( bluCh ),nn.MulConstant(0.587, false)( grnCh )}),
                        nn.MulConstant(0.299, false)( redCh )})
local augOut = nn.JoinTable(1,3)({bluCh,grnCh,redCh,gryCh})

local augNet = nn.gModule({augIn},{augOut})

yy = augNet:forward(xx)

print(yy:size())
print(yy)