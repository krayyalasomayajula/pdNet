----------------------------------------------------------------------
-- This script implements a test procedure, to report accuracy
-- on the test data. It is a modified version of E-net test module
-- https://github.com/e-lab/ENet-training
--
-- Written by  : Kalyan Ram Ayyalasomayajula
-- Dated       : September, 2017
----------------------------------------------------------------------
require 'nn';
require 'nngraph';
require 'torch'
package.path = "common/?/init.lua;" .. package.path -- this path should be w.r.t the startup lua script 
require('icgnn');
require('icgcunn');
require 'cudnn'
require 'cunn'

torch.setdefaulttensortype('torch.FloatTensor')

----------------------------------------------------------------------
print '==> define parameters'

local histClasses = opt.datahistClasses
local classes = opt.dataClasses

----------------------------------------------------------
--  Auxillary functions
----------------------------------------------------------
  
--[[
  Builf the TV-primal-dual network.
  Inputs:
--]]

function bregmanPDNet(opt)
  -- parameters
    local max_iter        = opt.maxiter
    local lambda          = 0.01
    --local egde_w          = 1--0.01--5
    local egde_w          = 0.7252*4
    
    local L               = 1.1*math.sqrt(8)
    local gamma           = 1
    local tau             = gamma/L
    local sigma           = 0.01--(1/L)/gamma
    local mu
    
    local alpha           = 0.25
    
    local INCLUDE_UNKNOWN_CLASS = opt.includeUnknown -- 1 with unknown 0 without unknown
    local DIBCO_TEMPLATE
    
    local COST_WGT_TRAIN   
    local EDGE_WGT_TRAIN   
    local LAMBDA_WGT_TRAIN 
    local SIGMA_WGT_TRAIN  
    local INIT_USUM_COST   
    
    if(opt.costWgtTrain == 1) then COST_WGT_TRAIN    = true else COST_WGT_TRAIN    = false end
    if(opt.edgeWgtTrain == 1) then EDGE_WGT_TRAIN    = true else EDGE_WGT_TRAIN    = false end
    if(opt.lmbdWgtTrain == 1) then LAMBDA_WGT_TRAIN  = true else LAMBDA_WGT_TRAIN  = false end
    if(opt.sigmWgtTrain == 1) then SIGMA_WGT_TRAIN   = true else SIGMA_WGT_TRAIN   = false end
    if(opt.initUcost    == 1) then INIT_USUM_COST    = true else INIT_USUM_COST    = false end
    
    local MAX_AMPLITUDE = 1e30
    local MIN_AMPLITUDE = 1e-8
    
    --print(opt)
    --print('INCLUDE_UNKNOWN_CLASS: ',INCLUDE_UNKNOWN_CLASS)
    --print('DIBCO_TEMPLATE: ', DIBCO_TEMPLATE)
    local lambda_modules  = {}
    local sigma_modules   = {}
    local alpha_modules   = {}
    
    local param_modules   = {}
    
    local dbgNodeNames    = {}
    local nodeName
    
    local input       = nn.Identity()():annotate{name = 'IN_tbl', description = 'AGTV input'}  
    
    local classes
    if(INCLUDE_UNKNOWN_CLASS == 1) then
        classes     = #opt.dataClasses
    else
        classes     = #opt.dataClasses-1
    end
    
    local in_image, in_cost, cost, edges

    -- ===== Dibco setting
    
    nodeName = 'IMG_IN'
    table.insert(dbgNodeNames, nodeName)
    in_image    = nn.SelectTable(1)(input):annotate{name = nodeName, description = 'input image for edges'}
    
    -- ===== Include unknown Class setting
    nodeName = 'UNARY'
    table.insert(dbgNodeNames, nodeName)
    local in_cost1    = nn.SelectTable(2)(input):annotate{name = nodeName, description = 'unary input'}
    
    nodeName = 'NRW_UNARY'
    table.insert(dbgNodeNames, nodeName)
    if(INCLUDE_UNKNOWN_CLASS == 1) then
      in_cost    = nn.Narrow(2, 1, -1)(in_cost1):annotate{name = nodeName, description = 'Include Unknown Class'}
    else
      in_cost    = nn.Narrow(2, 1, 2)(in_cost1):annotate{name = nodeName, description = 'Remove Unknown Class'}
    end
    
    nodeName = 'COST'
    table.insert(dbgNodeNames, nodeName)
    cost        = nn.MulConstant(-1, false)( in_cost ):annotate{name = nodeName, description = '-ve Cost '}
    
    nodeName = 'EDG_W'
    table.insert(dbgNodeNames, nodeName)
    if(EDGE_WGT_TRAIN == true) then
      local edge  = icgnn.IcgExpMul()
      edge.weight:fill(math.log(egde_w/4))
      param_modules['edge_mul'] = edge
      
      edges       = nn.Exp()( 
                             edge( 
                                  nn.Power(2)(icgnn.IcgL2Norm()( icgnn.IcgNabla()( in_image ) )) 
                                  ) 
                             ):annotate{name = nodeName, description = 'Edge weights'}
    else
      edges       = nn.Exp()( 
                              nn.MulConstant(egde_w/4,false)( 
                                      nn.Power(2)(icgnn.IcgL2Norm()( icgnn.IcgNabla()( in_image ) )) 
                              ) 
                            ):annotate{name = nodeName, description = 'Edge weights'}
    end 
        
    -- Debug 01
    
    -- accomodating that IcgNabla produces dx dy in contiguous layers
    -- along dim_1
    nodeName = 'W2'
    table.insert(dbgNodeNames, nodeName)
    local w2      = nn.SelectTable(1)(nn.SplitTable(1,4)(nn.Replicate(2*classes, 2, 3)( edges )))
    :annotate{name = nodeName, description = 'W2'}
    
    local u_sum, u
    -- iterations
    if(INIT_USUM_COST == true) then
        u_sum = nn.Identity()(in_cost)
        u     = nn.AddConstant(1/classes)(nn.MulConstant(0, false)( cost ) )
    else
        u_sum = nn.AddConstant(1/classes)(nn.MulConstant(0, false)( cost ) )
        u     = nn.Identity()(u_sum)
    end
    
    local p     = nn.MulConstant(0,false)( icgnn.IcgNabla()(cost) )
    
    -- Debug 02
    
    
    local dxy, exp_dual, l_u, l_p, a_u
    local wp, der_wp, div
    local u_nmr, u_dmr, u_
    
    local u_nmr_raw, u_nmr_neg, u_nmr_pos
    local u_dmr_unique
    
    local p11, p12, p_nmr, p_dmr
    local p_dmr_raw, p_dmr_neg, p_dmr_pos 
    --local iter = 1
    for iter = 1, max_iter do 
        -- Update primal variables
        if iter < max_iter then
          u_      = nn.Identity()(u)
        end
        
        nodeName = 'WP['..iter..']'
        table.insert(dbgNodeNames, nodeName)
        wp      = nn.CMulTable()({w2,p}):annotate{name = nodeName, description = 'WP'}
        
        nodeName = 'DER_WP['..iter..']'
        table.insert(dbgNodeNames, nodeName)
        der_wp  = icgnn.IcgNabla()(wp):annotate{name = nodeName, description = 'DER_WP'}

        --Debug 03.1
        
        nodeName = 'DIV['..iter..']'
        table.insert(dbgNodeNames, nodeName)
        
        if(INCLUDE_UNKNOWN_CLASS == 1) then
          div    = nn.JoinTable(1,3)({
                    nn.CAddTable()({nn.Narrow(2, 1, 1)(der_wp),nn.Narrow(2, 4, 1)(der_wp)}),
                    nn.CAddTable()({nn.Narrow(2, 5, 1)(der_wp),nn.Narrow(2, 8, 1)(der_wp)}),
                    nn.CAddTable()({nn.Narrow(2, 9, 1)(der_wp),nn.Narrow(2, 12, 1)(der_wp)})
                    }):annotate{name = nodeName, description = 'DIV'}
        else
          div    = nn.JoinTable(1,3)({
                    nn.CAddTable()({nn.Narrow(2, 1, 1)(der_wp),nn.Narrow(2, 4, 1)(der_wp)}),
                    nn.CAddTable()({nn.Narrow(2, 5, 1)(der_wp),nn.Narrow(2, 8, 1)(der_wp)})
                    }):annotate{name = nodeName, description = 'DIV'}
        end
        -- Debug 04          
        if(LAMBDA_WGT_TRAIN == true) then
            local lambda_u = icgnn.IcgExpMul()
            table.insert(lambda_modules, lambda_u)
            param_modules['lambda_u_'..iter] = lambda_u
            
            l_u = lambda_u(cost)
            
            nodeName = 'U_NMR['..iter..']'
            table.insert(dbgNodeNames, nodeName)
            u_nmr_raw   = nn.CMulTable()(
                                      {nn.Exp()(
                                                nn.MulConstant(2*tau, true)(
                                                                            icgnn.IcgAddition({1,-1})({div,l_u})
                                                                            )
                                                ), 
                                      u}
                                      ):annotate{name = nodeName, description = 'u numerator'}
        else
            nodeName = 'U_NMR['..iter..']'
            table.insert(dbgNodeNames, nodeName)
            u_nmr_raw   = nn.CMulTable()(
                                      {nn.Exp()(
                                                nn.MulConstant(2*tau, true)(
                                                                            icgnn.IcgAddition({1,-lambda})({div,cost})
                                                                            )
                                                ), 
                                      u}
                                      ):annotate{name = nodeName, description = 'u numerator'}
        end
        
        if(1==1) then
          u_nmr_neg = nn.Clamp(-MAX_AMPLITUDE,-MIN_AMPLITUDE)(u_nmr_raw)
          u_nmr_pos = nn.Clamp(MIN_AMPLITUDE,MAX_AMPLITUDE)(u_nmr_raw)
          u_nmr     = nn.AddConstant(MIN_AMPLITUDE, true)(nn.CAddTable()({u_nmr_neg,u_nmr_pos}))
        else
          u_nmr     = nn.AddConstant(0, true)(u_nmr_raw)
        end
        
        if(INCLUDE_UNKNOWN_CLASS == 1) then
          u_dmr_unique   = nn.CAddTable()( {  nn.Narrow(2, 1, 1)(u_nmr),
                                              nn.Narrow(2, 2, 1)(u_nmr),
                                              nn.Narrow(2, 3, 1)(u_nmr)
                                   } )        
        else
          u_dmr_unique   = nn.CAddTable()( {  nn.Narrow(2, 1, 1)(u_nmr),
                                              nn.Narrow(2, 2, 1)(u_nmr)
                                     } )
        end
        nodeName = 'U_DMR['..iter..']'
        table.insert(dbgNodeNames, nodeName)
        u_dmr          = nn.SelectTable(1)(nn.SplitTable(1,4)(nn.Replicate(classes, 2, 3)(u_dmr_unique))
                                            ):annotate{name = nodeName, description = 'u denominator'}
        
        -- Debug 05
        nodeName = 'U['..iter..']'
        table.insert(dbgNodeNames, nodeName)
        u        = nn.Clamp(MIN_AMPLITUDE,1)(nn.CDivTable()( { u_nmr,u_dmr })):annotate{name = nodeName, description = 'u value'}
        
        -- Debug 06
        local alpha_u = icgnn.IcgExpMul()
        table.insert(alpha_modules, alpha)
        param_modules['alpha_u_'..iter] = alpha_u
        
        a_u = alpha_u(u)
        nodeName = 'U_SUM['..iter..']'
        table.insert(dbgNodeNames, nodeName)
        --u_sum  = icgnn.IcgAddition({iter/(iter+1),1/(iter+1)})({u_sum,u}):annotate{name = nodeName, description = 'u_sum'}
        --u_sum  = icgnn.IcgAddition({1,0})({u_sum,u}):annotate{name = nodeName, description = 'u_sum'}
        u_sum  = icgnn.IcgAddition({1,1})({u_sum,a_u}):annotate{name = nodeName, description = 'u_sum'}
        --compute_primal_energy(u_out, cost, lambda)]
        
        -- Update dual variables
        if iter < max_iter then
          nodeName = 'U_['..iter..']'
          table.insert(dbgNodeNames, nodeName)
          u_    = icgnn.IcgAddition({2,-1})({u,u_}):annotate{name = nodeName, description = 'u value'}
          
          nodeName = 'DXY['..iter..']'
          table.insert(dbgNodeNames, nodeName)
          dxy = nn.CMulTable()({ w2, icgnn.IcgNabla()(u_) }):annotate{name = nodeName, description = 'u_ value'}
          
          if(SIGMA_WGT_TRAIN == true) then
              local lambda_p = icgnn.IcgExpMul()
              table.insert(sigma_modules, lambda_p)
              param_modules['lambda_p_'..iter] = lambda_p
              
              l_p = lambda_p(dxy)
              nodeName = 'EXP_D['..iter..']'
              table.insert(dbgNodeNames, nodeName)
              exp_dual = nn.Exp()( nn.MulConstant(2, true)( l_p ) ):annotate{name = nodeName, description = 'exponential of dual'}
          else
              nodeName = 'EXP_D['..iter..']'
              table.insert(dbgNodeNames, nodeName)
              exp_dual = nn.Exp()( nn.MulConstant(2*sigma, true)( dxy ) ):annotate{name = nodeName, description = 'exponential of dual'}
          end
          
          
          p12 = nn.AddConstant(-1, false)( p ) --Sign flipped 
          p11 = nn.CAddTable()( {nn.CMulTable()({ exp_dual, p}), exp_dual} )
          
          nodeName = 'P_N['..iter..']'
          table.insert(dbgNodeNames, nodeName)
          p_nmr = nn.CAddTable()( {p11,p12} ):annotate{name = nodeName, description = 'p numerator'}
          
          p_dmr_raw = icgnn.IcgAddition({1,-1})({p11,p12})
          p_dmr_neg = nn.Clamp(-MAX_AMPLITUDE,-MIN_AMPLITUDE)(p_dmr_raw)
          p_dmr_pos = nn.Clamp(MIN_AMPLITUDE,MAX_AMPLITUDE)(p_dmr_raw)
          
          nodeName = 'P_D['..iter..']'
          table.insert(dbgNodeNames, nodeName)
          p_dmr    = nn.AddConstant(MIN_AMPLITUDE, true)(nn.CAddTable()({p_dmr_neg,p_dmr_pos})):annotate{name = nodeName, description = 'p denominator'}
          
          nodeName = 'P['..iter..']'
          table.insert(dbgNodeNames, nodeName)
          p  = nn.Clamp(-1,1)(nn.CDivTable()( {p_nmr, p_dmr} )):annotate{name = nodeName, description = 'p value'}
          if(SIGMA_WGT_TRAIN == false) then
              sigma = sigma / 10
              sigma  = math.max(1e-3,sigma)
          end
        end
        -- Debug 03
        
    end -- for iter = 1, max_iter--]]
  -- set lambda parameter on modules
  if(LAMBDA_WGT_TRAIN == true) then
    for _, lambda_module in ipairs(lambda_modules) do
      lambda_module.weight:fill(math.log(lambda))
    end
  end
  
  if(SIGMA_WGT_TRAIN == true) then
    for _, sigma_module in ipairs(sigma_modules) do
      sigma_module.weight:fill(math.log(sigma))
    end
  end
  
  local net = nn.gModule({input}, {u_sum})
  
  
  return net, dbgNodeNames
  
end

----------------------------------------------------------------------
print '==> construct model'

-- Load the CNN from disk for the unary pixel cost
nn.DataParallelTable.deserializeNGPUs = 1
local unary     = torch.load(opt.CNNModel .. '/' .. 'model-' .. opt.year .. '.net')
if torch.typename(unary) == 'nn.DataParallelTable' then unary = unary:get(1) end
local pairwise, dgb_tbl  = bregmanPDNet(opt)
pairwise:cuda()
  
local unaryBlock = nn.ConcatTable()
unaryBlock:add(nn.Identity()) -- Image Node
unaryBlock:add(unary)

local pdNetwork = nn.Sequential()
pdNetwork:add(unaryBlock)
pdNetwork:add(pairwise)
  
pdNetwork:cuda()
pdNetwork.name = 'binarization_PDNetwork'

--print(opt)
local INCLUDE_UNKNOWN_CLASS = opt.includeUnknown -- 1 with unknown 0 without unknown

local histClasses = opt.datahistClasses
local classes     = #opt.dataClasses

--print(classes )

-- Loss: NLL
print('defining loss function:')
local normHist = histClasses / histClasses:sum()

print('=============================')
print('normHist: ', normHist)
print('=============================')

local classWeights, loss

if(INCLUDE_UNKNOWN_CLASS == 1) then
    classWeights = torch.Tensor(classes):fill(1)
    for i = 1, classes do
       -- Ignore unlabeled and egoVehicle
       if i == classes then
          classWeights[i] = 0
        end
       if histClasses[i] < 1 then
          print("Class " .. tostring(i) .. " not found")
          classWeights[i] = 0
       else
          --classWeights[i] = 1 / (torch.log(1.02 + normHist[i]))
          classWeights[i] = torch.pow(torch.log(1.02 + normHist[i]), -0.5)
       end
    end
    loss = cudnn.SpatialCrossEntropyCriterion(classWeights)
else
    classWeights = torch.Tensor(classes-1):fill(1) -- Remove unknown class
    for i = 1, classes-1 do
       -- Ignore unlabeled and egoVehicle
       if histClasses[i] < 1 then
          print("Class " .. tostring(i) .. " not found")
          classWeights[i] = 0
       else
          --classWeights[i] = 1 / (torch.log(1.02 + normHist[i]))
          classWeights[i] = torch.pow(torch.log(1.02 + normHist[i]), -0.5)
       end
    end
    loss = cudnn.SpatialCrossEntropyCriterion(classWeights)
end

print('=============================')
print('classWeights: ', classWeights)
print('=============================')

loss:cuda()

----------------------------------------------------------------------
--print '==> here is the model:'
--print(pdNetwork)


-- return package:
return {
   model = pdNetwork,
   loss = loss,
}
