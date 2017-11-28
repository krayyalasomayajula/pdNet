#!/bin/bash
#th run.lua --dataset bn --year 2009 --datapath ../datasets/xBalineseGray/128x256 --cachepath ../resources/xBalineseGray/128x256/cache/encoder --model models/encoder.lua --save ../resources/xBalineseGray/128x256/trained/encoder --imHeight 128 --imWidth 256 --labelHeight 16 --labelWidth 32 --batchSize 30 --maxepoch 20 --ptModel ../resources/xDibcoGray/128x256/trained/encoder/model-2009.net
#th run.lua --dataset bn --year 2009 --datapath ../datasets/xBalineseGray/128x256 --cachepath ../resources/xBalineseGray/128x256/cache/decoder --model models/decoder.lua --save ../resources/xBalineseGray/128x256/trained/decoder --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 30 --maxepoch 20 --CNNModel ../resources/xBalineseGray/128x256/trained/encoder --ptModel ../resources/xDibcoGray/128x256/trained/decoder/model-2009.net
#th run.lua --dataset bn --year 2009 --datapath ../datasets/xBalineseGray/128x256 --cachepath ../resources/xBalineseGray/128x256/cache/decoder --model models/pdNet.lua --save ../resources/xBalineseGray/128x256/trained/pdNet --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 20 --CNNModel ../resources/xBalineseGray/128x256/trained/decoder --maxepoch 20	
#cd ../visualize
#th demo_PDdibco.lua -i ../datasets/xBalineseGray/128x256/test -d ../resources/xBalineseGray/128x256/trained/ -m pdNet --year 2009 --dataset bn -o ../datasets/xBalineseGray/xOutput_pdnet_128x256 --devID 2
#cd ../utils
#python dibcoImgJoin.py -i ../datasets/xBalineseGray/xOutput_pdnet_128x256 -o ../datasets/xBalineseGray/jOutput_pdnet_128x256 -c ../../imgDataSets/balinese/pngs/test/img
th run.lua --dataset bn --year 2009 --datapath ../datasets/xBalinese/128x256 --cachepath ../resources/xBalinese/128x256/cache/encoder --model models/encoder.lua --save ../resources/xBalinese/128x256/trained/encoder --imHeight 128 --imWidth 256 --labelHeight 16 --labelWidth 32 --batchSize 30 --maxepoch 20 --ptModel ../resources/xDibco/128x256/trained/encoder/model-2009.net
th run.lua --dataset bn --year 2009 --datapath ../datasets/xBalinese/128x256 --cachepath ../resources/xBalinese/128x256/cache/decoder --model models/decoder.lua --save ../resources/xBalinese/128x256/trained/decoder --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 30 --maxepoch 20 --CNNModel ../resources/xBalinese/128x256/trained/encoder --ptModel ../resources/xDibco/128x256/trained/decoder/model-2009.net
th run.lua --dataset bn --year 2009 --datapath ../datasets/xBalinese/128x256 --cachepath ../resources/xBalinese/128x256/cache/decoder --model models/pdNet.lua --save ../resources/xBalinese/128x256/trained/pdNet --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 20 --CNNModel ../resources/xBalinese/128x256/trained/decoder --maxepoch 20	
cd ../visualize
th demo_PDdibco.lua -i ../datasets/xBalinese/128x256/test -d ../resources/xBalinese/128x256/trained/ -m pdNet --year 2009 --dataset bn -o ../datasets/xBalinese/xOutput_pdnet_128x256 --devID 2
cd ../utils
python dibcoImgJoin.py -i ../datasets/xBalinese/xOutput_pdnet_128x256 -o ../datasets/xBalinese/jOutput_pdnet_128x256 -c ../../imgDataSets/balinese/pngs/test/img

