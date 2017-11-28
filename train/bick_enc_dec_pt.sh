#!/bin/bash
th run.lua --dataset bk --year 2009 --datapath ../datasets/xBickleyDairy/128x256 --cachepath ../resources/xBickleyDairy/128x256/cache/encoder --model models/encoder.lua --save ../resources/xBickleyDairy/128x256/trained/encoder --imHeight 128 --imWidth 256 --labelHeight 16 --labelWidth 32 --batchSize 30 --maxepoch 20 --ptModel ../resources/xDibcoGray/128x256/trained/encoder/model-2009.net
th run.lua --dataset bk --year 2009 --datapath ../datasets/xBickleyDairy/128x256 --cachepath ../resources/xBickleyDairy/128x256/cache/decoder --model models/decoder.lua --save ../resources/xBickleyDairy/128x256/trained/decoder --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 30 --maxepoch 20 --CNNModel ../resources/xBickleyDairy/128x256/trained/encoder --ptModel ../resources/xDibcoGray/128x256/trained/decoder/model-2009.net
th run.lua --dataset bk --year 2009 --datapath ../datasets/xBickleyDairy/128x256 --cachepath ../resources/xBickleyDairy/128x256/cache/decoder --model models/pdNet.lua --save ../resources/xBickleyDairy/128x256/trained/pdNet --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 20 --CNNModel ../resources/xBickleyDairy/128x256/trained/decoder --maxepoch 20	
cd ../visualize
th demo_PDbickley.lua -i ../datasets/xBickleyDairy/128x256/test -d ../resources/xBickleyDairy/128x256/trained/ -m pdNet --year 2009 --dataset bk -o ../datasets/xBickleyDairy/xOutput_pdnet_128x256 --devID 2
cd ../utils
python dibcoImgJoin.py -i ../datasets/xBickleyDairy/xOutput_pdnet_128x256 -o ../datasets/xBickleyDairy/jOutput_pdnet_128x256 -c ../../imgDataSets/balinese/pngs/test/img

