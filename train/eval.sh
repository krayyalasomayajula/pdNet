#!/bin/bash
## declare an array of subject names
declare -a Aarr=("2009"
"2010"
"2011"
"2012"
"2013"
"2014"
"2016"
)
## declare an array of subject links in youtube
for i in `seq 0 ${#Aarr[@]}`;
do
	th run.lua --dataset db --year ${Aarr[$i]} --datapath ../datasets/xDIBCO/128x256 --cachepath ../resources/DIBCO/128x256/cache/encoder --model models/encoder.lua --save ../resources/DIBCO/128x256/trained/encoder --imHeight 128 --imWidth 256 --labelHeight 16 --labelWidth 32 --batchSize 30 --maxepoch 20
	th run.lua --dataset db --year ${Aarr[$i]} --datapath ../datasets/xDIBCO/128x256 --cachepath ../resources/DIBCO/128x256/cache/decoder --model models/decoder.lua --save ../resources/DIBCO/128x256/trained/decoder --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 30 --CNNModel ../resources/DIBCO/128x256/trained/encoder --maxepoch 20
	th run.lua --dataset db --year ${Aarr[$i]} --datapath ../datasets/xDIBCO/128x256 --cachepath ../resources/DIBCO/128x256/cache/decoder --model models/pdNet.lua --save ../resources/DIBCO/128x256/trained/pdNet --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 20 --CNNModel ../resources/DIBCO/128x256/trained/decoder --maxepoch 20
	cd ../visualize
	th demo_PDdibco.lua -i ../datasets/xDIBCO/128x256/valid -d ../resources/DIBCO/128x256/trained/ -m pdNet --year ${Aarr[$i]} --dataset db -o ../datasets/xDIBCO/xOutput_pdnet_128x256 --devID 2
	cd -
done
cd ../utils
python dibcoImgJoin.py -i ../datasets/xDIBCO/xOutput_pdnet_128x256 -o ../datasets/xDIBCO/jOutput_pdnet_128x256 -c ../../ENet/datasets/DIBCO/pngs/img

