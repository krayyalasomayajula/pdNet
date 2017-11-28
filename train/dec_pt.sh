#!/bin/bash
## declare an array of subject names
#declare -a Aarr=("2009"
#"2010"
#"2011"
#"2012"
#"2013"
#"2014"
#"2016"
#)
declare -a Aarr=("2010"
"2011"
"2012"
"2013"
"2014"
"2016"
)
## declare an array of subject links in youtube
for i in `seq 0 ${#Aarr[@]}`;
do
	#th run.lua --dataset db --year ${Aarr[$i]} --datapath ../datasets/xDibco/128x256 --cachepath ../resources/Dibco/128x256/cache/encoder --model models/encoder.lua --save ../resources/Dibco/128x256/trained/encoder --imHeight 128 --imWidth 256 --labelHeight 16 --labelWidth 32 --batchSize 30 --maxepoch 20 --ptModel ../resources/Dibco/128x256/trained/encoder/model-0.net
	th run.lua --dataset db --year ${Aarr[$i]} --datapath ../datasets/xDibco/128x256 --cachepath ../resources/Dibco/128x256/cache/decoder --model models/decoder.lua --save ../resources/Dibco/128x256/trained/decoder --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 30 --maxepoch 20 --CNNModel ../resources/Dibco/128x256/trained/encoder --ptModel ../resources/Dibco/128x256/trained/decoder/model-0.net
	th run.lua --dataset db --year ${Aarr[$i]} --datapath ../datasets/xDibco/128x256 --cachepath ../resources/Dibco/128x256/cache/decoder --model models/pdNet.lua --save ../resources/Dibco/128x256/trained/pdNet --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 20 --CNNModel ../resources/Dibco/128x256/trained/decoder --maxepoch 20	
	cd ../visualize
	th demo_PDdibco.lua -i ../datasets/xDibco/128x256/valid -d ../resources/Dibco/128x256/trained/ -m pdNet --year ${Aarr[$i]} --dataset db -o ../datasets/xDibco/xOutput_pdnet_128x256 --devID 2
	#mv ../resources/Dibco/128x256/trained/encoder/model-${Aarr[$i]}.net ../resources/Dibco/128x256/trained/encoder/model-${Aarr[$i]}_enc_pt.net
	#mv ../resources/Dibco/128x256/trained/decoder/model-${Aarr[$i]}.net ../resources/Dibco/128x256/trained/decoder/model-${Aarr[$i]}_enc_pt.net
	#mv ../resources/Dibco/128x256/trained/pdNet/model-${Aarr[$i]}.net ../resources/Dibco/128x256/trained/pdNet/model-${Aarr[$i]}_enc_pt.net
	cd -
done
cd ../utils
python dibcoImgJoin.py -i ../datasets/xDibco/xOutput_pdnet_128x256 -o ../datasets/xDibco/jOutput_pdnet_128x256 -c ../../imgDataSets/DIBCO/pngs/img

