#!/bin/bash
## declare an array of subject names
declare -a DSYEAR=("2009"
"2010"
"2011"
"2012"
"2013"
"2014"
"2016"
)
declare -a DB_CACHE=("DibcoGray"
"Dibco"
)
declare -a xDB_CLR_GS=("xDibcoGray"
"xDibco"
)
declare -a XSY_CLR_GS=("xSyntheticGray"
"xSynthetic"
)
## declare an array of subject links in youtube
for jj in `seq ${#DB_CACHE[@]}`;
do
	j=$(expr $jj - 1)
	th run.lua --dataset sn --year 0000 --datapath ../datasets/${XSY_CLR_GS[$j]}/128x256 --cachepath ../resources/${DB_CACHE[$j]}/128x256/cache/encoder --model models/encoder.lua --save ../resources/${DB_CACHE[$j]}/128x256/trained/encoder --imHeight 128 --imWidth 256 --labelHeight 16 --labelWidth 32 --batchSize 30 --maxepoch 20
	th run.lua --dataset sn --year 0000 --datapath ../datasets/${XSY_CLR_GS[$j]}/128x256 --cachepath ../resources/${DB_CACHE[$j]}/128x256/cache/decoder --model models/decoder.lua --save ../resources/${DB_CACHE[$j]}/128x256/trained/decoder --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 30 --maxepoch 20 --CNNModel ../resources/${DB_CACHE[$j]}/128x256/trained/encoder
	for ii in `seq ${#DSYEAR[@]}`;
	do
		i=$(expr $ii - 1)
		th run.lua --dataset db --year ${DSYEAR[$i]} --datapath ../datasets/${xDB_CLR_GS[$j]}/128x256 --cachepath ../resources/${DB_CACHE[$j]}/128x256/cache/encoder --model models/encoder.lua --save ../resources/${DB_CACHE[$j]}/128x256/trained/encoder --imHeight 128 --imWidth 256 --labelHeight 16 --labelWidth 32 --batchSize 30 --maxepoch 20 --ptModel ../resources/${DB_CACHE[$j]}/128x256/trained/encoder/model-0.net
		th run.lua --dataset db --year ${DSYEAR[$i]} --datapath ../datasets/${xDB_CLR_GS[$j]}/128x256 --cachepath ../resources/${DB_CACHE[$j]}/128x256/cache/decoder --model models/decoder.lua --save ../resources/${DB_CACHE[$j]}/128x256/trained/decoder --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 30 --maxepoch 20 --CNNModel ../resources/${DB_CACHE[$j]}/128x256/trained/encoder --ptModel ../resources/${DB_CACHE[$j]}/128x256/trained/decoder/model-0.net
		th run.lua --dataset db --year ${DSYEAR[$i]} --datapath ../datasets/${xDB_CLR_GS[$j]}/128x256 --cachepath ../resources/${DB_CACHE[$j]}/128x256/cache/decoder --model models/pdNet.lua --save ../resources/${DB_CACHE[$j]}/128x256/trained/pdNet --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 20 --CNNModel ../resources/${DB_CACHE[$j]}/128x256/trained/decoder --maxepoch 20	
		cd ../visualize
		th demo_PDdibco.lua -i ../datasets/${xDB_CLR_GS[$j]}/128x256/valid -d ../resources/${DB_CACHE[$j]}/128x256/trained/ -m pdNet --year ${DSYEAR[$i]} --dataset db -o ../datasets/${xDB_CLR_GS[$j]}/xOutput_pdnet_128x256 --devID 2
		cd -
	done
	cd ../utils
	python dibcoImgJoin.py -i ../datasets/${xDB_CLR_GS[$j]}/xOutput_pdnet_128x256 -o ../datasets/${xDB_CLR_GS[$j]}/jOutput_pdnet_128x256 -c ../../imgDataSets/DIBCO/pngs/img
done
