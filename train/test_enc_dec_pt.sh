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
	echo ${XSY_CLR_GS[$j]}
	for ii in `seq ${#DSYEAR[@]}`;
	do
		i=$(expr $ii - 1)
		echo ${DSYEAR[$i]}
	done
done
