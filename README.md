# Training pdNet

This work is under review for Special issue for Pattern Recognition letters on Graphonomics.

Currently the network can be trained on synthetic and DIBCO dataset:

| Datasets | Input Resolution | Output Resolution^ | # of classes |
|:--------:|:----------------:|:------------------:|:------------:|
| [Synthetic] | 128x256 | 16x32 | 2 |
| [DIBCO] | 128x256 | 16x32 | 2 |

^ is the encoder output resolution; decoder output resolution is the same as that of the input image. 
The datasets are avaialble [here](https://drive.google.com/open?id=1P6Lmo_vwMTJtbPJQLZPkSMlqqd-CapJo)

## Files/folders and their usage:

* [run.lua](train/run.lua)    : main file
* [opts.lua](train/opts.lua)  : contains all the input options used by the tranining script
* [data](data)          : data loaders for loading datasets
* models                : all the model architectures are defined here
* [train.lua](train/train.lua) : loading of models and error calculation
* [test.lua](train/test.lua)  : calculate testing error and save confusion matrices

## Example command for testing the code on DIBCO:
### Training encoder Synthetic data:
```
th run.lua --dataset sn --year 0000 --datapath ../datasets/xSynthetic/128x256 --cachepath ../resources/Dibco/128x256/cache/encoder --model models/encoder.lua --save ../resources/Dibco/128x256/trained/encoder --imHeight 128 --imWidth 256 --labelHeight 16 --labelWidth 32 --batchSize 30 --maxepoch 20
```
### Training decoder Synthetic data:
```
th run.lua --dataset sn --year 0000 --datapath ../datasets/xSynthetic/128x256 --cachepath ../resources/Dibco/128x256/cache/decoder --model models/decoder.lua --save ../resources/Dibco/128x256/trained/decoder --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 30 --maxepoch 20 --CNNModel ../resources/Dibco/128x256/trained/encoder
```

### Training encoder on DIBCO data for year 2009 using pre-trained encoder model:
```
th run.lua --dataset db --year 2009 --datapath ../datasets/xDibco/128x256 --cachepath ../resources/xDibco/128x256/cache/encoder --model models/encoder.lua --save ../resources/xDibco/128x256/trained/encoder --imHeight 128 --imWidth 256 --labelHeight 16 --labelWidth 32 --batchSize 30 --maxepoch 20 --ptModel ../resources/xDibco/128x256/trained/encoder/model-0.net
```

### Training decoder on DIBCO data for year 2009 using pre-trained decoder model:
```
th run.lua --dataset db --year 2009 --datapath ../datasets/xDibco/128x256 --cachepath ../resources/xDibco/128x256/cache/decoder --model models/decoder.lua --save ../resources/xDibco/128x256/trained/decoder --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 30 --maxepoch 20 --CNNModel ../resources/xDibco/128x256/trained/encoder --ptModel ../resources/xDibco/128x256/trained/decoder/model-0.net
```

### Training pdNet on DIBCO data for year 2009 using trained decoder model:
```
th run.lua --dataset db --year 2009 --datapath ../datasets/xDibco/128x256 --cachepath ../resources/xDibco/128x256/cache/decoder --model models/pdNet.lua --save ../resources/xDibco/128x256/trained/pdNet --imHeight 128 --imWidth 256 --labelHeight 128 --labelWidth 256 --batchSize 20 --CNNModel ../resources/xDibco/128x256/trained/decoder --maxepoch 20	
```

### Test the output on DIBCO 2009 dataset:
```
cd ../visualize
th demo_PDdibco.lua -i ../datasets/xDibco/128x256/valid -d ../resources/xDibco/128x256/trained/ -m pdNet --year 2009 --dataset db -o ../datasets/xDibco/xOutput_pdnet_128x256 --devID 2
```    
