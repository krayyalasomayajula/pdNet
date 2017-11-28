--------------------------------------------------------------------------------
-- Contains options required by demo.lua
--
-- Written by: Abhishek Chaurasia
-- Dated: 24th March, 2016
--------------------------------------------------------------------------------

local opts = {}

lapp = require 'pl.lapp'
function opts.parse(arg)
   local opt = lapp [[
   Command line options:
   ## I/O
   -i, --input    (default path/video.mp4)
                  Input folder/file location for image/video respectively or cam0 for camera
   -o, --output   (default path/video.mp4)
                  Output folder/file location for image/video respectively or cam0 for camera
   -r, --ratio    (default 1)
                  Rescale image size for faster processing
   -v, --verbose  Verbose mode
   -z, --zoom     (default 1)
                  Zoom for display
   --camRes       (default VGA)
                  Resolution of the camera: QHD/VGA/FWVGA/HD/FHD
   --dataset      (default cs)
                  cityscapes/camvid/sun: cs/cv/su
   --estBG        (default 0)
   
   ## Model
   -d, --dmodel   (default /media/HDD1/Models/)
                  Folder location of models
   -m, --model    (default enc_2_1)
                  Model version
   --year         (default 2009)        
                  Indicate the year of Dibco dataset to test

   ## Device
   --dev          (default cuda)
                  Device to be used: cpu/cuda
   --devID        (default 1)
                  GPU number
   ## For clean up
   --batch        (default 1)
   ]]

   return opt
end

return opts
