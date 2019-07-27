#!/bin/bash
#MOD09A1.A2014257.mosaic.7.tiff
for i in *.mosaic.*
do
  echo $i 
  clipshape='../lowa_vector/lowa.dbf'
  gdalwarp -q  -crop_to_cutline -dstalpha -cutline $clipshape -tr 0.001 0.001 -of GTiff $i $i.clip.tif
done

# gdalwarp -q  -crop_to_cutline -dstalpha -cutline $clipshape -tr 0.001 0.001 -of GTiff $i $i.clip.tif
# gdalwarp -ot Float32 -of GTiff -tr 0.0002777777778394884 -0.00027777777783948834 -tap -cutline Lanzhou.shp -crop_to_cutline -dstnodata -32768.0 Lanzhou.tif OUTPUT.tifs