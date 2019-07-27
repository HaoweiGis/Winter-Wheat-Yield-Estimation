#!/bin/bash
#MOD09A1.A2006041.mosaic.1.tiff.clip.tiff
year=$1
for i in MOD09A1.A$year*.clip.tif
do
  echo $i 
  gdal_calc.py -A ../LANDCOVER/$year.tif -B $i --outfile=$i.mask.tiff  --calc="A*B"
done
