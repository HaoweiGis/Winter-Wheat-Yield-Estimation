#!/bin/bash
for i in *.tif
do 
  echo $i
  gdalwarp -t_srs 'EPSG:4326' $i $i.tiff
done
