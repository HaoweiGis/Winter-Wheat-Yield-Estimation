#!/bin/bash
#MOD09A1.A2013289.mosaic.7.tiff.clip.tiff.mask.tiff

for i in *mask.tiff
do
  name=`echo $i|cut -d'.' -f2`;echo $name
  names=`ls *.mask.tiff|grep $name`
  echo $names
  gdalbuildvrt -separate $name.vrt $names
  gdal_translate $name.vrt moraic/$name.tif
done
