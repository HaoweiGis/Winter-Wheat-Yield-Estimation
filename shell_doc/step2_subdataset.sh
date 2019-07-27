#!/bin/bash


for band in `seq 1 7`
do
  for i in MOD09A1.A2013*.hdf
  do
    echo $i
    subset=`gdalinfo $i|grep 'SUBDATASET_'$band'_NAME'|cut -d'=' -f2`
    #echo subset$subset
    gdal_translate $subset $i.0$band.tif;
  done
done
