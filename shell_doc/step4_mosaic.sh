#!/bin/bash
#gdalwarp  -srcnodata -28672 -dstnodata -28672 -of GTiff MOD09A1.A2010105.h11v04.006.2015207023033.hdf.01.tif.tiff MOD09A1.A2010105.h10v04.006.2015207023933.hdf.01.tif.tiff  MOD09A1.A2010105.mosaic.01.tiff
#MOD09A1.A2014257.h11v04.006.2015290233257.hdf.07.tif.tiff
if [ $# != 1 ]
then
 echo 'input the band index 01 02 and so on'
 exit
fi

band=$1
for i in *h11v04*$band.tif.tiff
do
  echo $i
  base=`echo $i|cut -d'.' -f1-2`
  imgs=`ls $base*$band.tif.tiff`
  nodata=`gdalinfo $i|grep 'NoData Value'|cut -d'=' -f2`
  #rm $base.mosaic.$band.tiff
  gdalwarp  -srcnodata $nodata -dstnodata $nodata -of GTiff $imgs $base.mosaic.$band.tiff
  #base=`echo ls|grep $i|cut -d'.' -f1-2`
  #base_mosic=echo $base|grep '1.tif.tiff'
  #echo $base_mosic
  #exit
done

# gdalwarp  -srcnodata -32768 -dstnodata -32768 -of GTiff 