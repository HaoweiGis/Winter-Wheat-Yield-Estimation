#!/bin/bash
for i in *.mask.tiff
do
  gdal_calc.py -A $i --outfile=$i.calc.tif --type='Int16' --calc="A*(A>0)" --NoDataValue=0
done
