#!/bin/bash
year=$1
dir=`pwd`
cd $dir/$year
for day in *
do
  #echo $day
  cd $dir/$year/$day
  for i in *.hdf
  do 
     mv $i $dir/
  done
done

