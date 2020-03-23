#!/bin/bash

#for each file in the tar, extract it and delete it
testseq="tar"
for i in $(ls *.tar); do
    #if [[ $i =~ $testseq ]];
    #then
    tar -xvf $i && rm $i || exit 1
    echo $i
done