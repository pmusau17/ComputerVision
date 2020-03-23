#!/bin/bash

#for each file in the tar, extract it and remove it from the tar
testseq="tar"
for i in $(tar -tf ILSVRC2012_img_train.tar | tac); do
    if [[ $i =~ $testseq ]];
    then
       tar -xvf ILSVRC2012_img_train.tar $i && tar --verbose --delete --file=ILSVRC2012_img_train.tar $i || exit 1
       echo $i
    else 
       echo "False"
    fi
done