#!/bin/bash

for i in $(ls -d */); do
    #move the files into their correct directory
    mv ${i::-1}*.JPEG $i
done