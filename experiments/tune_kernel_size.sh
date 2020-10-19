#!/bin/bash
cd ..

for kernel_size in `seq $1 5 $2`
do
    echo "Attacking with step_size being $kernel_size"
    python ./blurattack.py -d mnist -g 0 -m mbAdv_mifgsm -i 0 -w stn -e 0.5,$kernel_size
done

cd experiments