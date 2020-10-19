#!/bin/bash
cd ..

for step_size in `seq $1 $2`
do
    echo "Attacking with step_size being $step_size"
    python ./blurattack.py -d dev -g 0 -m mbAdv_mifgsm -i 1 -w inceptionv3 -s $step_size
done

cd experiments