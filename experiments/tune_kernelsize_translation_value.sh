#!/bin/bash
cd ..

for kernel_size in `seq $1 5 $2`
do
    for translation in `seq $3 0.1 $4`
    do
        echo "Attacking with translation being $translation"
        python ./blurattack.py -d sharp -g 1 -m mbAdv_mifgsm -i 0 -w inceptionv3 -e $translation,$kernel_size -b whole
    done
done
#

cd experiments