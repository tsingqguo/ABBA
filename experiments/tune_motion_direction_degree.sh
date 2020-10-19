#!/bin/bash
cd ..

for fg_deg in `seq $1 20 $2`
do
    for bg_deg in `seq $3 20 $4`
    do
        if [ $5 == "test" ]
        then
            echo "Attacking with translation being $translation"
            python ./blurattack.py -d dev -g $6 -m mbAdv_mifgsm -i 1 -w inceptionv3 -e 0.4,15.0 -r $fg_deg,$bg_deg
        else
            #echo "Evaling with translation being $translation"
            python blurattack_eval.py -d dev -g $6 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_direction_"$fg_deg"_"$bg_deg"_blur_strategy_joint
        fi
    done
done


cd experiments