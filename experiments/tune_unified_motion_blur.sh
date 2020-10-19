#!/bin/bash
cd ..
# $5: umot_obj umot_whole umot_bg
for kernel_size in `seq $1 5 $2`
do
    for translation in `seq $3 0.1 $4`
    do
        if [ $6 == "test" ]
        then
            echo "Attacking with translation being $translation"
            python ./blurattack.py -d dev -g $5 -m mbAdv_mifgsm -i 1 -w inceptionv3 -e $translation,$kernel_size -n -2 -b umot_whole
        else
            python blurattack_eval.py -d dev -g $5 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_"$translation"_"$kernel_size"_stepsize_10_blur_strategy_umot_whole
            python blurattack_eval.py -d dev -g $5 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_"$translation"_"$kernel_size"_stepsize_10_blur_strategy_umot_whole_deblurgan
            python blurattack_eval.py -d dev -g $5 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_"$translation"_"$kernel_size"_stepsize_10_blur_strategy_umot_whole_deblurganv2
            python blurattack_eval.py -d dev -g $5 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_"$kernel_size"_stepsize_10_blur_strategy_joint
            python blurattack_eval.py -d dev -g $5 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_"$kernel_size"_stepsize_10_blur_strategy_joint_deblurgan
            python blurattack_eval.py -d dev -g $5 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_"$kernel_size"_stepsize_10_blur_strategy_joint_deblurganv2
        fi
    done
done
#

cd experiments