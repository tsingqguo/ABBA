#!/bin/bash
cd ..

#for kernel_size in `seq $1 0.02 $2`
#do
#    echo "Attacking with step_size being $kernel_size"
#    python ./blurattack.py -d dev -g 1 -m gblur -i 1 -w inceptionv3 -s $kernel_size -n -1 -e 1 -b whole
#done

#python ./blurattack.py -d dev -g 1 -m gblur -i 1 -w inceptionv3 -s 0.008 -n -1 -e 1 -b whole
#python ./blurattack.py -d dev -g 1 -m gblur -i 1 -w inceptionv3 -s 0.008 -n -1 -e 1 -b obj
#python ./blurattack.py -d dev -g 1 -m gblur -i 1 -w inceptionv3 -s 0.008 -n -1 -e 1 -b bg
python ./blurattack.py -d dev -g 1 -m gblur -i 1 -w inceptionv3 -s 0.008 -n -1 -e 1 -b att


cd experiments

#python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_gblur -s eplison_1.0_stepsize_0.01_blur_strategy_whole
#python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_gblur -s eplison_1.0_stepsize_0.03_blur_strategy_whole
#python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_gblur -s eplison_1.0_stepsize_0.05_blur_strategy_whole
#python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_gblur -s eplison_1.0_stepsize_0.07_blur_strategy_whole
#python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_gblur -s eplison_1.0_stepsize_0.09_blur_strategy_whole