#!/bin/bash
cd ..
# bbmodel_names = ["inceptionresnetv2","inceptionv3","inceptionv4","xception"]
gpuid=$2
python ./blurattack.py -d dev -g $gpuid -m mbAdv_mifgsm -i 1 -w $1 -e 0.4,15.0 -b whole
python ./blurattack.py -d dev -g $gpuid -m mbAdv_mifgsm -i 1 -w $1 -e 0.4,15.0 -b obj
python ./blurattack.py -d dev -g $gpuid -m mbAdv_mifgsm -i 1 -w $1 -e 0.4,15.0 -b backg
python ./blurattack.py -d dev -g $gpuid -m mbAdv_mifgsm -i 1 -w $1 -e 0.4,15.0 -b joint
python ./blurattack.py -d dev -g $gpuid -m mbAdv_mifgsm -i 1 -w $1 -e 0.4,15.0 -b pixel -n 700


# evaluation:
#python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_whole
#python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_obj
#python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_backg
#python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_pixel








