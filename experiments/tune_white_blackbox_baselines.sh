#!/usr/bin/env bash
# this code is used to run the baseline methods with different eplison
cd ..
# parameter: $1 sbAdv_bim, bim, fgsm, gblur
# res50_res50_sbAdv_bim_imagenet
if [ $1 == "bim" ]
then
    #
    python ./blurattack.py -d dev -g 0 -m bim -i 1 -w inceptionv3 -s 0.05 -n -3 -e 0.3 -b whole
    #
elif [ $1 == "cw" ]
then
    #
    python ./blurattack.py -d dev -g 0 -m cw -i 1 -w inceptionv3 -s 0.05 -n -3 -e 5 -b whole
    #
elif [ $1 == "mifgsm" ]
then
    #
    if [ $2 == "eval" ]
    then
        python blurattack_eval.py -d dev -g 0 -r 1 -p inceptionv3_inceptionv3_mifgsm
    else
        python ./blurattack.py -d dev -g 1 -m mifgsm -i 1 -w inceptionv3 -s 0.03 -n -1 -e $3 -b whole
    fi
    #
elif [ $1 == "fgsm" ]
then
    #
    if [ $2 == "eval" ]
    then
        python blurattack_eval.py -d dev -g 0 -r 1 -p inceptionv3_inceptionv3_fgsm
    else
        python ./blurattack.py -d dev -g 1 -m fgsm -i 1 -w inceptionv3 -s 1 -n -3 -e $3 -b whole
    fi
    #
elif [ $1 == "gblur" ]
then
    #
    python ./blurattack.py -d dev -g 1 -m gblur -i 1 -w inceptionv3 -s $3 -n -3 -e 1 -b whole
    #
elif [ $1 == "dblur" ]
then
    if [ $2 == "eval" ]
    then
        python blurattack_eval.py -d dev -g 0 -r 1 -p inceptionv3_inceptionv3_dblur
    else
        python ./blurattack.py -d dev -g 1 -m dblur -i 1 -w inceptionv3 -s $3 -n -3 -e 1 -b whole
    fi
elif [ $1 == "mblur" ]
then
    if [ $2 == "eval" ]
    then
        python blurattack_eval.py -d dev -g 0 -r 1 -p inceptionv3_inceptionv3_mblur
    else
        for kernel_size in `seq $3 5 $4`
        do
            python ./blurattack.py -d dev -g 1 -m mblur -i 1 -w inceptionv3 -s $kernel_size -n -3 -e 1 -b whole
        done
    fi
    #
fi

cd experiments
