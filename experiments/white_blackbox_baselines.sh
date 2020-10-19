#!/usr/bin/env bash
# this script is for running the baseline methods of mifgsm, fgsm, gaussian blur, etc.
cd ..
whitemodelname=$3
gpuid=$4
# parameter: $1 sbAdv_bim, bim, fgsm, gblur
# res50_res50_sbAdv_bim_imagenet
if [ $1 == "mifgsm" ]
then
    #
    if [ $2 == "eval" ]
    then
        python blurattack_eval.py -d dev -g 0 -r 1 -p $whitemodelname"_"$whitemodelname"_mifgsm" -w $whitemodelname
    else
        python ./blurattack.py -d dev -g $gpuid -m mifgsm -i 1 -w $whitemodelname -s 0.03 -n -3 -e 0.3 -b whole
        python ./blurattack.py -d dev -g $gpuid -m mifgsm -i 1 -w $whitemodelname -s 0.03 -n -3 -e 0.3 -b obj
        python ./blurattack.py -d dev -g $gpuid -m mifgsm -i 1 -w $whitemodelname -s 0.03 -n -3 -e 0.3 -b bg
        python ./blurattack.py -d dev -g $gpuid -m mifgsm -i 1 -w $whitemodelname -s 0.03 -n -3 -e 0.3 -b att
    fi
    #
elif [ $1 == "fgsm" ]
then
    #
    if [ $2 == "eval" ]
    then
        echo $whitemodelname"_"$whitemodelname"_fgsm"
        python blurattack_eval.py -d dev -g 0 -r 1 -p $whitemodelname"_"$whitemodelname"_fgsm" -w $whitemodelname
    else
        python ./blurattack.py -d dev -g $gpuid -m fgsm -i 1 -w $whitemodelname -s 1 -n -3 -e 0.3 -b whole
        python ./blurattack.py -d dev -g $gpuid -m fgsm -i 1 -w $whitemodelname -s 1 -n -3 -e 0.3 -b obj
        python ./blurattack.py -d dev -g $gpuid -m fgsm -i 1 -w $whitemodelname -s 1 -n -3 -e 0.3 -b bg
        python ./blurattack.py -d dev -g $gpuid -m fgsm -i 1 -w $whitemodelname -s 1 -n -3 -e 0.3 -b att
    fi
    #
elif [ $1 == "gblur" ]
then
    #
    if [ $2 == "eval" ]
    then
        echo $whitemodelname"_"$whitemodelname"_gblur"
        python blurattack_eval.py -d dev -g 0 -r 1 -p $whitemodelname"_"$whitemodelname"_gblur" -w $whitemodelname
    else
        python ./blurattack.py -d dev -g $gpuid -m gblur -i 1 -w $whitemodelname -s 0.0063 -n -3 -e 0.063 -b whole
        python ./blurattack.py -d dev -g $gpuid -m gblur -i 1 -w $whitemodelname -s 0.0063 -n -3 -e 0.063 -b obj
        python ./blurattack.py -d dev -g $gpuid -m gblur -i 1 -w $whitemodelname -s 0.0063 -n -3 -e 0.063 -b bg
        python ./blurattack.py -d dev -g $gpuid -m gblur -i 1 -w $whitemodelname -s 0.0063 -n -3 -e 0.063 -b att
    #
    fi
elif [ $1 == "dblur" ]
then
    if [ $2 == "eval" ]
    then
        echo $whitemodelname"_"$whitemodelname"_dblur"
        python blurattack_eval.py -d dev -g 0 -r 1 -p $whitemodelname"_"$whitemodelname"_dblur" -w $whitemodelname
    else
        python ./blurattack.py -d dev -g $gpuid -m dblur -i 1 -w $whitemodelname -s 7.0 -n -3 -e 1 -b whole
        python ./blurattack.py -d dev -g $gpuid -m dblur -i 1 -w $whitemodelname -s 7.0 -n -3 -e 1 -b obj
        python ./blurattack.py -d dev -g $gpuid -m dblur -i 1 -w $whitemodelname -s 7.0 -n -3 -e 1 -b bg
        python ./blurattack.py -d dev -g $gpuid -m dblur -i 1 -w $whitemodelname -s 7.0 -n -3 -e 1 -b att
    fi
fi

cd experiments
