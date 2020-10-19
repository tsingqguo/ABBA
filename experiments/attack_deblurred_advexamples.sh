#!/bin/bash
cd ..

deblurgan=$2

echo "deblurring method:$deblurgan"

if [ $1 == 'mbAdv' ]
then
    for kernel_size in `seq $4 5.0 $5`
    do
        for translation in `seq $6 0.1 $7`
        do
            if [ $3 == 'test' ]
            then
                echo "Attacking with translation being $translation"
                python ./blurattack.py -d dev -g 1 -m mbAdv_mifgsm -i 1 -w inceptionv3 -e $translation,$kernel_size -u $deblurgan -b $8
            else
                if [ $deblurgan == 'None' ]
                then
                    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_"$translation"_"$kernel_size"_stepsize_10_blur_strategy_"$8"
                else
                    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_"$translation"_"$kernel_size"_stepsize_10_blur_strategy_"$8"_"$deblurgan"
                fi
            fi
        done
    done
elif [ $1 == 'fgsm' ]
then
     if [ $3 == 'test' ]
     then
         echo "Attacking with fgsm"
         python ./blurattack.py -d dev -g 1 -m fgsm -i 1 -w inceptionv3 -s 1 -n -3 -e 0.3 -b $4 -u $deblurgan
     else
         if [ $deblurgan == 'None' ]
         then
            python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_fgsm -s eplison_0.3_stepsize_1.0_blur_strategy_"$4"
         else
            python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_fgsm -s eplison_0.3_stepsize_1.0_blur_strategy_"$4"_"$deblurgan"
         fi
     fi
elif [ $1 == 'mifgsm' ]
then
     if [ $3 == 'test' ]
     then
         echo "Attacking with mifgsm"
         python ./blurattack.py -d dev -g 1 -m mifgsm -i 1 -w inceptionv3 -s 0.03 -n -3 -e 0.3 -b $4 -u $deblurgan
     else
        if [ $deblurgan == 'None' ]
        then
            python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mifgsm -s eplison_0.3_stepsize_0.03_blur_strategy_"$4"
        else
            python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mifgsm -s eplison_0.3_stepsize_0.03_blur_strategy_"$4"_"$deblurgan"
        fi
     fi
elif [ $1 == 'gblur' ]
then
     if [ $3 == 'test' ]
     then
         echo "Attacking with gblur"
         python ./blurattack.py -d dev -g 1 -m gblur -i 1 -w inceptionv3 -s 0.008 -n -3 -e 1.0 -b $4 -u $deblurgan
     else
        if [ $deblurgan == 'None' ]
        then
            python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_gblur -s eplison_1.0_stepsize_0.008_blur_strategy_"$4"
        else
            python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_gblur -s eplison_1.0_stepsize_0.008_blur_strategy_"$4"_"$deblurgan"
        fi
     fi
else
    echo "the method is not existed."
fi




cd experiments