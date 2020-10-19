#!/usr/bin/env bash
cd ../tools/DeblurGAN/
echo "method:$1"

if [ $1 == 'mbAdv' ]
then
    for kernelsize in `seq $2 5 $3`
    do
        for translation in `seq $4 0.1 $5`
        do
            # parameter: $1 sbAdv_bim, bim, fgsm, gblur
            dataroot=/path of project/results/dev/inceptionv3_inceptionv3_mbAdv_mifgsm/
            #
            blurredfold="eplison_${translation}_${kernelsize}_stepsize_10_blur_strategy_joint"
            #
            deblurredfold="eplison_${translation}_${kernelsize}_stepsize_10_blur_strategy_joint_deblurgan"
            #
            echo $dataroot$deblurredfold
            #
            mkdir $dataroot$deblurredfold
            #
            python test.py --dataroot $dataroot$blurredfold --model test --dataset_mode single --learn_residual --results_dir $dataroot$deblurredfold
        done
    done
elif [ $1 == 'mbAdv_umot_whole' ]
then
    for kernelsize in `seq $2 5 $3`
    do
        for translation in `seq $4 0.1 $5`
        do
            # parameter: $1 sbAdv_bim, bim, fgsm, gblur
            dataroot=/path of project/results/dev/inceptionv3_inceptionv3_mbAdv_mifgsm/
            #
            blurredfold="eplison_${translation}_${kernelsize}_stepsize_10_blur_strategy_umot_whole"
            #
            deblurredfold="eplison_${translation}_${kernelsize}_stepsize_10_blur_strategy_umot_whole_deblurgan"
            #
            echo $dataroot$deblurredfold
            #
            mkdir $dataroot$deblurredfold
            #
            python test.py --dataroot $dataroot$blurredfold --model test --dataset_mode single --learn_residual --results_dir $dataroot$deblurredfold
        done
    done
elif [ $1 == 'mbAdv_regions' ]
then
    # parameter: $1 sbAdv_bim, bim, fgsm, gblur
    dataroot=/path of project/results/dev/inceptionv3_inceptionv3_mbAdv_mifgsm/
    #
    blurredfold="eplison_0.4_15.0_stepsize_10_blur_strategy_$2"
    #
    deblurredfold="eplison_0.4_15.0_stepsize_10_blur_strategy_$2_deblurgan"
    #
    echo $dataroot$deblurredfold
    #
    mkdir $dataroot$deblurredfold
    #
    python test.py --dataroot $dataroot$blurredfold --model test --dataset_mode single --learn_residual --results_dir $dataroot$deblurredfold
elif [ $1 == 'fgsm' ]
then
    dataroot=/path of project/BlurAttack/results/dev/inceptionv3_inceptionv3_fgsm/
     #
    blurredfold  = "eplison_0.3_stepsize_1.0_blur_strategy_$2"
     #
    deblurredfold="eplison_0.3_stepsize_1.0_blur_strategy_$2_deblurgan"
     #
    echo $dataroot$deblurredfold
     #
    mkdir $dataroot$deblurredfold
     #
    python test.py --dataroot $dataroot$blurredfold --model test --dataset_mode single --learn_residual --results_dir $dataroot$deblurredfold

elif [ $1 == 'mifgsm' ]
then
    dataroot=/path of project/results/dev/inceptionv3_inceptionv3_mifgsm/
     #
    blurredfold  = "eplison_0.3_stepsize_0.03_blur_strategy_$2"
     #
    deblurredfold="eplison_0.3_stepsize_0.03_blur_strategy_$2_deblurgan"
     #
    echo $dataroot$deblurredfold
     #
    mkdir $dataroot$deblurredfold
     #
    python test.py --dataroot $dataroot$blurredfold --model test --dataset_mode single --learn_residual --results_dir $dataroot$deblurredfold
elif [ $1 == 'gblur' ]
then
    dataroot=/path of project/results/dev/inceptionv3_inceptionv3_gblur/
     #
    blurredfold  = "eplison_1.0_stepsize_0.008_blur_strategy_$2"
     #
    deblurredfold="eplison_1.0_stepsize_0.008_blur_strategy_$2_deblurgan"
     #
    echo $dataroot$deblurredfold
     #
    mkdir $dataroot$deblurredfold
     #
    python test.py --dataroot $dataroot$blurredfold --model test --dataset_mode single --learn_residual --results_dir $dataroot$deblurredfold
else
    echo "the method is not existed."
fi
