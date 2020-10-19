#!/bin/bash
# $1: test or eval
# $2: deblurganv2, deblurgan, none
#bash attack_deblurred_advexamples.sh fgsm $2 $1 whole
#bash attack_deblurred_advexamples.sh fgsm $2 $1 obj
#bash attack_deblurred_advexamples.sh fgsm $2 $1 bg
#bash attack_deblurred_advexamples.sh fgsm $2 $1 att
###
#bash attack_deblurred_advexamples.sh mifgsm $2 $1 whole
#bash attack_deblurred_advexamples.sh mifgsm $2 $1 obj
#bash attack_deblurred_advexamples.sh mifgsm $2 $1 bg
##
#bash attack_deblurred_advexamples.sh gblur $2 $1 whole
#bash attack_deblurred_advexamples.sh gblur $2 $1 obj
#bash attack_deblurred_advexamples.sh gblur $2 $1 bg
#bash attack_deblurred_advexamples.sh gblur $2 $1 att
##
#bash attack_deblurred_advexamples.sh mbAdv $2 $1 15.0 15.0 0.4 0.4 pixel
#bash attack_deblurred_advexamples.sh mbAdv $2 $1 15.0 15.0 0.4 0.4 obj
#bash attack_deblurred_advexamples.sh mbAdv $2 $1 15.0 15.0 0.4 0.4 bg
#bash attack_deblurred_advexamples.sh mbAdv $2 $1 15.0 15.0 0.4 0.4 whole
#bash attack_deblurred_advexamples.sh mbAdv $2 $1 15.0 15.0 0.4 0.4 joint
#bash attack_deblurred_advexamples.sh mbAdv None eval 5.0 15.0 0.1 0.4 umot_obj


#echo mbAdv_original
#bash attack_deblurred_advexamples.sh mbAdv None eval 5.0 35.0 0.4 0.4 joint
#echo mbAdv_deblurgan
#bash attack_deblurred_advexamples.sh mbAdv deblurgan eval 5.0 35.0 0.4 0.4 joint
#echo mbAdv_deblurganv2
#bash attack_deblurred_advexamples.sh mbAdv deblurganv2 eval 5.0 35.0 0.4 0.4 joint
#

#echo mbAdv_original
#bash attack_deblurred_advexamples.sh mbAdv None eval 5.0 35.0 0.16 0.16 umot_whole
#echo mbAdv_deblurgan
#bash attack_deblurred_advexamples.sh mbAdv deblurgan eval 5.0 35.0 0.16 0.16 umot_whole
#echo mbAdv_deblurganv2
#bash attack_deblurred_advexamples.sh mbAdv deblurganv2 eval 5.0 35.0 0.16 0.16 umot_whole


if [ $1 == 'umot_whole' ]
then
    #testing retrained deblurganv2 on umot_whole
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_DVD_best_fpn_null_abba test 15.0 15.0 0.16 0.16 umot_whole
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_DVD_best_fpn_office_abba test 15.0 15.0 0.16 0.16 umot_whole
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_GOPRO_NFS_GOPRO_best_fpn_null_abba test 15.0 15.0 0.16 0.16 umot_whole
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_GOPRO_NFS_GOPRO_best_fpn_office_abba test 15.0 15.0 0.16 0.16 umot_whole
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_GOPRO_best_fpn_null_abba test 15.0 15.0 0.16 0.16 umot_whole
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_GOPRO_best_fpn_office_abba test 15.0 15.0 0.16 0.16 umot_whole
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_NFS_best_fpn_null_abba test 15.0 15.0 0.16 0.16 umot_whole
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_NFS_best_fpn_office_abba test 15.0 15.0 0.16 0.16 umot_whole
    cd ..
    # eval:
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.16_15.0_stepsize_10_blur_strategy_umot_whole_deblurganv2_DVD_best_fpn_null_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.16_15.0_stepsize_10_blur_strategy_umot_whole_deblurganv2_DVD_best_fpn_office_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.16_15.0_stepsize_10_blur_strategy_umot_whole_deblurganv2_GOPRO_NFS_GOPRO_best_fpn_null_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.16_15.0_stepsize_10_blur_strategy_umot_whole_deblurganv2_GOPRO_NFS_GOPRO_best_fpn_office_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.16_15.0_stepsize_10_blur_strategy_umot_whole_deblurganv2_GOPRO_best_fpn_null_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.16_15.0_stepsize_10_blur_strategy_umot_whole_deblurganv2_GOPRO_best_fpn_office_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.16_15.0_stepsize_10_blur_strategy_umot_whole_deblurganv2_NFS_best_fpn_null_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.16_15.0_stepsize_10_blur_strategy_umot_whole_deblurganv2_NFS_best_fpn_office_abba
else
    #testing retrained deblurganv2 on joint
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_DVD_best_fpn_null_abba test 15.0 15.0 0.4 0.4 joint
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_DVD_best_fpn_office_abba test 15.0 15.0 0.4 0.4 joint
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_GOPRO_NFS_GOPRO_best_fpn_null_abba test 15.0 15.0 0.4 0.4 joint
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_GOPRO_NFS_GOPRO_best_fpn_office_abba test 15.0 15.0 0.4 0.4 joint
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_GOPRO_best_fpn_null_abba test 15.0 15.0 0.4 0.4 joint
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_GOPRO_best_fpn_office_abba test 15.0 15.0 0.4 0.4 joint
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_NFS_best_fpn_null_abba test 15.0 15.0 0.4 0.4 joint
#    bash attack_deblurred_advexamples.sh mbAdv deblurganv2_NFS_best_fpn_office_abba test 15.0 15.0 0.4 0.4 joint
    cd ..
    # eval:
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_joint_deblurganv2_DVD_best_fpn_null_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_joint_deblurganv2_DVD_best_fpn_office_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_joint_deblurganv2_GOPRO_NFS_GOPRO_best_fpn_null_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_joint_deblurganv2_GOPRO_NFS_GOPRO_best_fpn_office_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_joint_deblurganv2_GOPRO_best_fpn_null_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_joint_deblurganv2_GOPRO_best_fpn_office_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_joint_deblurganv2_NFS_best_fpn_null_abba
    python blurattack_eval.py -d dev -g 1 -r 1 -p inceptionv3_inceptionv3_mbAdv_mifgsm -s eplison_0.4_15.0_stepsize_10_blur_strategy_joint_deblurganv2_NFS_best_fpn_office_abba
fi