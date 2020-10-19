#!/usr/bin/env bash
#bash deblurGanv2.sh fgsm whole
#bash deblurGanv2.sh fgsm obj
#bash deblurGanv2.sh fgsm bg
#bash deblurGanv2.sh fgsm att
##
#bash deblurGanv2.sh mifgsm whole
#bash deblurGanv2.sh mifgsm obj
#bash deblurGanv2.sh mifgsm bg
##
#bash deblurGanv2.sh gblur whole
#bash deblurGanv2.sh gblur obj
#bash deblurGanv2.sh gblur bg
#bash deblurGanv2.sh gblur att
#
#bash deblurGanv2.sh mbAdv_regions whole
#bash deblurGanv2.sh mbAdv_regions obj
#bash deblurGanv2.sh mbAdv_regions bg
#bash deblurGanv2.sh mbAdv_regions att
#
#echo mbAdv_original
#bash attack_deblurred_advexamples.sh mbAdv None eval 5.0 35.0 0.4 0.4 joint
#echo mbAdv_deblurgan
#bash attack_deblurred_advexamples.sh mbAdv deblurgan eval 5.0 35.0 0.4 0.4 joint
#echo mbAdv_deblurganv2
#bash attack_deblurred_advexamples.sh mbAdv deblurganv2 eval 5.0 35.0 0.4 0.4 joint
#
echo mbAdv_umot_whole
bash deblurGanv2.sh mbAdv_umot_whole 5.0 35.0 0.16 0.16
#
echo mbAdv
bash deblurGanv2.sh mbAdv 5.0 35.0 0.16 0.16

