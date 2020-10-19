#!/usr/bin/env bash
#
bash deblurGan.sh mbAdv_regions whole
bash deblurGan.sh mbAdv_regions obj
bash deblurGan.sh mbAdv_regions bg
bash deblurGan.sh mbAdv_regions att
#
bash deblurGan.sh fgsm whole
bash deblurGan.sh fgsm obj
bash deblurGan.sh fgsm bg
bash deblurGan.sh fgsm att
#
bash deblurGan.sh mifgsm whole
bash deblurGan.sh mifgsm obj
bash deblurGan.sh mifgsm bg
#
bash deblurGan.sh gblur whole
bash deblurGan.sh gblur obj
bash deblurGan.sh gblur bg
bash deblurGan.sh gblur att
#
bash deblurGan.sh dblur whole
bash deblurGan.sh dblur obj
bash deblurGan.sh dblur bg
bash deblurGan.sh dblur att

