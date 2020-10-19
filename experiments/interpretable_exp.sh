#!/usr/bin/env bash
cd ..
# parameter: $1 sbAdv_bim, bim, fgsm, gblur
# res50_res50_sbAdv_bim_imagenet
python ./blurregion_interpretation.py -d dev -g 1 -m mbAdv_mifgsm -i 1 -w inceptionv3 -e 0.4,15.0 -b joint
python ./blurregion_interpretation.py -d dev -g 1 -m mifgsm -i 1 -w inceptionv3 -s 0.03 -n -3 -e 0.3 -b whole
python ./blurregion_interpretation.py -d dev -g 1 -m gblur -i 1 -w inceptionv3 -s 0.008 -n -3 -e 1.0 -b whole
python ./blurregion_interpretation.py -d dev -g 1 -m fgsm -i 1 -w inceptionv3 -s 1 -n -3 -e 0.3 -b whole

cd experiments