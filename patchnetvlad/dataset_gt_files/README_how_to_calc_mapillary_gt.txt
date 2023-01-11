README

In this readme I will explain how to evaluate the ground truth for the mapillary val set (consisting of sf and cph cities).

Prerequisites: 
1) Please download the Github repo: https://github.com/mapillary/mapillary_sls
This repo is required to evaluate the ground truth on the Mapillary dataset.
2) Please download the msls dataset: https://www.mapillary.com/dataset/places

To calculate the recall based on the ground truth, please run the commands below, replacing paths with your own as required:

cat results/mapillarycph/PatchNetVLAD_predictions.txt results/mapillarysf/PatchNetVLAD_predictions.txt > results/PatchNetVLAD_predictions_combined_mapval.txt
python ./patchnetvlad/training_tools/convert_kapture_to_msls.py /path/to/PatchNetVLAD_predictions_combined.txt /save/path/for/PatchNetVLAD_predictions_combined_msls.txt
python /path/to/mapillary_sls/evaluate.py --msls-root=/path/to/msls/dataset --cities=cph,sf --prediction=path/to/PatchNetVLAD_predictions_combined_msls.txt
