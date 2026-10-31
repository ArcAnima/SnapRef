#!/bin/bash
# Features (imgutils in ~/ccip-venv, torch in ~/comfy-venv) of one renders dir of the detail eval.
#   train/detail_eval/features.sh ~/detail_eval/renders/<model>
set -e
unset LD_LIBRARY_PATH
cd "$(dirname "$0")/../.."
SET=${SET:-$HOME/detail_eval/set}
~/ccip-venv/bin/python train/detail_eval/features_imgutils.py --set "$SET" --renders "$1"
~/comfy-venv/bin/python train/detail_eval/features_torch.py --set "$SET" --renders "$1"
