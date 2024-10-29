#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh

export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=1

max_epochs=30
experiment="iccv_hadr_unet_experiments"
batch_size=32
# Change debug to False if you want to run on the full dataset
debug=True

for target_shift in 1 
do
  echo "Experiment with target_shift=${target_shift}"
  python src/train.py target_shift=${target_shift} datamodule.debug=${debug} trainer.max_epochs=${max_epochs} datamodule.batch_size=${batch_size} logger=wandb logger.wandb.project=${experiment} model.loss=ce model.encoder="efficientnet-b1" experiment=unet
done