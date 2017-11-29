set -e

PRETRAINED_CHECKPOINT_DIR=run/checkpoint #The directory where checkpoints are located.
TRAIN_DIR=../result #The directory where you want to save your trained models.
DATASET_DIR=../process #The directory where processed.tf records are located.


python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=tumors \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=inception_v3 \
  --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR}/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=20000 \
  --batch_size=32 \
  --learning_rate=0.01 \
  --save_interval_secs=100 \
  --save_summaries_secs=100 \
  --log_every_n_steps=300 \
  --optimizer=rmsprop \
  --weight_decay=0.00004 \
  --clone_on_cpu=True
