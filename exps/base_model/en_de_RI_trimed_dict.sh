export START_TRAIN_TIME=`date -R`
home=
echo "high resource translation with mbart: wmt19-ende"
databin=${3:-"$home/data-in/en_XX-de_DE_small"}
warm=${4:-6000}
lr_scheduler=${5:-"inverse_sqrt"}

export MKL_SERVICE_FORCE_INTEL=1;
export MKL_THREADING_LAYER=GNU;
export NCCL_IB_DISABLE=1  

echo ">> training..."
train(){
  export start_TRAIN_TIME=`date -R`
  job_feature=${1:-"bpe_from_scratch"}
  checkpoint_dir=$home/checkpoints/en_XX-de_DE_${job_feature}_trimed_dict
  mkdir -p ${checkpoint_dir}

  set -x
  max_tokens=${2:-8192}
  update_freq=${3:-14}
  lr=${4:-0.0007}
  continue=${5:-0}
  step=${6:-60000}
  dropout=${7:-0.3}

  if [ ${continue} == 0 ];then
    echo ">> begin training from step 0..."
    fairseq-train ${databin} \
      --encoder-normalize-before --decoder-normalize-before \
      --arch mbart_large --layernorm-embedding \
      --save-dir ${checkpoint_dir} \
      --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 20 --no-epoch-checkpoints \
      --share-all-embeddings \
      --source-lang en_XX --target-lang de_DE \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --lr $lr --lr-scheduler ${lr_scheduler} \
      --weight-decay 0.0 --clip-norm 0.0 --dropout ${dropout} \
      --max-update $step --warmup-updates ${warm} \
      --ddp-backend=no_c10d \
      --max-tokens $max_tokens --update-freq $update_freq \
      --fp16 --local_rank $SLURM_LOCALID \
      --reset-dataloader --seed 222 \
      --log-format simple --log-interval 2 --tensorboard-logdir ${checkpoint_dir}/tensorboard > ${checkpoint_dir}/en_XX-de_DE.log
  elif [ ${continue} == 1 ];then
    echo ">> continue training until ${step} steps..."
    PRETRAIN=${checkpoint_dir}/checkpoint_last.pt
    fairseq-train ${databin} \
      --restore-file $PRETRAIN \
      --encoder-normalize-before --decoder-normalize-before \
      --arch mbart_large --layernorm-embedding \
      --save-dir ${checkpoint_dir} \
      --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 20 --no-epoch-checkpoints \
      --share-all-embeddings \
      --source-lang en_XX --target-lang de_DE \
      --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
      --optimizer adam --adam-betas '(0.9, 0.98)' --lr $lr --lr-scheduler ${lr_scheduler} \
      --weight-decay 0.0 --clip-norm 0.0 --dropout ${dropout} \
      --max-update $step --warmup-updates ${warm} \
      --ddp-backend=no_c10d \
      --max-tokens $max_tokens --update-freq $update_freq \
      --fp16 --local_rank $SLURM_LOCALID \
      --seed 222 \
      --log-format simple --log-interval 2 --tensorboard-logdir ${checkpoint_dir}/tensorboard >> ${checkpoint_dir}/en_XX-de_DE.log
  else
      echo ">> error args"
  fi

}

# 112*gpu batch size 917504
train bpe_from_scratch 8192 1 0.0007 1 60000 0.2 


