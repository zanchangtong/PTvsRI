home=
continue=${1:-0}
step=${2:-100000}
lr=${3:-3e-05}
checkpoint_dir=${home}/checkpoints/trimed_en_XX-de_DE_PT
export NCCL_IB_DISABLE=1  

mkdir $checkpoint_dir
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
# 16*gpu
if [ ${continue} == 0 ];then
    echo ">> begin fine-tuning from step 0..."
    PRETRAIN=$home/2_High_Resource_Translation/checkpoints/trimed_mbart/19ende_trimed_mbart.pt
    fairseq-train ${home}/data-in/en_XX-de_DE_trimed \
        --save-dir ${checkpoint_dir} \
        --encoder-normalize-before --decoder-normalize-before \
        --arch mbart_large --layernorm-embedding \
        --task translation_from_pretrained_bart \
        --source-lang en_XX --target-lang de_DE \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
        --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
        --lr-scheduler polynomial_decay --lr ${lr} --warmup-updates 2500 --total-num-update $step --max-update $step \
        --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
        --max-tokens 8192 --update-freq 1 \
        --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 11 --no-epoch-checkpoints \
        --seed 222 --log-format simple --log-interval 2 \
        --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
        --restore-file $PRETRAIN \
        --langs $langs \
        --fp16 --local_rank $SLURM_LOCALID \
        --ddp-backend no_c10d \
        --tensorboard-logdir ${checkpoint_dir}/tensorboard > $checkpoint_dir/en_XX-de_DE.log 2>&1 
elif [ ${continue} == 1 ];then
    echo ">> continue fine-tuning..."
    PRETRAIN=${checkpoint_dir}/checkpoint_last.pt
    fairseq-train ${home}/data-in/en_XX-de_DE_trimed \
        --save-dir ${checkpoint_dir} \
        --encoder-normalize-before --decoder-normalize-before \
        --arch mbart_large --layernorm-embedding \
        --task translation_from_pretrained_bart \
        --source-lang en_XX --target-lang de_DE \
        --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
        --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
        --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update $step --max-update $step \
        --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
        --max-tokens 8192 --update-freq 1 \
        --save-interval 1 --save-interval-updates 5000 --keep-interval-updates 50 --no-epoch-checkpoints \
        --seed 222 --log-format simple --log-interval 2 \
        --restore-file $PRETRAIN \
        --langs $langs \
        --fp16 --local_rank $SLURM_LOCALID \
        --ddp-backend no_c10d \
        --tensorboard-logdir ${checkpoint_dir}/tensorboard >> $checkpoint_dir/en_XX-de_DE.log 2>&1 
else
    echo ">> error args"
fi

