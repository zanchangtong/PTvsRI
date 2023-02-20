# set -x
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
home=
checkpoint_home=
MODEL=$home/mbart.cc25/sentence.bpe.model
spm_decode=$home/Tools/sentencepiece/build/src/spm_decode
export LC_ALL=C.UTF-8 
language_pair=en_XX-de_DE
TOKENIZER=sacremoses

data_dir=./multi_refs_ende
# set -x 

generate_raw(){
  source_file=$1
  checkpoint_dir=$2
  checkpoint_name=$3
  cuda_id=$4
  source_lang=en_XX
  target_lang=de_DE
  if [ ! -d $checkpoint_dir/multi-refs ];then
    mkdir $checkpoint_dir/multi-refs
  fi
  cat $data_dir/${source_file} | CUDA_VISIBLE_DEVICES=$cuda_id fairseq-interactive $checkpoint_dir \
      --path $checkpoint_dir/$checkpoint_name \
      --task translation_from_pretrained_bart \
      --bpe "sentencepiece" \
      --sentencepiece-model ${MODEL} \
      --buffer-size 1024 \
      --batch-size 60 -s en_XX -t de_DE --langs $langs \
      --beam 5 \
      --fp16 > $checkpoint_dir/multi-refs/${checkpoint_name}_multi_refs.gen_log 2>&1 
}

generate_raw_ri(){
  source_file=$1
  checkpoint_dir=$2
  checkpoint_name=$3
  cuda_id=$4
  source_lang=en_XX
  target_lang=de_DE
  if [ ! -d $checkpoint_dir/multi-refs ];then
    mkdir $checkpoint_dir/multi-refs
  fi
  cat $data_dir/${source_file} | CUDA_VISIBLE_DEVICES=$cuda_id fairseq-interactive $checkpoint_dir \
      --path $checkpoint_dir/$checkpoint_name \
      --task translation \
      --bpe "sentencepiece" \
      --sentencepiece-model ${MODEL} \
      --buffer-size 1024 \
      --batch-size 60 --source-lang en_XX --target-lang de_DE \
      --beam 5 \
      --fp16 > $checkpoint_dir/multi-refs/${checkpoint_name}_multi_refs.gen_log 2>&1 
}


record_sacrebleu(){
    source_file=$1
    checkpoint_dir=$2
    checkpoint_name=$3
    source_lang=en_XX
    target_lang=de_DE
    cat $checkpoint_dir/multi-refs/${checkpoint_name}_multi_refs.gen_log | grep -P "^D-" | cut -f3 > $checkpoint_dir/multi-refs/${checkpoint_name}_multi_refs.hyp

    echo ">> ${checkpoint_name}_${source_lang}_${target_lang}_tgt_${reference_file}: detailed sacrebleu" 
    sacrebleu $data_dir/T $data_dir/R1 $data_dir/R2 $data_dir/R3 $data_dir/R4 $data_dir/R5 $data_dir/R6 $data_dir/R7 $data_dir/R8 $data_dir/R9 $data_dir/R1 \
    -i $checkpoint_dir/multi-refs/${checkpoint_name}_multi_refs.hyp -l "${source_lang: 0: 2}-${target_lang: 0: 2}" 
    
}

# PT model
printf "\n\n"
checkpoint_dir=$checkpoint_home/trimed_en_XX-de_DE_128X
echo ">> checkpoint dir is: $checkpoint_dir"
generate_raw S $checkpoint_dir checkpoint_best.pt 0
record_sacrebleu S $checkpoint_dir checkpoint_best.pt 

# RI model
printf "\n\n"
checkpoint_dir=$checkpoint_home/en_XX-de_DE_bpe_from_scratch_4_trimed_dict
echo ">> checkpoint dir is: $checkpoint_dir"
generate_raw_ri S $checkpoint_dir checkpoint_avg5.pt 0
record_sacrebleu S $checkpoint_dir checkpoint_avg5.pt 

# fusion model
printf "\n\n"
checkpoint_dir=$checkpoint_home/ende_trimed_0.9_FT_0.1_RI_FT_main
echo ">> checkpoint dir is: $checkpoint_dir"
generate_raw S $checkpoint_dir checkpoint_best.pt 0
record_sacrebleu S $checkpoint_dir checkpoint_best.pt 



