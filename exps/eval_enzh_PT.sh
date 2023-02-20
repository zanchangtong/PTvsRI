set -x
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
home=
MODEL=$home/mbart.cc25/sentence.bpe.model
spm_decode=$home/Tools/sentencepiece/build/src/spm_decode
export LC_ALL=C.UTF-8 
cuda_id=${2:-0}
data_bin=$home/data-in/en_XX-zh_CN_small
language_pair=en_XX-zh_CN
TOKENIZER=sacremoses

generate_best(){

  TGT=$3
  TEST=test
  SRC=$2
  checkpoint_dir=$1

  echo "SRC: $SRC" 
  echo ">> $language_pair evluate: best ..."

  echo "cuda:$cuda_id"
  CUDA_VISIBLE_DEVICES=$cuda_id fairseq-generate ${data_bin} \
    --path $checkpoint_dir/checkpoint_best.pt \
    --task translation_from_pretrained_bart \
    --gen-subset test \
    -t ${TGT} -s ${SRC} \
    --batch-size 60 --langs $langs > $checkpoint_dir/${language_pair}_best
}

generate_last(){
  TGT=$3
  TEST=test
  SRC=$2
  checkpoint_dir=$1

  echo ">> $language_pair evluate: last ..."

  echo "cuda:$cuda_id"
  CUDA_VISIBLE_DEVICES=$cuda_id fairseq-generate ${data_bin} \
    --path $checkpoint_dir/checkpoint_last.pt \
    --task translation_from_pretrained_bart \
    --gen-subset test \
    -t ${TGT} -s ${SRC} \
    --batch-size 60 --langs ${langs} > $checkpoint_dir/${language_pair}_last
}

generate_avg5(){
  TGT=$3
  TEST=test
  SRC=$2
  checkpoint_dir=$1

  echo ">> $language_pair evluate: avg5 ..."

  python ${home}/code/fairseq-official/scripts/average_checkpoints.py \
    --inputs $checkpoint_dir \
    --num-update-checkpoints 5 \
    --output $checkpoint_dir/checkpoint_avg5.pt

  CUDA_VISIBLE_DEVICES=$cuda_id fairseq-generate ${data_bin} \
    --path $checkpoint_dir/checkpoint_avg5.pt \
    --task translation_from_pretrained_bart \
    --gen-subset test \
    -t ${TGT} -s ${SRC} \
    --batch-size 60 --langs $langs > $checkpoint_dir/${language_pair}_avg5
}

# Example: bash eval_enzh_PT.sh trimed_en_XX-zh_CN_PT
post_=$1 
checkpoint_dir=
TEST=test
SRC=en_XX
TGT=zh_CN
exp=$checkpoint_dir/$post_
MODEL=$home/mbart.cc25/sentence.bpe.model

generate_best ${exp} $SRC $TGT &
generate_last ${exp} $SRC $TGT &
generate_avg5 ${exp} $SRC $TGT &

wait 
echo " " >> en-zh.out
echo ">>${exp} best_BLEU:" >> en-zh.out
cat ${exp}/${language_pair}_best | grep -P "^H" |sort -V |cut -f 3- | ${spm_decode} --model=${MODEL} | sed 's/\[zh_CN\]//g' > ${exp}/${language_pair}_best.hyp
cat ${exp}/${language_pair}_best | grep -P "^T" |sort -V |cut -f 2- | ${spm_decode} --model=${MODEL} | sed 's/\[zh_CN\]//g' > ${exp}/${language_pair}_best.ref
sacrebleu -i ${exp}/${language_pair}_best.hyp -t wmt17 --language-pair "${SRC: 0: 2}-${TGT: 0: 2}" >> en-zh.out
wait
echo ">>${exp} last_BLEU:" >> en-zh.out
cat ${exp}/${language_pair}_last | grep -P "^H" |sort -V |cut -f 3- | ${spm_decode} --model=${MODEL} | sed 's/\[zh_CN\]//g' > ${exp}/${language_pair}_last.hyp
cat ${exp}/${language_pair}_last | grep -P "^T" |sort -V |cut -f 2- | ${spm_decode} --model=${MODEL} | sed 's/\[zh_CN\]//g' > ${exp}/${language_pair}_last.ref
sacrebleu -i ${exp}/${language_pair}_last.hyp -t wmt17 --language-pair "${SRC: 0: 2}-${TGT: 0: 2}" >> en-zh.out
wait
echo ">>${exp} avg5_BLEU:" >> en-zh.out
cat ${exp}/${language_pair}_avg5 | grep -P "^H" |sort -V |cut -f 3- | ${spm_decode} --model=${MODEL} | sed 's/\[zh_CN\]//g' > ${exp}/${language_pair}_avg5.hyp
cat ${exp}/${language_pair}_avg5 | grep -P "^T" |sort -V |cut -f 2- | ${spm_decode} --model=${MODEL} | sed 's/\[zh_CN\]//g' > ${exp}/${language_pair}_avg5.ref
sacrebleu -i ${exp}/${language_pair}_avg5.hyp -t wmt17 --language-pair "${SRC: 0: 2}-${TGT: 0: 2}" >> en-zh.out
wait


