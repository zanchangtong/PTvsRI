output_dir="./LV_LD"
# $output_dir/*_all.text: total corpus LV score
checkpoint_home=

INPUT="$checkpoint_home/en_XX-de_DE_bpe_from_scratch_4_trimed_dict/en_XX-de_DE_avg5.hyp"
OUTPUT_LV="$output_dir/RI_LV.txt"
OUTPUT_LD="$output_dir/RI_LD.txt"
OUTPUT="$output_dir/RI_all.txt"
OUTPUT_L="$output_dir/RI_length.txt"
NUM_WORKERS=100
LANG="de"
MODEL="./german-gsd-ud-2.4-190531.udpipe"

mkdir -p "$output_dir/"

python ./get_sentence_level_LV_LD_de.py --input_path $INPUT \
	--output_LV $OUTPUT_LV \
	--output_LD $OUTPUT_LD \
	--output_L $OUTPUT_L \
	--output $OUTPUT \
	--num_workers $NUM_WORKERS \
	--lang $LANG \
	--model $MODEL \



INPUT="$checkpoint_home/ende_trimed_0.9_FT_0.1_RI_FT_main/en_XX-de_DE_best.hyp"
OUTPUT_LV="$output_dir/Fusion_LV.txt"
OUTPUT_LD="$output_dir/Fusion_LD.txt"
OUTPUT="$output_dir/Fusion_all.txt"
OUTPUT_L="$output_dir/Fusion_length.txt"
NUM_WORKERS=100
LANG="de"
MODEL="./german-gsd-ud-2.4-190531.udpipe"

mkdir -p "$output_dir/"

python ./get_sentence_level_LV_LD_de.py --input_path $INPUT \
	--output_LV $OUTPUT_LV \
	--output_LD $OUTPUT_LD \
	--output_L $OUTPUT_L \
	--output $OUTPUT \
	--num_workers $NUM_WORKERS \
	--lang $LANG \
	--model $MODEL \


INPUT="$checkpoint_home/trimed_en_XX-de_DE_128X/en_XX-de_DE_best.hyp"
OUTPUT_LV="$output_dir/PT_LV.txt"
OUTPUT_LD="$output_dir/PT_LD.txt"
OUTPUT="$output_dir/PT_all.txt"
OUTPUT_L="$output_dir/PT_length.txt"
NUM_WORKERS=100
LANG="de"
MODEL="./german-gsd-ud-2.4-190531.udpipe"

mkdir -p "$output_dir/"

python ./get_sentence_level_LV_LD_de.py --input_path $INPUT \
	--output_LV $OUTPUT_LV \
	--output_LD $OUTPUT_LD \
	--output_L $OUTPUT_L \
	--output $OUTPUT \
	--num_workers $NUM_WORKERS \
	--lang $LANG \
	--model $MODEL \