ratio=0.1
python -u fuse_with_translation_task.py ende $RI_model $PT_model $fused_model $ratio
wait
python -u convert_model.py --fused_state $fused_model.pt --base_model base_FT_ende_trimed.pt
