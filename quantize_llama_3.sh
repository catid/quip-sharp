CKPT=ckpt
HF=hf
LOG=log
HESS=hess
BASE=Meta-Llama-3-8B-Instruct
RESULT=llama3_8b_6144

mkdir -p $CKPT $HF $LOG $HESS

# generate hessians (takes a while, only use this if there aren't pregenerated hessians)
python hessian_offline_llama.py --batch_size 8 --devset_size 6144 --ctx_size 8192 --base_model $BASE --save_path $HESS/$RESULT

# quantize with finetuning
python -m quantize_llama.quantize_finetune_llama --save_path $CKPT/$RESULT --codebook E8P12  --scale_override 0.9 --base_model $BASE  --hessian_path $HESS/$RESULT/ --devset_size 384 --ft_valid_size 128 >> $LOG/$RESULT 2>&1

# convert model to hf format for end to end fine tuning
python -m quantize_llama.hfize_llama --quantized_path $CKPT/$RESULT --hf_output_path $HF/$RESULT >> $LOG/$RESULT 2>&1

# end to end fine tuning
python -m quantize_llama.finetune_e2e_llama --base_model $BASE --hf_path $HF/$RESULT --devset_size 384 --ft_valid_size 128 --ft_epochs 8  --ft_bs 4 --ctx_size 8192 --ft_update_freq 2 --ft_train_mode --ckpt_path $CKPT/$RESULT >> $LOG/$RESULT 2>&1

# eval
python -m quantize_llama.hfize_llama --quantized_path $CKPT/$RESULT --hf_output_path $HF/$RESULT >> $LOG/$RESULT 2>&1 
