CKPT=ckpt
HF=hf
LOG=log
HESS=hess
BASE=Meta-Llama-3-8B-Instruct
RESULT=llama3_8b_6144

# 3 bits
#CODEBOOK=E8P12RVQ3B

# 4 bits
CODEBOOK=E8P12RVQ4B

mkdir -p $CKPT $HF $LOG $HESS

# quantize with finetuning
python -m quantize_finetune_llama.py --save_path $CKPT/$RESULT --codebook $CODEBOOK  --scale_override 0.9 --base_model $BASE  --hessian_path $HESS/$RESULT/ --devset_size 384 --ft_valid_size 128 >> $LOG/$RESULT 2>&1

# convert model to hf format for end to end fine tuning
python -m hfize_llama.py --quantized_path $CKPT/$RESULT --hf_output_path $HF/$RESULT >> $LOG/$RESULT 2>&1

# end to end fine tuning
python -m finetune_e2e_llama.py --base_model $BASE --hf_path $HF/$RESULT --devset_size 384 --ft_valid_size 128 --ft_epochs 8  --ft_bs 4 --ctx_size 8192 --ft_update_freq 2 --ft_train_mode --ckpt_path $CKPT/$RESULT >> $LOG/$RESULT 2>&1

# eval
python -m hfize_llama.py --quantized_path $CKPT/$RESULT --hf_output_path $HF/$RESULT >> $LOG/$RESULT 2>&1 
