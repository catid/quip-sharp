CKPT=ckpt
HF=hf
LOG=log
HESS=hess
BASE=/home/catid/models/Meta-Llama-3-8B-Instruct
#BASE=Meta-Llama-3-8B-Instruct
RESULT=llama3_8b_6144

# 3 bits
#CODEBOOK=E8P12RVQ3B

# 4 bits
CODEBOOK=E8P12RVQ4B

mkdir -p $CKPT $HF $LOG $HESS

echo "quantize_finetune_llama"

# quantize with finetuning
python quantize_finetune_llama.py --save_path $CKPT/$RESULT --codebook $CODEBOOK  --scale_override 0.9 --base_model $BASE  --hessian_path $HESS/$RESULT/ --devset_size 384 --ft_valid_size 128
# >> $LOG/$RESULT 2>&1

echo "hfize_llama"

# convert model to hf format for end to end fine tuning
python hfize_llama.py --quantized_path $CKPT/$RESULT --hf_output_path $HF/$RESULT
# >> $LOG/$RESULT 2>&1
