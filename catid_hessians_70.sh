CKPT=ckpt
HF=hf
LOG=log
HESS=hess
BASE=Meta-Llama-3-70B-Instruct
RESULT=llama3_70b_6144

# 3 bits
CODEBOOK=E8P12RVQ3B

# 4 bits
#CODEBOOK=E8P12RVQ4B

mkdir -p $CKPT $HF $LOG $HESS

# generate hessians (takes a while, only use this if there aren't pregenerated hessians)
python hessian_offline_llama.py --batch_size 4 --devset_size 6144 --ctx_size 8192 --base_model $BASE --save_path $HESS/$RESULT
