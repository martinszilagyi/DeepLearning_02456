#BSUB -J preprocessing
#BSUB -o preprocessing%J.out
#BSUB -q gpul40s 
#BSUB -n 4
#BSUB -R "rusage[mem=4GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

python ./src/ais_to_parquet.py
python ./src/normalize_parquet_data.py
python ./src/final_clean.py