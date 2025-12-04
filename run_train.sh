#BSUB -J lstm
#BSUB -o lstm%J.out
#BSUB -q gpul40s 
#BSUB -n 8
#BSUB -R "rusage[mem=8GB]"
#BSUB -gpu "num=1:mode=exclusive_process"
#BSUB -W 24:00

python ./src/main.py --isbsub
python ./src/plot_prediction.py --isbsub
python ./src/loss_pair_visualization.py --isbsub
