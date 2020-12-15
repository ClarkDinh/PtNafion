#!/bin/bash 
#SBATCH --ntasks=16
#SBATCH --output=./output_1.txt
#SBATCH --cpus-per-task=16
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp100_a0.5_singleFalse.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp100_a0.8_singleFalse.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp200_a0.2_singleTrue.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp200_a0.5_singleTrue.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp200_a0.8_singleTrue.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp200_a0.2_singleFalse.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp200_a0.5_singleFalse.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp200_a0.8_singleFalse.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp500_a0.2_singleTrue.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp500_a0.5_singleTrue.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp500_a0.8_singleTrue.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp500_a0.2_singleFalse.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp500_a0.5_singleFalse.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs10_msp500_a0.8_singleFalse.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs50_msp10_a0.2_singleTrue.sh &
srun --ntasks=1 --nodes=1 sh /Users/nguyennguyenduong/Dropbox/Document/2020/Nagoya_ctxafs/code/sh/mcs50_msp10_a0.5_singleTrue.sh &
wait
