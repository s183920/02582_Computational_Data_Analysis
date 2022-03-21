#!/bin/sh


# create dir for logs
mkdir -p "logs"

### General options
### –- specify queue --
#BSUB -q hpc
### -- set the job Name --
#BSUB -J CDA
### -- ask for number of cores (default: 1) -- 
#BSUB -n 1 
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now 
#BSUB -W 4:00 
### -- request 5GB of system-memory --
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
##BSUB -u s183920@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion-- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o logs/CDA_%J.out 
#BSUB -e logs/CDA_%J.err 
### -- end of LSF options --

# activate env
source CDA-case-env/bin/activate

# load additional modules
# module load cuda/11.4

# run scripts
python model_hpc.py