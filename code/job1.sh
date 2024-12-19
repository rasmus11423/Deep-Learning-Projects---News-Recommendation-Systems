#!/bin/bash
# embedded options to bsub - start with #BSUB
# -- our name ---
#BSUB -J ActiveLearinig
# -- choose queue --
#BSUB -q gpua100
# -- specify that we need 4GB of memory per core/slot --
# so when asking for 4 cores, we are really asking for 4*4GB=16GB of memory 
# for this job. 
#BSUB -R "rusage[mem=4GB]"
# -- Notify me by email when execution begins --
#BSUB -B
# -- Notify me by email when execution ends   --
#BSUB -N
# -- email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u s215160@dtu.dk
# -- Output File --
#BSUB -o 4GB_4_mem_frac_01%J.out
# -- Error File --
#BSUB -e 4GB_4_mem_frac_01_%J.err
# -- estimated wall clock time (execution time): hh:mm -- 
#BSUB -W 4:00 
# -- Number of cores requested -- 
#BSUB -n 4
# -- Specify the distribution of the cores: on a single node --
#BSUB -R "span[hosts=1]"
# -- end of LSF options --

#nvidia-smi
# Your VENV path
module load python3/3.11.9
source "/zhome/06/9/168972/Deep_Learning/Deep-Learning-Projects---News-Recommendation-Systems/Deep_venv/bin/activate"

export NEPTUNE_API_TOKEN="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3ZmM0ZDRjMC00MDU4LTRmNTctYjViNC04NjA3ZWY4NzZmMGUifQ=="

# here call torchrun
#torchrun --standalone --nproc_per_node=1 
python "/zhome/06/9/168972/Deep_Learning/Deep-Learning-Projects---News-Recommendation-Systems/code/modelfoo_pytorch.py" --dataset "ebnerd_small"