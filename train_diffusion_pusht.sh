#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=30G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=12:00:00
##SBATCH --account=mp-account
#SBATCH --account=leonmkim-account
##SBATCH --account=bdlk-delta-gpu
##SBATCH --partition=batch
##SBATCH --partition=posa-compute
#SBATCH --partition=dineshj-compute
##SBATCH --partition=gpuA100x4
##SBATCH --partition=gpuA100x8
##SBATCH --partition=gpuA40x4
##SBATCH --qos=normal
##SBATCH --qos=mp-med
##SBATCH --qos=dj-high
#SBATCH --qos=dj-med
#SBATCH --signal=SIGUSR1@90 # this is for pytorch-lightning
#SBATCH --requeue
#SBATCH --array=0
#SBATCH --job-name=lerobot
#SBATCH --output=slurm_output/slurm-%A_%a.out
#SBATCH --error=slurm_error/slurm-%A_%a.err
#SBATCH --exclude=dj-2080ti-0.grasp.maas,kd-2080ti-1.grasp.maas,mp-2080ti-0.grasp.maas,kd-2080ti-2.grasp.maas,kd-2080ti-3.grasp.maas,kd-2080ti-4.grasp.maas
##SBATCH --exclude=mp-2080ti-0.grasp.maas,dj-a40-0.grasp.maas,kd-a40-0.grasp.maas

LOCAL_RUN=false
ON_NCSA_CLUSTER=false
DEBUG_MODE=false

if [ "$LOCAL_RUN" = false ]; then
    # if not local run, then we're on a cluster
    # load conda
    eval "$(conda shell.bash hook)"
    # activate the conda environment
    # conda activate lerobot
    if [ "$ON_NCSA_CLUSTER" = true ]; then
        conda activate lerobot
        OUTPUT_DIR="/work/hdd/bdlk/leonmkim/FISH"
    else
        conda activate just_lerobot
        OUTPUT_DIR="/mnt/grasp_high_usage/leonmkim/contact_estimation/lerobot/pusht/diffusion_policy/train_with_added_metrics"
    fi
    # SNAPSHOT_ROOT_DIR="/mnt/kostas-graid/datasets/extrinsic_contact_data/FISH"
    # SEED=${SLURM_ARRAY_TASK_ID}
else
    # ROOT_DIR="/home/serialexperimentsleon/fish_leon"
    OUTPUT_DIR="/mnt/crucialSSD/lerobot/pusht/diffusion_policy/train_with_added_metrics"
fi

if [ "$DEBUG_MODE" = true ]; then
    SAVE_FREQ=100
    LOG_FREQ=50
    EVAL_FREQ=100
    INFERENCE_ON_TRAINING_FREQ=100
    INFERENCE_ON_VALIDATION_FREQ=100
    TRAIN_STEPS=500
    TRAIN_EPISODES="[1,2]"
    VALID_EPISODES="[3,4]"
else
    SAVE_FREQ=10000
    LOG_FREQ=200
    EVAL_FREQ=10000
    INFERENCE_ON_TRAINING_FREQ=5000
    INFERENCE_ON_VALIDATION_FREQ=5000
    TRAIN_STEPS=200000
    TRAIN_EPISODES="[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,117,118,119,120,121,122,123,124,125,126,127,128,130,131,132,133,134,135,136,137,138,139,140,141,142,143,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205]"
    VALID_EPISODES="[129,116,144,193,6]"
    # TRAIN_EPISODES="[0,1,2,3,4,5,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,107,108,109,110,111,112,113,114,115,117,118,119,120,121,122,123,124,125,126,127,128,130,131,132,133,134,135,136,137,138,139,140,141,142,143,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,194,195,196,197,198,199,200,201,202,203,204,205]"
    # VALID_EPISODES="[129,116,144,193,6,25,177,54,106,90]"
fi

SEED=100000
BATCH_SIZE=64

# DROP_N_LAST_FRAMES=7 # default
DROP_N_LAST_FRAMES=0

DOWN_DIMS="[512,1024,2048]" #default
# DOWN_DIMS="[256,512,1024]" # small
# DOWN_DIMS="[64,128,256]" # small

# NOISE_SCHEDULER_TYPE="DDPM" # default
NOISE_SCHEDULER_TYPE="DDIM" 

# NUM_TRAIN_STEPS=100 #default
NUM_TRAIN_STEPS=50

# OPTIMIZER="adam" # default
OPTIMIZER="adamw" 

REQUEUE_RUN_ID=""
NOTES="_"
RUN_ID="null"
RESUME=false
CONFIG_PATH_FLAG=""
if [ "$LOCAL_RUN" = false ]; then
    # if REQUEUE_RUN_ID is not empty then we're requeuing
    if [ -z "$SLURM_ARRAY_JOB_ID" ]; then
        CURRENT_SLURM_JOB_ID="${SLURM_JOB_ID}"
    else
        CURRENT_SLURM_JOB_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    fi
    if [ -z "$REQUEUE_RUN_ID" ]; then
        # if we're not requeuing then we're running a new job
        # if we're running an array job then use the array job id and task id
        # otherwise just use the job id
        RUN_ID="${CURRENT_SLURM_JOB_ID}"
    else
        # if we're requeuing then we're running an array job
        # so use the requeue run id and the task id
        RUN_ID="${REQUEUE_RUN_ID}"
        NOTES="${NOTES}req_${CURRENT_SLURM_JOB_ID}"
        RESUME=true
    fi
    # if [ -z "$SLURM_ARRAY_JOB_ID" ]; then
    #     RUN_ID="${SLURM_JOB_ID}"
    # else
    #     RUN_ID="${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    # fi
    NOTES="${RUN_ID}${NOTES}"

    # if SLURM_RESTART_COUNT greater than 0, then this job has been requeued
    if [[ ${SLURM_RESTART_COUNT} -gt 0 ]]; then
        # auto-requeued runs will have same job id
        NOTES="${NOTES}restarted_${SLURM_RESTART_COUNT}"
    fi

    # also save a copy of this script to slurm_scripts/
    cp $0 "slurm_scripts/${CURRENT_SLURM_JOB_ID}.bash"
else
    # if requeue run id is not empty, then assign it to run id
    if [ -z "$REQUEUE_RUN_ID" ]; then
        RUN_ID=$(tr -dc A-Za-z0-9 </dev/urandom | head -c 13; echo)
    else
        RUN_ID="${REQUEUE_RUN_ID}"
    fi
fi


WANDB_ENTITY="serialexperimentsleon"
WANDB_PROJECT="lerobot"
OUTPUT_DIR="${OUTPUT_DIR}/${RUN_ID}"
# append slurm job id to output dir

# if resuming
if [ "$RESUME" = true ]; then
    CONFIG_PATH_FLAG="--config_path=${OUTPUT_DIR}/checkpoints/last/pretrained_model/train_config.json"
fi

WANDB__SERVICE_WAIT=1200

# python lerobot/scripts/train_dp_with_added_metrics.py \
srun python lerobot/scripts/train_dp_with_added_metrics.py \
--resume=${RESUME} \
--output_dir=${OUTPUT_DIR} \
--policy.type=diffusion \
--seed=${SEED} \
--dataset.repo_id=lerobot/pusht \
--dataset.episodes=${TRAIN_EPISODES} \
--valid_dataset.repo_id=lerobot/pusht \
--valid_dataset.episodes=${VALID_EPISODES} \
--env.type=pusht \
--batch_size=${BATCH_SIZE} \
--policy.drop_n_last_frames=${DROP_N_LAST_FRAMES} \
--policy.down_dims=${DOWN_DIMS} \
--policy.noise_scheduler_type=${NOISE_SCHEDULER_TYPE} \
--policy.num_train_timesteps=${NUM_TRAIN_STEPS} \
--policy.optimizer=${OPTIMIZER} \
--steps=${TRAIN_STEPS} \
--wandb.enable=true \
--wandb.entity=${WANDB_ENTITY} \
--wandb.project=${WANDB_PROJECT} \
--wandb.notes=${NOTES} \
--wandb.run_id=${RUN_ID} \
--log_freq=${LOG_FREQ} \
--save_freq=${SAVE_FREQ} \
--eval_freq=${EVAL_FREQ} \
--inference_on_training_freq=${INFERENCE_ON_TRAINING_FREQ} \
--inference_on_validation_freq=${INFERENCE_ON_VALIDATION_FREQ} \
--wandb.disable_artifact=true \
${CONFIG_PATH_FLAG}

PYTHON_EXIT_CODE=$?
echo "PYTHON_EXIT_CODE: ${PYTHON_EXIT_CODE}"

# manually requeue the job if exit code 3
if [ "$LOCAL_RUN" = false ]; then
    if [ $PYTHON_EXIT_CODE -eq 3 ]; then
        echo "Requeueing job"
        # requeue the job
        scontrol requeue ${CURRENT_SLURM_JOB_ID}
    fi
fi
exit $PYTHON_EXIT_CODE