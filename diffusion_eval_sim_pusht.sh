#!/bin/bash
#SBATCH --gpus=1
#SBATCH --mem-per-gpu=30G
#SBATCH --cpus-per-gpu=8
#SBATCH --time=20:00:00
#SBATCH --account=mp-account
##SBATCH --account=leonmkim-account
#SBATCH --partition=batch
##SBATCH --partition=posa-compute
##SBATCH --partition=dineshj-compute
#SBATCH --qos=normal
##SBATCH --qos=mp-med
##SBATCH --qos=dj-high
##SBATCH --qos=dj-med
#SBATCH --signal=SIGUSR1@90 # this is for pytorch-lightning
#SBATCH --requeue
#SBATCH --array=0-20
#SBATCH --job-name=dp_eval_sim
#SBATCH --output=test_slurm_output/slurm-%A_%a.out
#SBATCH --error=test_slurm_error/slurm-%A_%a.err
#SBATCH --exclude=mp-2080ti-0.grasp.maas,dj-a40-0.grasp.maas,kd-a40-0.grasp.maas

LOCAL_RUN=true

export HYDRA_FULL_ERROR=1

# if not local run
if [ "$LOCAL_RUN" = false ]; then
    # if not local run, then we're on a cluster
    # load conda
    eval "$(conda shell.bash hook)"
    # activate the conda environment
    conda activate just_lerobot
    OUTPUT_ROOT_DIR="/mnt/grasp_high_usage/leonmkim/lerobot_pusht_evals"
    CHECKPOINT_ROOT_DIR="/mnt/grasp_high_usage/leonmkim/contact_estimation/lerobot/pusht/diffusion_policy/train_with_added_metrics"
    ROOT_DIR="/mnt/kostas-graid/datasets/extrinsic_contact_data"
else
    OUTPUT_ROOT_DIR="/mnt/crucialSSD/lerobot_pusht_evals"
    CHECKPOINT_ROOT_DIR="/mnt/crucialSSD/lerobot/pusht/diffusion_policy/train_with_added_metrics"
    ROOT_DIR="/mnt/crucialSSD/datasetsSSD/fish_datasets/simulated/teleop"
fi

DETERMINISTIC_ACTIONS=true
DETERMINISTIC_ACTIONS_SEED=0

USE_EXISTING_EVAL_SEEDS=false
# NUM_SEEN_EVAL=100
# NUM_UNSEEN_EVAL=100
NUM_SEEN_EVAL=0
NUM_UNSEEN_EVAL=50
UNIV_UNSEEN_ENV_SEED_START=2000000

RUN_ID_ARRAY_LENGTH=${#RUN_ID_ARRAY[@]}
# generate the include groups list by repeating the same group for each run id
# INCLUDE_GROUPS_LIST_ARRAY=()
# for ((i=0;i<RUN_ID_ARRAY_LENGTH;i++)); do
#     # INCLUDE_GROUPS_LIST_ARRAY+=("all")
#     INCLUDE_GROUPS_LIST_ARRAY+=("['greece_twodim']")
# done

# RUN_ID_ARRAY=("53844_0" "53823_0" "53848_0" "53846_0" "53820_0" "54232_0" "54234_0" "53847_0" "53842_0" "53825_0" "53091_0" "54235_0" "54233_0" "53854_0" "53849_0" "53853_0" "53850_0" "53843_0" "53104_0" "53096_0" "53095_0")
RUN_ID_ARRAY=("VrZathNUl0Jle")
# EPOCH_ARRAY=("last")
EPOCH="last"

AGENT="diffusion_pusht"
SUITE="frankagym"
TASK="insertion"

ARRAY_LENGTH=${#RUN_ID_ARRAY[@]}

WANDB__SERVICE_WAIT=1200

# for RUN_ID in "${RUN_ID_ARRAY[@]}"; do
if [ "$LOCAL_RUN" = true ]; then
    for ((i=0;i<ARRAY_LENGTH;i++)); do
        echo "RUN_ID: ${RUN_ID_ARRAY[i]}"
        python lerobot/scripts/eval_policy_sim_pusht.py \
        eval_cfg="sim_pusht" \
        env="lerobot_pusht" \
        agent=${AGENT} \
        suite=${SUITE} \
        suite/frankagym_task@_global_=${TASK} \
        eval_cfg.use_existing_eval_seeds=${USE_EXISTING_EVAL_SEEDS} \
        eval_cfg.num_seen_eval_envs=${NUM_SEEN_EVAL} \
        eval_cfg.num_unseen_eval_envs=${NUM_UNSEEN_EVAL} \
        eval_cfg.wandb_run_id="'${RUN_ID_ARRAY[i]}'" \
        eval_cfg.checkpoint_epoch=${EPOCH} \
        eval_cfg.checkpoint_root_dir=${CHECKPOINT_ROOT_DIR} \
        eval_cfg.output_root_dir=${OUTPUT_ROOT_DIR} \
        eval_cfg.root_dir=${ROOT_DIR} \
        eval_cfg.universal_unseen_env_seed_start=${UNIV_UNSEEN_ENV_SEED_START} \
        agent.config.deterministic_actions=${DETERMINISTIC_ACTIONS} \
        agent.config.deterministic_actions_seed=${DETERMINISTIC_ACTIONS_SEED}
    done
else
    srun python lerobot/scripts/eval_policy_sim_pusht.py \
    eval_cfg="sim_pusht" \
    env="lerobot_pusht" \
    agent=${AGENT} \
    suite=${SUITE} \
    suite/frankagym_task@_global_=${TASK} \
    eval_cfg.use_existing_eval_seeds=${USE_EXISTING_EVAL_SEEDS} \
    eval_cfg.num_seen_eval_envs=${NUM_SEEN_EVAL} \
    eval_cfg.num_unseen_eval_envs=${NUM_UNSEEN_EVAL} \
    eval_cfg.wandb_run_id="'${RUN_ID_ARRAY[$SLURM_ARRAY_TASK_ID]}'" \
    eval_cfg.checkpoint_epoch=${EPOCH} \
    eval_cfg.checkpoint_root_dir=${CHECKPOINT_ROOT_DIR} \
    eval_cfg.output_root_dir=${OUTPUT_ROOT_DIR} \
    eval_cfg.root_dir=${ROOT_DIR} \
    eval_cfg.universal_unseen_env_seed_start=${UNIV_UNSEEN_ENV_SEED_START} \
    agent.config.deterministic_actions=${DETERMINISTIC_ACTIONS} \
    agent.config.deterministic_actions_seed=${DETERMINISTIC_ACTIONS_SEED}
fi

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