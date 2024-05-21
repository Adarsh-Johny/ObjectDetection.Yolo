#!/bin/bash -l

#SBATCH --job-name=Project_Viscando

# Resource Allocation

# Define, how long the job will run in real time. This is a hard cap meaning
# that if the job runs longer than what is written here, it will be
# force-stopped by the server. If you make the expected time too long, it will
# take longer for the job to start. Here, we say the job will take 20 minutes
#              d-hh:mm:ss
SBATCH --time=0-02:00:00
# Define resources to use for the defined job. Resources, which are not defined
# will not be provided.

# For simplicity, keep the number of tasks to one
SBATCH --ntasks 1 
# Select number of required GPUs (maximum 1)
SBATCH --gres=gpu:1
# Select number of required CPUs per task (maximum 16)
SBATCH --cpus-per-task 16

# you may not place bash commands before the last SBATCH directive

echo "Now processing task id:: ${SLURM_JOB_ID} on ${SLURMD_NODENAME}"
mkdir "log_${SLURM_JOB_ID}"
python yolov5-traffic-monitoring/train.py --img 300 --batch 16 --epochs 100 --data depth_images.yaml --cfg yolov5-traffic-monitoring/models/custom_yolov5s.yaml --weights yolov5s.pt --workers 4 --name kitti_yolov5s > output_${SLURM_JOB_ID}.txt

echo "finished task with id:: ${SLURM_JOB_ID}"
# happy end
exit 0
