set -euo pipefail

task_name=$1
traj_num=$2
date=$3
gpu_idx=$4

traj_num_per_robot=$(($traj_num / 5))
num_procs=$((traj_num_per_robot / 4))

h5_file_name=merged_data_${traj_num}_${date}.h5
zarr_file_name=ManiSkill_${task_name}_${date}.zarr.zip

export CUDA_VISIBLE_DEVICES=${gpu_idx}

cd ~/ManiSkill/
python multi_robot_data_collection.py -e ${task_name}-v1 -f ${h5_file_name} -n ${traj_num_per_robot} -c pd_ee_pose --save-video --num-procs ${num_procs}
cd ~/universal_manipulation_interface/
python convert_hdf5_to_umi_zarr.py -i /home/wshf/ManiSkill/demos/${task_name}-v1/motionplanning/${h5_file_name} -o data/${zarr_file_name}
python train.py --config-name=train_diffusion_unet_timm_maniskill_workspace task.dataset_path=data/${zarr_file_name} env_name=${task_name}
