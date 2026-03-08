# Examples:
# gaiban-var: bash scripts/run_aiws.sh aiws_sub "GenPose2/data/AIWS/000/001_color.png" "GenPose2/data/AIWS/000/001_depth.exr" 0
# gaiban: bash scripts/run_aiws.sh aiws_sub "GenPose2/data/AIWS/G90/1_color.png" "GenPose2/data/AIWS/G90/1_depth.exr" 0
# fangguan: bash scripts/run_aiws.sh aiws_sub "GenPose2/data/AIWS/F120/1_color.png" "GenPose2/data/AIWS/F120/1_depth.exr" 0 
# labakou: bash scripts/run_aiws.sh aiws_sub "GenPose2/data/AIWS/L75/1_color.png" "GenPose2/data/AIWS/L75/1_depth.exr" 0  
# Hxinggang: bash scripts/run_aiws.sh aiws_sub "GenPose2/data/AIWS/Hbeam/1_color.png" "GenPose2/data/AIWS/Hbeam/1_depth.exr" 0  
#
# Optional image overrides:
#   export AIWS_YOLO_IMAGE=yolov11-seg:infer
#   export AIWS_GENPOSE2_IMAGE=genpose2-env:test
#   export AIWS_FOUNDATIONPOSE_IMAGE=foundationpose-env:test

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

DEBUG=True

config_name=${1}
rgb_path=${2}
depth_path=${3}
gpu_id=${4}
echo -e "\033[33mgpu id (to use): ${gpu_id}\033[0m"

export HYDRA_FULL_ERROR=1 
export CUDA_VISIBLE_DEVICES=${gpu_id}
conda run -n aiws python workspace.py --config-name=${config_name}.yaml \
                                      debug="$DEBUG" \
                                      rgb_path="$rgb_path" \
                                      depth_path="$depth_path"
                            



                                
