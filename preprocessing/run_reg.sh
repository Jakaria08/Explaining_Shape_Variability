salloc --gres=gpu:v100l:1 --time=0-3:0:0 --cpus-per-task=8 --mem-per-cpu=6GB
module load CCconfig gentoo/2020 gcccore/.9.3.0 imkl/2020.1.217 intel/2020.1.217 StdEnv/2020 mii/1.1.2 libffi/3.3 python/3.8.10 cudacore/.11.0.2 cuda/11.0 gdrcopy/2.1 ucx/1.8.0 libfabric/1.10.1 openmpi/4.0.3 cmake/3.18.4
source ~/ENV_reg/bin/activate
python reg.py