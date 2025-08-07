# UMI-ARX Cup Task Deployment Tutorial

## Hardware

### "The UMI cup"

Before running the cup arrangement policy on other unseen cups, we suggest testing a standard cup (a 3d model that has similar shape to the ones in the collected data). Although this 3d printed cup is also unseen in the training data, it has the closest shape compared to a random cup.

**Onshape Model Link**: [Onshape](https://cad.onshape.com/documents/...)

#### 3D Printing Config
- **Material**: PLA, 100% infill, tree support
- **Color**: Sky blue color is recommended; other colors should work as well (white, black, grey are tested)

### ARX Robot Arm
To purchase ARX robot arms, please visit their official website: https://arx-x.com/

## Software

### Diffusion Policy Setup

Clone the repository:
```bash
git clone git@github.com:real-stanford/detached-umi-policy.git
cd detached-umi-policy
mkdir data && cd data && mkdir models && mkdir experiments
```

Download checkpoint and put it into `data/models`

Install python environment (recommend using mamba to create environments. Usage is the same as conda):
```bash
mamba env create -f conda_environment.yaml
conda activate umi
```

Test whether diffusion policy is successfully installed:
```bash
python detached_policy_inference.py -i data/models/cup_wild_vit_l_1img.ckpt
```

After `PolicyInferenceNode` is listening on `0.0.0.0:8766`, the policy inference process is successfully set up. Keep it running in the background when running umi code.

#### Troubleshooting
If you have trouble importing huggingface:
```
ImportError: Error loading 'diffusion_policy.workspace.train_diffusion_unet_image_workspace.TrainDiffusionUnetImageWorkspace':
ImportError("cannot import name 'cached_download' from 'huggingface_hub' (/data/yihuai/miniforge3/envs/umi/lib/python3.9/site-packages/huggingface_hub/__init__.py)")
```

Please open `$CONDA_PREFIX/lib/python3.9/site-packages/diffusers/utils/dynamic_modules_utils.py` and remove `cached_download` in:
```python
from huggingface_hub import HfFolder, cached_download, hf_hub_download, model_info
```

### ARX5 Robot Arm Setup

Clone the repository:
```bash
git clone git@github.com:real-stanford/arx5-sdk.git
cd arx5-sdk
```

Install conda environment for sdk compilation (python 3.10 is tested, while other versions may work as well):
```bash
mamba env create -f conda_environments/py310_environment.yaml
conda activate arx-py310
```

Compile sdk (if failed, remove the build directory completely and create a new one):
```bash
mkdir build && cd build
cmake .. && make -j
```

Setup robot arm connection and spacemouse service following instructions in README of arx5-sdk repository.

⚠️ **Important**: When turning on the ARX5 arm, make sure the gripper is fully closed.

Test spacemouse cartesian space control. After modifying the ARX robot arm, `YOUR_MODEL` should be either `X5_umi` or `L5_umi`:
```bash
cd python
conda activate arx-py310
python examples/spacemouse_teleop.py YOUR_MODEL YOUR_CAN_INTERFACE
```

Calibrate gripper (after installing the fin-ray gripper fingers):
```bash
python examples/calibrate.py YOUR_MODEL YOUR_CAN_INTERFACE
```

Turn on server for arx5 (under arx-py310 environment; cd to arx5-sdk/python):
```bash
python communication/zmq_server.py YOUR_MODEL YOUR_CAN_INTERFACE
```

This server will keep running in the background by default on `0.0.0.0:8765`. Robot arm will set to home pose then passive if no commands are sent to the server in 60 seconds to avoid overheating.

### UMI Minimal Deployment for ARX Robot Arm

Clone the github repository:
```bash
git clone git@github.com:real-stanford/umi-arx.git
```

Conda environment:
```bash
mamba env create -f umi_arx_environment.yaml
mamba activate umi-arx
mkdir -p data/experiments
```

Check camera connection: there should be a blue circle with an hdmi icon on the front screen of gopro.

#### Run deployment code

First make sure the policy inference node and arx5 server are already turned on.

After the elgato capture card is connected, run the following line:
```bash
sudo chmod -R 777 /dev/bus/usb
```

Then run:
```bash
python scripts/eval_arx5.py -i /home/detached-umi-policy/data/models/cup_wild_vit_l_1img.ckpt -o data/experiments/DATE --no_mirror
```

**Setup Instructions**:
1. Use spacemouse to set the gripper to an initial pose. The initial pose should be roughly **15°** over the table.
2. Press `c` to start policy inference, `s` to stop. 
3. ⚠️ **Stop the policy immediately after the cup is put on the dish**, otherwise arx arm may easily go into singularity state and is not safe.
4. When this program is running, it will monitor all your keyboard movements, even if the cursor is not in the terminal. Stop the program if you want to type anything else.
5. After a few rounds of policy running, press `q` to stop.

## Experiments

After the environment is setup, you only need to activate the three environments and run the programs accordingly.

💡 **Tip**: To save time and energy, highly recommend using zsh with plugin `zsh-autocomplete` and `zsh-autosuggestions`. A simple setup script can be found at [ubuntu-config](https://github.com/yihuai-gao/ubuntu-config).

### Quick Start Commands

#### 1. Check hardware connection
(Please check arx5-sdk to find out which CAN adapter you are using)

**SLCAN**:
```bash
sudo slcand -o -f -s8 /dev/arxcan0 can0 && sudo ifconfig can0 up
```

**candleLight**:
```bash
sudo ip link set up can0 type can bitrate 1000000
```

#### 2. USB permission:
```bash
sudo chmod -R 777 /dev/bus/usb
```

#### 3. Policy inference node:
```bash
cd detached-umi-policy && conda activate umi
python detached_policy_inference.py -i data/models/cup_wild_vit_l_1img.ckpt
```

#### 4. ARX5 communication server:
```bash
cd arx5-sdk && conda activate arx-py310
python python/communication/zmq_server.py YOUR_MODEL YOUR_CAN_INTERFACE
```

#### 5. Policy deployment code:
```bash
cd umi-arx && conda activate umi-arx
pkill -f eval_arx5 # Important! 
# If there are running python processes in the background, the robot may move crazily
python scripts/eval_arx5.py -i YOUR_PATH_TO/detached-umi-policy/data/models/cup_wild_vit_l_1img.ckpt -o data/experiments/DATE --no_mirror
```

⚠️ **Important Note**: Due to the python multiprocessing issue, your subprocesses may not quit properly after pressing Ctrl-C. Please call `pkill -f eval_arx5` before running again to avoid conflicts with the zombie process.