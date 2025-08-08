"""
python detached_policy_inference.py -i data/models/cup_wild_vit_l_1img.ckpt
"""
import sys
import os
import time
import click
import numpy as np
import torch
import dill
import hydra
import zmq
import logging
import datetime
import json

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from umi.real_world.real_inference_util import get_real_obs_resolution, get_real_umi_action
from diffusion_policy.common.pytorch_util import dict_apply
import omegaconf
import traceback

# Setup logging
def setup_logger(name, log_file, level=logging.INFO):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)
    
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    
    # Also log to console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def save_data_with_timestamp(data, filepath, description=""):
    timestamp = datetime.datetime.now().isoformat()
    data_with_timestamp = {
        "timestamp": timestamp,
        "description": description,
        "data": data
    }
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    if isinstance(data, (np.ndarray, torch.Tensor)):
        np.save(filepath.replace('.json', '.npy'), data_with_timestamp, allow_pickle=True)
    else:
        with open(filepath, 'w') as f:
            json.dump(data_with_timestamp, f, indent=2, default=str)
    
    return timestamp

def echo_exception():
    exc_type, exc_value, exc_traceback = sys.exc_info()
    # Extract unformatted traceback
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
    # Print line of code where the exception occurred

    return "".join(tb_lines)
class PolicyInferenceNode:
    def __init__(self, ckpt_path: str, ip: str, port: int, device: str):
        # Setup logging
        script_dir = os.path.dirname(os.path.abspath(__file__))
        log_dir = os.path.join(script_dir, "logs", "policy_inference")
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = setup_logger('PolicyInference', os.path.join(log_dir, f'policy_inference_{timestamp}.log'))
        self.data_log_dir = os.path.join(script_dir, "data_logs", f"policy_inference_{timestamp}")
        os.makedirs(self.data_log_dir, exist_ok=True)
        
        self.logger.info("=== PolicyInferenceNode 초기화 시작 ===")
        
        self.ckpt_path = ckpt_path
        if not self.ckpt_path.endswith('.ckpt'):
            self.ckpt_path = os.path.join(self.ckpt_path, 'checkpoints', 'latest.ckpt')
        
        self.logger.info(f"체크포인트 로드 시작: {self.ckpt_path}")
        payload = torch.load(open(self.ckpt_path, 'rb'), map_location='cpu', pickle_module=dill)
        save_data_with_timestamp(
            {"ckpt_path": self.ckpt_path, "payload_keys": list(payload.keys())},
            f"{self.data_log_dir}/checkpoint_loaded.json",
            "체크포인트 로드 완료"
        )
        self.logger.info(f"체크포인트 로드 완료: payload keys = {list(payload.keys())}")
        
        self.cfg = payload['cfg']
        # export cfg to yaml
        cfg_path = self.ckpt_path.replace('.ckpt', '.yaml')
        with open(cfg_path, 'w') as f:
            f.write(omegaconf.OmegaConf.to_yaml(self.cfg))
            print(f"Exported config to {cfg_path}")
        print(f"Loading configure: {self.cfg.name}, workspace: {self.cfg._target_}, policy: {self.cfg.policy._target_}, model_name: {self.cfg.policy.obs_encoder.model_name}")
        self.obs_res = get_real_obs_resolution(self.cfg.task.shape_meta)
        self.get_class_start_time = time.monotonic()

        cls = hydra.utils.get_class(self.cfg._target_)
        self.workspace = cls(self.cfg)
        self.workspace: BaseWorkspace
        
        self.logger.info("workspace payload 로드 시작")
        self.workspace.load_payload(payload, exclude_keys=None, include_keys=None)
        self.logger.info("workspace payload 로드 완료")

        self.policy:BaseImagePolicy = self.workspace.model
        if self.cfg.training.use_ema:
            self.policy = self.workspace.ema_model
            print("Using EMA model")
            self.logger.info("EMA 모델 사용")
        else:
            self.logger.info("일반 모델 사용")
            
        self.policy.num_inference_steps = 16
        
        obs_pose_rep = self.cfg.task.pose_repr.obs_pose_repr
        action_pose_repr = self.cfg.task.pose_repr.action_pose_repr
        print('obs_pose_rep', obs_pose_rep)
        print('action_pose_repr', action_pose_repr)
        
        self.device = torch.device(device)
        self.policy.eval().to(self.device)
        self.policy.reset()
        self.ip = ip
        self.port = port
        
        save_data_with_timestamp(
            {
                "obs_pose_rep": obs_pose_rep,
                "action_pose_repr": action_pose_repr,
                "device": str(self.device),
                "num_inference_steps": self.policy.num_inference_steps
            },
            f"{self.data_log_dir}/policy_setup_complete.json",
            "정책 설정 완료"
        )
        self.logger.info("=== PolicyInferenceNode 초기화 완료 ===")
        
        self.prediction_count = 0

    def predict_action(self, obs_dict_np: dict):
        self.prediction_count += 1
        self.logger.info(f"=== 액션 예측 시작 #{self.prediction_count} ===")
        
        # Save input observation
        save_data_with_timestamp(
            obs_dict_np,
            f"{self.data_log_dir}/obs_input_{self.prediction_count:06d}.json",
            f"입력 관찰값 #{self.prediction_count}"
        )
        self.logger.info(f"입력 관찰값 저장: keys = {list(obs_dict_np.keys())}")
        
        with torch.no_grad():
            # Convert to torch tensors
            self.logger.info("관찰값을 텐서로 변환 시작")
            obs_dict = dict_apply(obs_dict_np, 
                lambda x: torch.from_numpy(x).unsqueeze(0).to(self.device))
            self.logger.info(f"텐서 변환 완료: shapes = {[(k, v.shape) for k, v in obs_dict.items()]}")
            
            # Run policy prediction
            prediction_start_time = time.time()
            self.logger.info("정책 예측 시작")
            print("!!!!", self.policy)
            result = self.policy.predict_action(obs_dict)
            prediction_end_time = time.time()
            self.logger.info(f"정책 예측 완료 (소요시간: {prediction_end_time - prediction_start_time:.4f}s)")
            
            # Extract action
            action = result['action_pred'][0].detach().to('cpu').numpy()
            self.logger.info(f"액션 추출 완료: shape = {action.shape}")
            
            # Save predicted action
            save_data_with_timestamp(
                {
                    "action": action.tolist(),
                    "action_shape": action.shape,
                    "prediction_time": prediction_end_time - prediction_start_time,
                    "result_keys": list(result.keys())
                },
                f"{self.data_log_dir}/action_predicted_{self.prediction_count:06d}.json",
                f"예측된 액션 #{self.prediction_count}"
            )
            
            del result
            del obs_dict
            
        self.logger.info(f"=== 액션 예측 완료 #{self.prediction_count} ===")
        return action
    
    def run_node(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(f"tcp://{self.ip}:{self.port}")
        print(f"PolicyInferenceNode is listening on {self.ip}:{self.port}")
        self.logger.info(f"ZMQ 서버 시작: {self.ip}:{self.port}")
        
        communication_count = 0
        
        while True:
            communication_count += 1
            self.logger.info(f"=== ZMQ 통신 #{communication_count} 시작 ===")
            
            # Receive observation
            recv_start_time = time.time()
            self.logger.info("관찰값 수신 대기 중...")
            obs_dict_np = socket.recv_pyobj()
            recv_end_time = time.time()
            self.logger.info(f"관찰값 수신 완료 (소요시간: {recv_end_time - recv_start_time:.4f}s)")
            
            # Save received observation
            save_data_with_timestamp(
                obs_dict_np,
                f"{self.data_log_dir}/zmq_obs_received_{communication_count:06d}.json",
                f"ZMQ로 수신된 관찰값 #{communication_count}"
            )
            
            try:
                start_time = time.monotonic()
                action = self.predict_action(obs_dict_np)
                inference_time = time.monotonic() - start_time
                print(f'Inference time: {inference_time:.3f} s')
                self.logger.info(f'총 추론 시간: {inference_time:.3f} s')
            except Exception as e:
                err_str = echo_exception()
                print(f'Error: {err_str}')
                self.logger.error(f'예측 오류 발생: {err_str}')
                action = err_str
                
                # Save error info
                save_data_with_timestamp(
                    {"error": err_str, "obs_keys": list(obs_dict_np.keys()) if isinstance(obs_dict_np, dict) else "non-dict"},
                    f"{self.data_log_dir}/error_{communication_count:06d}.json",
                    f"예측 오류 #{communication_count}"
                )
            
            # Send action
            send_start_time = time.monotonic()
            self.logger.info("액션 송신 시작")
            socket.send_pyobj(action)
            send_end_time = time.monotonic()
            send_time = send_end_time - send_start_time
            print(f'Send time: {send_time:.3f} s')
            self.logger.info(f'액션 송신 완료 (소요시간: {send_time:.3f}s)')
            
            # Save sent action
            if not isinstance(action, str):  # not an error
                save_data_with_timestamp(
                    {"action": action.tolist() if hasattr(action, 'tolist') else action},
                    f"{self.data_log_dir}/zmq_action_sent_{communication_count:06d}.json",
                    f"ZMQ로 송신된 액션 #{communication_count}"
                )
            
            self.logger.info(f"=== ZMQ 통신 #{communication_count} 완료 ===")
    
@click.command()
@click.option('--input', '-i', required=True, help='Path to checkpoint')
@click.option('--ip', default="0.0.0.0")
@click.option('--port', default=8766, help="Port to listen on")
@click.option('--device', default="cuda", help="Device to run on")
def main(input, ip, port, device):
    node = PolicyInferenceNode(input, ip, port, device)
    node.run_node()
                

if __name__ == '__main__':
    main()