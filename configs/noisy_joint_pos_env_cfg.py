# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import torch
from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab_assets.robots.franka import FRANKA_PANDA_CFG

from isaaclab.utils.noise.noise_cfg import NoiseModelCfg, NoiseCfg
from isaaclab.utils.noise.noise_model import NoiseModel

@configclass
class EpisodeVaryingGaussianNoiseCfg(NoiseCfg):
    func = None # Used dynamically
    operation = "add"

class EpisodeVaryingGaussianNoiseModel(NoiseModel):
    def __init__(self, cfg, num_envs, device):
        super().__init__(cfg, num_envs, device)
        self.sigma_min = cfg.sigma_min
        self.sigma_max = cfg.sigma_max
        self.current_sigma = torch.zeros((num_envs, 1), device=device)

    def reset(self, env_ids=None):
        if env_ids is None:
            num_reset = self.current_sigma.shape[0]
            env_ids = slice(None)
        else:
            if isinstance(env_ids, torch.Tensor):
                num_reset = env_ids.numel()
            else:
                num_reset = len(env_ids)
        # Sample uniform sigma between [sigma_min, sigma_max] per resetting env
        self.current_sigma[env_ids] = torch.rand((num_reset, 1), device=self._device) * (self.sigma_max - self.sigma_min) + self.sigma_min

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        # Add gaussian noise with standard deviation `current_sigma`
        return data + self.current_sigma * torch.randn_like(data)

@configclass
class EpisodeVaryingGaussianNoiseModelCfg(NoiseModelCfg):
    noise_cfg: EpisodeVaryingGaussianNoiseCfg = EpisodeVaryingGaussianNoiseCfg()
    sigma_min: float = 0.0
    sigma_max: float = 0.03
    class_type: type = EpisodeVaryingGaussianNoiseModel


@configclass
class FrankaCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        self.rewards.reaching_object.weight = 0.1

        # Set Franka as robot
        self.scene.robot = FRANKA_PANDA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Set actions for the specific robot type (franka)
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=["panda_joint.*"], scale=0.5, use_default_offset=True
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )
        # Set the body name for the end effector
        self.commands.object_pose.body_name = "panda_hand"

        # Set Cube as object
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.8, 0.8, 0.8),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Listens to the required transforms
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(
                        pos=[0.0, 0.0, 0.1034],
                    ),
                ),
            ],
        )

        # ====== INJECT OBSERVATION NOISE ======
        self.observations.policy.object_position.noise = EpisodeVaryingGaussianNoiseModelCfg(sigma_min=0.0, sigma_max=0.03)

@configclass
class FrankaCubeLiftEnvCfg_PLAY(FrankaCubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
