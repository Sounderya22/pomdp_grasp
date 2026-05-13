import gymnasium as gym

from . import custom_lift_env_cfg

gym.register(
    id="Custom-Isaac-Lift-Cube-Franka-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.custom_lift_env_cfg:LiftEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.custom_rsl_rl_ppo_cfg:CustomLiftCubePPORunnerCfg",
    },
    disable_env_checker=True,
)
