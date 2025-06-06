# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster


import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

# conos
from isaaclab.markers import VisualizationMarkers
from .waypoint import WAYPOINT_CFG


##
# Pre-defined configs
##
from isaaclab_assets.robots.anymal import ANYMAL_C_CFG  # isort: skip
#  from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,  # type: ignore
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 0.8),
            "dynamic_friction_range": (0.6, 0.6),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base"),
            "mass_distribution_params": (-5.0, 5.0),
            "operation": "add",
        },
    )


@configclass
class AnymalCFlatEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 200.0
    decimation = 4
    action_scale = 0.5
    action_space = 12
    observation_space = 51
    state_space = 0
    waypoint_cfg = WAYPOINT_CFG

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=r"C:\isaaclab\IsaacLab\source\isaaclab\ICRA\ICRA2024_Quadruped_Competition\urdf\mapa15.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096, env_spacing=0.0, replicate_physics=True
    )

    # events
    events: EventCfg = EventCfg()

    # robot
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # reward scales

    joint_torque_reward_scale = -2.5e-5
    joint_accel_reward_scale = -2.5e-7
    action_rate_reward_scale = -0.01
    feet_air_time_reward_scale = 0.5
    undesired_contact_reward_scale = -1.0  # -1.0
    flat_orientation_reward_scale = -5.0


@configclass
class AnymalCRoughEnvCfg(AnymalCFlatEnvCfg):
    # env
    # observation_space = 235
    # conos
    observation_space = 238

    #    terrain = TerrainImporterCfg(
    #        prim_path="/World/ground",
    #        terrain_type="generator",
    #        terrain_generator=ROUGH_TERRAINS_CFG,
    #        max_init_terrain_level=9,
    #        collision_group=-1,
    #        physics_material=sim_utils.RigidBodyMaterialCfg(
    #            friction_combine_mode="multiply",
    #            restitution_combine_mode="multiply",
    #            static_friction=1.0,
    #            dynamic_friction=1.0,
    #        ),
    #        visual_material=sim_utils.MdlFileCfg(
    #            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
    #            project_uvw=True,
    #        ),
    #        debug_vis=False,
    #    )

    # terrain = TerrainImporterCfg(
    #    prim_path="/World/ground",
    #    terrain_type="plane",
    #    collision_group=-1,
    #    physics_material=sim_utils.RigidBodyMaterialCfg(
    #        friction_combine_mode="multiply",
    #        restitution_combine_mode="multiply",
    #        static_friction=1.0,
    #        dynamic_friction=1.0,
    #        restitution=0.0,
    #    ),
    #    debug_vis=False,
    # )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=r"C:\isaaclab\IsaacLab\source\isaaclab\ICRA\ICRA2024_Quadruped_Competition\urdf\mapa15.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
        ),
        debug_vis=False,
    )

    # we add a height scanner for perceptive locomotion
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # type: ignore
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],

    )

    # reward scales (override from flat config)
    flat_orientation_reward_scale = 0.0


class AnymalCEnv(DirectRLEnv):
    cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg

    def __init__(
        self,
        cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(cfg, render_mode, **kwargs)

        # Joint position command (deviation from default joint positions)
        self._actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )
        self._previous_actions = torch.zeros(
            self.num_envs,
            gym.spaces.flatdim(self.single_action_space),
            device=self.device,
        )

        # X/Y linear velocity and yaw angular velocity commands
        self._commands = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "position_progress",
                "target_heading",
                "goal_reached",
                "dof_torques_l2",
                "dof_acc_l2",
                "action_rate_l2",
                "feet_air_time",
                "undesired_contacts",
                "flat_orientation_l2",
                "base_contact_penalty",  # Add this if it's not already there
                "thigh_contact_penalty",  # Add this new key
                "fall_penalty",  # Añadir penalización por caída
                "target_timeout_penalty",  # Add this new key
            ]
        }
        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        # self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*THIGH")
        # self._undesired_contact_body_ids += self._base_id  # Añadir el ID de la base a la lista de contactos no deseados

        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_thigh")
        
        # Add counter for thigh contacts
        self._thigh_ids, _ = self._contact_sensor.find_bodies(".*_thigh")
        self._thigh_contact_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._thigh_contact_threshold = 10  # Reset after 10 consecutive frames

        # Add counter for base contacts - similar to thigh contacts
        self._base_contact_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._base_contact_threshold = 10  # Reset after 10 consecutive frames with base contact

        # los de los conos
        self.env_spacing = self.scene.cfg.env_spacing
        self._goal_reached = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )
        self.task_completed = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.bool
        )
        self._num_goals = 29
        self._target_positions = torch.zeros(
            (self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32
        )
        self._markers_pos = torch.zeros(
            (self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32
        )
        self.course_length_coefficient = 15  # 15
        self.course_width_coefficient = 15  # 15
        self.position_tolerance = 0.15  # 0.15
        self.goal_reached_bonus = 30.0  # 10.0
        self.position_progress_weight = 400.0  # 1.0  # 5000.0
        self.heading_coefficient = 0.25  # 0.25
        self.heading_progress_weight = 0.1  # 0.05
        self.base_contact_penalty_weight = -5  # -50
        self._target_index = torch.zeros(
            (self.num_envs), device=self.device, dtype=torch.int32
        )

        # Add persistent tracking for base contacts to avoid false positives
        # self._base_contact_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # self._base_contact_threshold = 3  # Number of consecutive frames to confirm base contact
        
        # Add termination reason tracking
        self._termination_reason = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # 0: not terminated, 1: base contact, 2: timeout, 3: task completed

        # Add termination reason tracking - Actualizar para incluir caída
        self._termination_reason = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        # 0: not terminated, 1: base contact, 2: timeout, 3: task completed, 4: fall below height
        
        # Agregar castigo por caída y umbral de altura
        self.min_height_threshold = -1.0  # Umbral mínimo de altura
        self.fall_penalty_weight = -10  # Igual que el de contacto de base

        # Add thigh contact penalty weight parameter
        self.thigh_contact_penalty_weight = -5  # Same as base contact penalty weight

        # Añadir contador de tiempo para cada objetivo
        self._target_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._target_timeout = 6.0  # Timeout de 6 segundos para alcanzar el siguiente punto
        self._target_timeout_penalty_weight = -10  # Mismo peso que la caída

        # Modificar _termination_reason para incluir el nuevo tipo de terminación
        # 0: not terminated, 1: base contact, 2: timeout episode, 3: task completed, 
        # 4: fall below height, 5: thigh contact, 6: target timeout

        # Añadir nueva clave para el episodio sum
        self._episode_sums["target_timeout_penalty"] = torch.zeros(
            self.num_envs, dtype=torch.float, device=self.device
        )

    def _setup_scene(self):
        # Set up the scene
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            # we add a height scanner for perceptive locomotion
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # conos
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.object_state = []

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        )

    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()

        # conos
        current_target_positions = self._target_positions[
            self._robot._ALL_INDICES, self._target_index  # type: ignore
        ]
        self._position_error_vector = (
            current_target_positions - self._robot.data.root_pos_w[:, :2]
        )
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        heading = self._robot.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self._robot._ALL_INDICES, self._target_index, 1]  # type: ignore
            - self._robot.data.root_link_pos_w[:, 1],
            self._target_positions[self._robot._ALL_INDICES, self._target_index, 0]  # type: ignore
            - self._robot.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )

        height_data = None
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            height_data = (
                self._height_scanner.data.pos_w[:, 2].unsqueeze(1)
                - self._height_scanner.data.ray_hits_w[..., 2]
                - 0.5
            ).clip(-1.0, 1.0)
        obs = torch.cat(
            [
                tensor
                for tensor in (
                    self._position_error.unsqueeze(dim=1),
                    torch.cos(self.target_heading_error).unsqueeze(dim=1),
                    torch.sin(self.target_heading_error).unsqueeze(dim=1),
                    self._robot.data.root_lin_vel_b,
                    self._robot.data.root_ang_vel_b,
                    self._robot.data.projected_gravity_b,
                    self._commands,
                    self._robot.data.joint_pos - self._robot.data.default_joint_pos,
                    self._robot.data.joint_vel,
                    height_data,
                    self._actions,

                )
                if tensor is not None
            ],
            dim=-1,
        )
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # Linear velocity tracking
        # lin_vel_error = torch.sum(torch.square(self._commands[:, :2] - self._robot.data.root_lin_vel_b[:, :2]), dim=1)
        # lin_vel_error_mapped = torch.exp(-lin_vel_error / 0.25)

        # Yaw rate tracking
        # yaw_rate_error = torch.square(self._commands[:, 2] - self._robot.data.root_ang_vel_b[:, 2])
        # yaw_rate_error_mapped = torch.exp(-yaw_rate_error / 0.25)

        # Z velocity tracking
        # ang_vel_error = torch.sum(torch.square(self._robot.data.root_ang_vel_b[:, :2]), dim=1)

        # Conos

        position_progress_rew = torch.nn.functional.elu(self._previous_position_error - self._position_error)
        print(f"Previous error: {self._previous_position_error[0].item():.10f}, Current error: {self._position_error[0].item():.10f}, Diff: {(self._previous_position_error[0] - self._position_error[0]).item():.10f}")
        target_heading_rew = torch.exp(
            -torch.abs(self.target_heading_error) / self.heading_coefficient
        )
        goal_reached = self._position_error < self.position_tolerance
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals - 1)
        self._target_index = self._target_index % self._num_goals

        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        # joint torques
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        # joint acceleration
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        # action rate
        action_rate = torch.sum(
            torch.square(self._actions - self._previous_actions), dim=1
        )
        # feet air time
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[
            :, self._feet_ids
        ]
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]  # type: ignore
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1) * (
            torch.norm(self._commands[:, :2], dim=1) > 0.1
        )
        # undesired contacts
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (
            torch.max(
                torch.norm(
                    net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1  # type: ignore
                ),
                dim=1,
            )[0]
            > 1.0
        )
        contacts = torch.sum(is_contact, dim=1)

        base_contact_forces = torch.max(
            torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1  # type: ignore
        )[0]
        
        base_contact_penalty = torch.sum(base_contact_forces > 0.1, dim=1)  # Increased threshold from 1.0 to 1.5

        # Calculate thigh contact penalty (new code)
        thigh_contact_forces = torch.max(
            torch.norm(net_contact_forces[:, :, self._thigh_ids], dim=-1), dim=1  # type: ignore
        )[0]
        thigh_contact_penalty = torch.sum(thigh_contact_forces > 0.1, dim=1)

        # flat orientation
        flat_orientation = torch.sum(
            torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1
        )

        # Detectar caída por debajo de la altura mínima
        below_height_limit = self._robot.data.root_pos_w[:, 2] < self.min_height_threshold
        fall_penalty = below_height_limit.to(torch.float)  # Convert boolean to float (0.0 or 1.0)
        
        # Target timeout penalty
        self._target_timer += self.step_dt
        target_timeout = self._target_timer > self._target_timeout
        target_timeout_penalty = target_timeout.to(torch.float)
        
        # Reiniciar el contador cuando se alcanza un objetivo
        self._target_timer[goal_reached] = 0.0

        rewards = {
            # conos
            "position_progress": position_progress_rew * self.position_progress_weight,
            "target_heading": target_heading_rew * self.heading_progress_weight,
            "goal_reached": goal_reached * self.goal_reached_bonus,
            "dof_torques_l2": joint_torques
            * self.cfg.joint_torque_reward_scale
            * self.step_dt,
            "action_rate_l2": action_rate
            * self.cfg.action_rate_reward_scale
            * self.step_dt,
            "dof_acc_l2": joint_accel
            * self.cfg.joint_accel_reward_scale
            * self.step_dt,
            "feet_air_time": air_time
            * self.cfg.feet_air_time_reward_scale
            * self.step_dt,
            "undesired_contacts": contacts
            * self.cfg.undesired_contact_reward_scale
            * self.step_dt,
            "flat_orientation_l2": flat_orientation
            * self.cfg.flat_orientation_reward_scale
            * self.step_dt,
            "base_contact_penalty": base_contact_penalty * self.base_contact_penalty_weight,
            "thigh_contact_penalty": thigh_contact_penalty * self.thigh_contact_penalty_weight,  # Add this line
            "fall_penalty": fall_penalty * self.fall_penalty_weight,  # Añadir castigo por caída
            "target_timeout_penalty": target_timeout_penalty * self._target_timeout_penalty_weight,  # Add this line
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Track time outs
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        
        # Detect if robot's base touches the ground
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        base_contact_current = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1
            )[0] > 0.0,
            dim=1,
        )
        
        # Update counter for base contacts
        self._base_contact_counter[base_contact_current] += 1
        self._base_contact_counter[~base_contact_current] = 0
        
        # Check if base contact threshold reached
        base_contact_persistent = self._base_contact_counter >= self._base_contact_threshold
        
        # Detect if any thigh touches the ground
        thigh_contact_current = torch.any(
            torch.max(
                torch.norm(net_contact_forces[:, :, self._thigh_ids], dim=-1), dim=1
            )[0] > 0.0,
            dim=1,
        )
        
        # Update counter for thigh contacts
        self._thigh_contact_counter[thigh_contact_current] += 1
        self._thigh_contact_counter[~thigh_contact_current] = 0
        
        # Check if thigh contact threshold reached
        thigh_contact_persistent = self._thigh_contact_counter >= self._thigh_contact_threshold
        
        # Detect falls below minimum height
        below_height_limit = self._robot.data.root_pos_w[:, 2] < self.min_height_threshold

        # Detect target timeout
        target_timeout = self._target_timer > self._target_timeout
        
        # Set termination reasons
        self._termination_reason[:] = 0
        self._termination_reason[below_height_limit] = 4  # Fall below height
        self._termination_reason[base_contact_persistent & ~below_height_limit] = 1  # Base contact
        self._termination_reason[thigh_contact_persistent & ~base_contact_persistent & ~below_height_limit] = 5  # Thigh contact
        self._termination_reason[time_out & ~base_contact_persistent & ~thigh_contact_persistent & ~below_height_limit] = 2  # Timeout
        self._termination_reason[self.task_completed & ~base_contact_persistent & ~thigh_contact_persistent & ~time_out & ~below_height_limit] = 3  # Task complete
        self._termination_reason[target_timeout & ~base_contact_persistent & ~thigh_contact_persistent & ~time_out & ~below_height_limit] = 6  # Target timeout
        
        # Return combined termination conditions and task completion
        return base_contact_persistent | time_out | below_height_limit | thigh_contact_persistent | target_timeout, self.task_completed

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES  # type: ignore
        self._robot.reset(env_ids)  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore

        num_reset = len(env_ids)

        # Set env origins manually
        self._terrain.env_origins[env_ids, 0] = 5.5
        self._terrain.env_origins[env_ids, 1] = 3.0
        self._terrain.env_origins[env_ids, 2] = 0.2

        # Always reset episode length buffer for reset environments
        self.episode_length_buf[env_ids] = 0
        
        # For full reset, spread out to avoid spikes
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf[:] = torch.randint_like(
                self.episode_length_buf, high=int(self.max_episode_length / 4)
            )
            
        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0
        # Sample new commands
        self._commands[env_ids] = torch.zeros_like(self._commands[env_ids]).uniform_(
            -1.0, 1.0
        )
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        
        # Añadir variación aleatoria a la posición (±0.1m en X e Y)
        position_variation = torch.rand((len(env_ids), 2), device=self.device) * 1 - 0.5
        default_root_state[:, 0] += position_variation[:, 0]  # Variación en X
        
        # Añadir el origen del entorno
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        
        # Generar una rotación aleatoria alrededor del eje Z (ángulo completo aleatorio)
        random_yaw = torch.rand((len(env_ids),), device=self.device) * 2 * 3.14159  # Ángulo aleatorio entre 0 y 2π
        
        # Convertir el ángulo de yaw a cuaternión (rotación alrededor del eje Z)
        qx = torch.sin(random_yaw * 0.5)
        qy = torch.zeros_like(random_yaw)
        qz = torch.zeros_like(random_yaw)
        qw = torch.cos(random_yaw * 0.5)
        
        # Asignar el cuaternión al estado raíz
        default_root_state[:, 3:7] = torch.stack([qx, qy, qz, qw], dim=1)
        
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)  # type: ignore
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)  # type: ignore
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)  # type: ignore

        # conos
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, 2] = 0.5

        self._target_positions[env_ids, 0, 0] = 4.7
        self._target_positions[env_ids, 0, 1] = 3.0
        self._target_positions[env_ids, 1, 0] = 2.8
        self._target_positions[env_ids, 1, 1] = 3.0
        self._target_positions[env_ids, 2, 0] = 2.9
        self._target_positions[env_ids, 2, 1] = 4.25
        self._target_positions[env_ids, 3, 0] = 1.53
        self._target_positions[env_ids, 3, 1] = 4.25
        self._target_positions[env_ids, 4, 0] = 0.17
        self._target_positions[env_ids, 4, 1] = 4.25
        self._target_positions[env_ids, 5, 0] = 0.17
        self._target_positions[env_ids, 5, 1] = 3.0
        self._target_positions[env_ids, 6, 0] = -1.27
        self._target_positions[env_ids, 6, 1] = 3.0
        self._target_positions[env_ids, 7, 0] = -2.8
        self._target_positions[env_ids, 7, 1] = 3.0
        self._target_positions[env_ids, 8, 0] = -2.8
        self._target_positions[env_ids, 8, 1] = 4.25
        self._target_positions[env_ids, 9, 0] = -4.72
        self._target_positions[env_ids, 9, 1] = 4.25
        self._target_positions[env_ids, 10, 0] = -5.76
        self._target_positions[env_ids, 10, 1] = 4.25
        self._target_positions[env_ids, 11, 0] = -5.76
        self._target_positions[env_ids, 11, 1] = 3.0
        self._target_positions[env_ids, 12, 0] = -6.13
        self._target_positions[env_ids, 12, 1] = 1.78
        self._target_positions[env_ids, 13, 0] = -6.13
        self._target_positions[env_ids, 13, 1] = 0.6
        self._target_positions[env_ids, 14, 0] = -6.13
        self._target_positions[env_ids, 14, 1] = -0.6
        self._target_positions[env_ids, 15, 0] = -6.13
        self._target_positions[env_ids, 15, 1] = -1.9
        self._target_positions[env_ids, 16, 0] = -5.8
        self._target_positions[env_ids, 16, 1] = -3.0
        self._target_positions[env_ids, 17, 0] = -5.8
        self._target_positions[env_ids, 17, 1] = -4.25
        self._target_positions[env_ids, 18, 0] = -4.12
        self._target_positions[env_ids, 18, 1] = -4.25
        self._target_positions[env_ids, 19, 0] = -2.63
        self._target_positions[env_ids, 19, 1] = -4.27
        self._target_positions[env_ids, 20, 0] = -2.63
        self._target_positions[env_ids, 20, 1] = -3.0
        self._target_positions[env_ids, 21, 0] = -1.4
        self._target_positions[env_ids, 21, 1] = -3.0
        self._target_positions[env_ids, 22, 0] = 0.21
        self._target_positions[env_ids, 22, 1] = -3.0
        self._target_positions[env_ids, 23, 0] = 0.21
        self._target_positions[env_ids, 23, 1] = -4.25
        self._target_positions[env_ids, 24, 0] = 1.81
        self._target_positions[env_ids, 24, 1] = -4.25
        self._target_positions[env_ids, 25, 0] = 3.17
        self._target_positions[env_ids, 25, 1] = -4.25
        self._target_positions[env_ids, 26, 0] = 3.17
        self._target_positions[env_ids, 26, 1] = -3.0
        self._target_positions[env_ids, 27, 0] = 4.77
        self._target_positions[env_ids, 27, 1] = -3.0
        self._target_positions[env_ids, 28, 0] = 6.2
        self._target_positions[env_ids, 28, 1] = -3.0

        # Aplicar el desplazamiento por entorno
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        current_target_positions = self._target_positions[self._robot._ALL_INDICES, self._target_index ]
        self._position_error_vector = (
            current_target_positions[:, :2] - self._robot.data.root_pos_w[:, :2]
        )
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        heading = self._robot.data.heading_w[:]
        target_heading_w = torch.atan2(
            self._target_positions[:, 0, 1] - self._robot.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self._robot.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(
            torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading)
        )
        self._previous_heading_error = self._heading_error.clone()

        # Reiniciar el contador de tiempo para cada objetivo
        self._target_timer[env_ids] = 0.0

        # Logging
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = (
                episodic_sum_avg / self.max_episode_length_s
            )
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        
        # Track termination reasons more clearly - Agregar registro de caídas y contacto de muslos
        base_contact_count = torch.sum(self._termination_reason[env_ids] == 1).item()
        timeout_count = torch.sum(self._termination_reason[env_ids] == 2).item()
        task_complete_count = torch.sum(self._termination_reason[env_ids] == 3).item()
        fall_count = torch.sum(self._termination_reason[env_ids] == 4).item()
        thigh_contact_count = torch.sum(self._termination_reason[env_ids] == 5).item()  # Add thigh contact count
        target_timeout_count = torch.sum(self._termination_reason[env_ids] == 6).item()  # Add target timeout count
        
        extras["Episode_Termination/base_contact"] = base_contact_count
        extras["Episode_Termination/time_out"] = timeout_count
        extras["Episode_Termination/task_completed"] = task_complete_count
        extras["Episode_Termination/fall_below_height"] = fall_count
        extras["Episode_Termination/thigh_contact"] = thigh_contact_count
        extras["Episode_Termination/target_timeout"] = target_timeout_count
        
        self.extras["log"].update(extras)