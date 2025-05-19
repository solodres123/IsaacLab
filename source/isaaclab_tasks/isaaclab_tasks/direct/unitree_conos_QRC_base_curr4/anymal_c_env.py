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
from isaaclab.markers import VisualizationMarkers
from .waypoint import WAYPOINT_CFG
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG


@configclass
class EventCfg:

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
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )


@configclass
class AnymalCFlatEnvCfg(DirectRLEnvCfg):
    episode_length_s = 84.0
    decimation = 4
    action_scale = 0.25
    action_space = 12
    observation_space = 51
    state_space = 0
    waypoint_cfg = WAYPOINT_CFG

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
        usd_path=r"C:\isaaclab\IsaacLab\source\isaaclab\ICRA\ICRA2024_Quadruped_Competition\urdf\mapaCurriculumAyuda.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
        ),
        debug_vis=False,
    )

    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=0.0, replicate_physics=True)

    events: EventCfg = EventCfg()

    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")  # type: ignore

    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=True,
    )

    # reward scales
    joint_torque_reward_scale = -2e-5  # -2.5e-5
    joint_accel_reward_scale = -2.5e-7  # -2.5e-7
    action_rate_reward_scale = -0.01  # -0.01

    feet_air_time_reward_scale = 0.5  # 0.01

    undesired_contact_reward_scale = -1.0  # -1.0
    flat_orientation_reward_scale = -5.0
    goal_reached_reward_scale = 3.0
    position_progress_reward_scale = 5  # 1.0  # 5000.0 #200  #800

    heading_progress_reward_scale = 0.02  # 0.05  #1

    base_contact_penalty_reward_scale = -1.0  # Cambiado de -5 a 0
    thigh_contact_penalty_reward_scale = -1.0  # Cambiado de -5 a 0
    fall_penalty_reward_scale = -1.0  # Cambiado de -5 a 0
    target_timeout_penalty_reward_scale = -1.0  # Cambiado de -5 a 0
    tipped_penalty_reward_scale = -1.0  # Añadir esta línea
    calf_contact_penalty_reward_scale = -1.0  # Mismo peso que los otros contactos


@configclass
class AnymalCRoughEnvCfg(AnymalCFlatEnvCfg):
    # observation_space = 238
    observation_space = 235
    flat_orientation_reward_scale = -0.0
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=r"C:\isaaclab\IsaacLab\source\isaaclab\ICRA\ICRA2024_Quadruped_Competition\urdf\mapaCurriculumAyuda.usd",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
        ),
        debug_vis=False,
    )
    height_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),  # type: ignore
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )


class AnymalCEnv(DirectRLEnv):
    cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg

    # Init -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def __init__(self , cfg: AnymalCFlatEnvCfg | AnymalCRoughEnvCfg , render_mode: str | None = None , **kwargs,):
        super().__init__(cfg, render_mode, **kwargs)

        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device,)
        self._previous_actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device,)

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
                "base_contact_penalty",
                "thigh_contact_penalty",
                "fall_penalty",
                "target_timeout_penalty",
                "tipped_penalty"]
        }

        # Get specific body indices
        self._base_id, _ = self._contact_sensor.find_bodies("base")
        self._feet_ids, _ = self._contact_sensor.find_bodies(".*_foot")
        self._undesired_contact_body_ids, _ = self._contact_sensor.find_bodies(".*_thigh|.*_base")
        
        # Add counter for tipped detection
        self._tipped_counter = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._max_tilt_rad = torch.pi / 2.0  # 90 degrees in radians
        self._tipped_threshold = 10  # Reset after 10 consecutive frames tipped over

        # Waypoint configuration
        self._num_goals = 14
        self.env_spacing = self.scene.cfg.env_spacing
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._termination_reason = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._target_timer = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._target_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.position_tolerance = 0.15  # Target reach tolerance
        self.min_height_threshold = -1.0  # Minimum height threshold
        self._target_timeout = 6.0  # Timeout for reaching next point

        # Curriculum learning counters
        self._circuit_completions = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._circuits_advanced = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._max_circuits = 5  # Increase max circuits to include the new first level (0-4)
        self._completions_required = 1  # Default completions required to advance
        
        # For level 0 (random targets)
        self._level0_targets_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32)
        self._level0_targets_required = 3  # Need to reach 3 targets in level 0 to advance
        
        # Offset for curriculum
        self._circuit_x_offset = -0.0
        self._waypoint_offset_x = -11.0  # Default offset for higher levels

    # Definicion de la escena -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        if isinstance(self.cfg, AnymalCRoughEnvCfg):
            self._height_scanner = RayCaster(self.cfg.height_scanner)
            self.scene.sensors["height_scanner"] = self._height_scanner
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)
        self.waypoints.set_visibility(False)
        self.object_state = []

    # Definicion del step ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone()
        self._processed_actions = (
            self.cfg.action_scale * self._actions + self._robot.data.default_joint_pos
        )

    # Definicion aplicar acciones -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _apply_action(self):
        self._robot.set_joint_position_target(self._processed_actions)

    # Definicion de observaciones -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        current_target_positions = self._target_positions[self._robot._ALL_INDICES, self._target_index]  # type: ignore
        self._position_error_vector = current_target_positions - self._robot.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        heading = self._robot.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self._robot._ALL_INDICES, self._target_index, 1] - self._robot.data.root_link_pos_w[:, 1],  # type: ignore
            self._target_positions[self._robot._ALL_INDICES, self._target_index, 0] - self._robot.data.root_link_pos_w[:, 0])  # type: ignore
        target_heading_w = torch.atan2(torch.sin(target_heading_w), torch.cos(target_heading_w))
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

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

    # Definicion de rewards -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        position_progress_rew = torch.nn.functional.elu(self._previous_position_error - self._position_error)
        target_heading_rew = torch.cos(self.target_heading_error) + 1  # Normalizado entre 0 y 2
        goal_reached = self._position_error < self.position_tolerance
        self._target_index = self._target_index + goal_reached

        # Detectar cuando un robot completa todo el circuito
        circuit_completed = self._target_index > (self._num_goals - 1)

        # Reiniciar el índice de waypoint al completar el circuito
        self._target_index = self._target_index % self._num_goals

        # Para los robots que acaban de completar un circuito
        self._circuit_completions[circuit_completed] += 1

        # Verificar qué robots han completado suficientes veces el circuito para avanzar
        ready_to_advance = (self._circuit_completions >= self._completions_required) & (self._circuits_advanced < self._max_circuits)

        # Actualizar el contador de circuitos avanzados para los robots listos
        self._circuits_advanced[ready_to_advance] += 1

        # Resetear el contador de completaciones para los que avanzan
        self._circuit_completions[ready_to_advance] = 0

        # Establecer la bandera de tarea completada para los que acaban un circuito
        self.task_completed = circuit_completed

        self._target_index = self._target_index % self._num_goals
        self._target_timer[goal_reached] = 0.0

        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)
        joint_torques = torch.sum(torch.square(self._robot.data.applied_torque), dim=1)
        joint_accel = torch.sum(torch.square(self._robot.data.joint_acc), dim=1)
        action_rate = torch.sum(torch.square(self._actions - self._previous_actions), dim=1)
        first_contact = self._contact_sensor.compute_first_contact(self.step_dt)[:, self._feet_ids]  # type: ignore
        last_air_time = self._contact_sensor.data.last_air_time[:, self._feet_ids]  # type: ignore
        air_time = torch.sum((last_air_time - 0.5) * first_contact, dim=1)

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_contact = (torch.max(torch.norm(net_contact_forces[:, :, self._undesired_contact_body_ids], dim=-1), dim=1,)[0] > 1.0)  # type: ignore
        contacts = torch.sum(is_contact, dim=1)

        flat_orientation = torch.sum(torch.square(self._robot.data.projected_gravity_b[:, :2]), dim=1)

        below_height_limit = self._robot.data.root_pos_w[:, 2] < self.min_height_threshold
        fall_penalty = below_height_limit.to(torch.float)

        self._target_timer += self.step_dt
        target_timeout = self._target_timer > self._target_timeout
        target_timeout_penalty = target_timeout.to(torch.float)
        self._target_timer[goal_reached] = 0.0

        # Obtener la gravedad proyectada en el sistema de referencia del cuerpo
        projected_gravity_b = self._robot.data.projected_gravity_b

        # Calcular el ángulo de inclinación desde la vertical
        gravity_norm = torch.norm(projected_gravity_b, dim=1)
        cos_tilt = -projected_gravity_b[:, 2] / gravity_norm

        # El ángulo en radianes
        tilt_angle = torch.acos(cos_tilt.clamp(-1.0, 1.0))

        # Detección de robot tumbado (ángulo > 90 grados)
        is_tipped = tilt_angle > self._max_tilt_rad
        tipped_penalty = is_tipped.to(torch.float)

        rewards = {
            "position_progress": position_progress_rew * self.cfg.position_progress_reward_scale * target_heading_rew,
            "target_heading": target_heading_rew * self.cfg.heading_progress_reward_scale,
            "goal_reached": goal_reached * self.cfg.goal_reached_reward_scale,

            "dof_torques_l2": joint_torques * self.cfg.joint_torque_reward_scale * self.step_dt,
            "action_rate_l2": action_rate * self.cfg.action_rate_reward_scale * self.step_dt,
            "dof_acc_l2": joint_accel * self.cfg.joint_accel_reward_scale * self.step_dt,
            "feet_air_time": air_time * self.cfg.feet_air_time_reward_scale * self.step_dt,
            "undesired_contacts": contacts * self.cfg.undesired_contact_reward_scale,
            "flat_orientation_l2": flat_orientation * self.cfg.flat_orientation_reward_scale * self.step_dt,

            "fall_penalty": fall_penalty * self.cfg.fall_penalty_reward_scale,
            "target_timeout_penalty": target_timeout_penalty * self.cfg.target_timeout_penalty_reward_scale,
            "tipped_penalty": tipped_penalty * self.cfg.tipped_penalty_reward_scale,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging rewards
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # Definicion de terminaciones -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:

        projected_gravity_b = self._robot.data.projected_gravity_b
        gravity_norm = torch.norm(projected_gravity_b, dim=1)
        cos_tilt = -projected_gravity_b[:, 2] / gravity_norm
        tilt_angle = torch.acos(cos_tilt.clamp(-1.0, 1.0))

        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        is_tipped_current = tilt_angle > self._max_tilt_rad
        died = torch.any(torch.max(torch.norm(net_contact_forces[:, :, self._base_id], dim=-1), dim=1)[0] > 1.0, dim=1) # type: ignore

        self._tipped_counter[is_tipped_current] += 1
        self._tipped_counter[~is_tipped_current] = 0

        tipped_persistent = self._tipped_counter >= self._tipped_threshold
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        below_height_limit = self._robot.data.root_pos_w[:, 2] < self.min_height_threshold
        target_timeout = self._target_timer > self._target_timeout

        # Set termination reasons
        self._termination_reason[:] = 0
        self._termination_reason[time_out] = 1  # Timeout
        self._termination_reason[self.task_completed] = 2  # Task complete
        self._termination_reason[below_height_limit] = 3  # Fall below height
        self._termination_reason[target_timeout] = 4  # Target timeout
        self._termination_reason[tipped_persistent] = 5  # Tipped over
        self._termination_reason[died] = 6  # Died

        # Return dones and task completed
        return time_out | below_height_limit | target_timeout | tipped_persistent | died, self.task_completed
    
    # Definicion de reseteos -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES  # type: ignore

        self._robot.reset(env_ids)  # type: ignore
        super()._reset_idx(env_ids)  # type: ignore

        position_variation = torch.rand((len(env_ids), 2), device=self.device) * 1 - 0.3
        random_yaw = torch.rand((len(env_ids),), device=self.device) * 2 * 3.14159  # Random angle between 0 and 2π

        qw = torch.cos(random_yaw * 0.5) 
        qx = torch.zeros_like(random_yaw)
        qy = torch.zeros_like(random_yaw)
        qz = torch.sin(random_yaw * 0.5) 

        self._actions[env_ids] = 0.0
        self._previous_actions[env_ids] = 0.0

        # Define terrain origins array
        terrain_origins = torch.tensor([
            [3.5, -6.6, 0.2],  # Level 1 (after random targets)
            [-1.3, -6.6, 0.2], # Level 2
            [3.5, -1.8, 0.2],  # Level 3
            [3.5, 3.0, 0.2],   # Level 4
            [-1.3, -1.8, 0.2], # Level 5
            [-1.3, 3.0, 0.2],  # Level 6
        ], device=self.device)

        # Set terrain origins based on circuit level
        for i, env_id in enumerate(env_ids):
            # For level 0 (random targets), use a default position
            if self._circuits_advanced[env_id] == 0:
                # Use first terrain origin but with a small offset to differentiate
                self._terrain.env_origins[env_id] = torch.tensor([0.0, 0.0, 0.2], device=self.device)
            else:
                # For higher levels, use the circuit patterns with proper origins
                # Subtract 1 from circuit level to get the terrain index (since level 0 is special)
                terrain_idx = (self._circuits_advanced[env_id] - 1) % 6
                self._terrain.env_origins[env_id] = terrain_origins[terrain_idx]

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]

        # Add position variation and set root state
        for i, env_id in enumerate(env_ids):
            # Apply normal variation and base offset
            offset_x = position_variation[i, 0] + 0.2
            
            # For levels after 0, apply curriculum offset
            if self._circuits_advanced[env_id] > 0:
                offset_x += self._waypoint_offset_x
            
            default_root_state[i, 0] += offset_x  # Apply total X offset
            default_root_state[i, :3] += self._terrain.env_origins[env_id]  # Add terrain origin
            
        # Set quaternion
        default_root_state[:, 3:7] = torch.stack([qw, qx, qy, qz], dim=1)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)  # type: ignore
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)  # type: ignore
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)  # type: ignore

        # Reset waypoint data
        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, 2] = 0.5
        
        # Handle waypoints based on level
        for i, env_id in enumerate(env_ids):
            # Level 0 - Random target positions like in anymal_c_conos_copy
            if self._circuits_advanced[env_id] == 0:
                # Define the range for random target distribution
                x_range = (-4.0, 4.0)
                y_range = (-4.0, 4.0)
                
                # Generate random positions in the defined area - only generate the first 3 waypoints for level 0
                num_level0_targets = 3  # We only need 3 active targets for level 0
                
                # Initialize all targets (even though we only use the first 3)
                self._target_positions[env_id, :, 0] = torch.rand(self._num_goals, device=self.device) * (x_range[1] - x_range[0]) + x_range[0]
                self._target_positions[env_id, :, 1] = torch.rand(self._num_goals, device=self.device) * (y_range[1] - y_range[0]) + y_range[0]
                
                # Apply the environment offset
                self._target_positions[env_id, :, :] += self._terrain.env_origins[env_id, :2]
                
                # Reset level 0 target counter
                self._level0_targets_reached[env_id] = 0
            else:
                # For higher levels, use the circuit patterns
                terrain_idx = (self._circuits_advanced[env_id] - 1) % 6
                
                if terrain_idx == 0:
                    waypoints = self._define_waypoints_terrain_0()
                elif terrain_idx == 1:
                    waypoints = self._define_waypoints_terrain_1()
                elif terrain_idx == 2:
                    waypoints = self._define_waypoints_terrain_2()
                elif terrain_idx == 3:
                    waypoints = self._define_waypoints_terrain_3()
                elif terrain_idx == 4:
                    waypoints = self._define_waypoints_terrain_4()
                else:  # terrain_idx == 5
                    waypoints = self._define_waypoints_terrain_5()
                
                # Apply terrain offset
                self._target_positions[env_id] = waypoints + self._terrain.env_origins[env_id, :2].unsqueeze(0)

        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        current_target_positions = self._target_positions[env_ids, self._target_index[env_ids]]
        
        if not hasattr(self, '_position_error'):
            # If self._position_error doesn't exist yet (first reset):
            self._position_error_vector = (current_target_positions - self._robot.data.root_pos_w[env_ids, :2])
            self._position_error = torch.norm(self._position_error_vector, dim=-1)
            self._previous_position_error = self._position_error.clone()
        else:
            # If self._position_error already exists:
            # Save current error for env_ids
            current_error = self._position_error[env_ids].clone()
            
            # Update for specific env_ids
            self._position_error_vector[env_ids] = current_target_positions - self._robot.data.root_pos_w[env_ids, :2]
            self._previous_position_error[env_ids] = current_error  # Use current error as previous
            self._position_error[env_ids] = torch.norm(self._position_error_vector[env_ids], dim=-1)

        # Calculate heading error
        heading = self._robot.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self._robot._ALL_INDICES, self._target_index, 1] - self._robot.data.root_link_pos_w[:, 1],
            self._target_positions[self._robot._ALL_INDICES, self._target_index, 0] - self._robot.data.root_link_pos_w[:, 0])
        
        # Normalize to [-pi, pi]
        target_heading_w = torch.atan2(torch.sin(target_heading_w), torch.cos(target_heading_w))
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))

        # Reset target timer
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
        thigh_contact_count = torch.sum(self._termination_reason[env_ids] == 5).item()
        target_timeout_count = torch.sum(self._termination_reason[env_ids] == 6).item()
        tipped_over_count = torch.sum(self._termination_reason[env_ids] == 7).item()  # Nuevo contador
        calf_contact_count = torch.sum(self._termination_reason[env_ids] == 8).item()

        extras["Episode_Termination/base_contact"] = base_contact_count
        extras["Episode_Termination/time_out"] = timeout_count
        extras["Episode_Termination/task_completed"] = task_complete_count
        extras["Episode_Termination/fall_below_height"] = fall_count
        extras["Episode_Termination/thigh_contact"] = thigh_contact_count
        extras["Episode_Termination/target_timeout"] = target_timeout_count
        extras["Episode_Termination/tipped_over"] = tipped_over_count  # Nuevo log
        extras["Episode_Termination/calf_contact"] = calf_contact_count

        # En el método _reset_idx, añadir al logging:
        extras["Curriculum/circuit_completions"] = torch.mean(self._circuit_completions.float()).item()
        extras["Curriculum/circuits_advanced"] = torch.mean(self._circuits_advanced.float()).item()
        for i in range(self._max_circuits + 1):
            extras[f"Curriculum/robots_in_circuit_{i}"] = torch.sum(self._circuits_advanced == i).item() / self.num_envs

        # Add level 0 specific logging
        extras["Curriculum/level0_targets_reached"] = torch.mean(self._level0_targets_reached.float()).item()
        extras["Curriculum/level0_agents"] = torch.sum(self._circuits_advanced == 0).item() / self.num_envs

        self.extras["log"].update(extras)

    def _define_waypoints_terrain_0(self):
        # Origin for terrain 0: [3.5, -6.6]
        # Waypoints (with origin subtraction pre-calculated)
        waypoints = torch.zeros((self._num_goals, 2), device=self.device)
        waypoints[0] = torch.tensor([-1.7, 0.0], device=self.device)   # [1.8, -6.6] - [3.5, -6.6]
        waypoints[1] = torch.tensor([-2.8, 0.0], device=self.device)   # [0.7, -6.6] - [3.5, -6.6]
        waypoints[2] = torch.tensor([-2.8, 1.2], device=self.device)   # [0.7, -5.4] - [3.5, -6.6]
        waypoints[3] = torch.tensor([-1.7, 1.2], device=self.device)   # [1.8, -5.4] - [3.5, -6.6]
        waypoints[4] = torch.tensor([-0.5, 1.2], device=self.device)   # [3.0, -5.4] - [3.5, -6.6]
        waypoints[5] = torch.tensor([0.7, 1.2], device=self.device)    # [4.2, -5.4] - [3.5, -6.6]
        waypoints[6] = torch.tensor([0.7, 2.4], device=self.device)    # [4.2, -4.2] - [3.5, -6.6]
        waypoints[7] = torch.tensor([0.7, 3.5], device=self.device)    # [4.2, -3.1] - [3.5, -6.6]
        waypoints[8] = torch.tensor([-0.5, 3.5], device=self.device)   # [3.0, -3.1] - [3.5, -6.6]
        waypoints[9] = torch.tensor([-0.5, 2.4], device=self.device)   # [3.0, -4.2] - [3.5, -6.6]
        waypoints[10] = torch.tensor([-1.7, 2.4], device=self.device)  # [1.8, -4.2] - [3.5, -6.6]
        waypoints[11] = torch.tensor([-1.7, 3.5], device=self.device)  # [1.8, -3.1] - [3.5, -6.6]
        waypoints[12] = torch.tensor([-2.8, 3.5], device=self.device)  # [0.7, -3.1] - [3.5, -6.6]
        waypoints[13] = torch.tensor([-2.8, 2.4], device=self.device)  # [0.7, -4.2] - [3.5, -6.6]
        return waypoints

    def _define_waypoints_terrain_1(self):
        # Origin for terrain 1: [3.5, -1.8]
        # Waypoints (with origin subtraction pre-calculated)
        waypoints = torch.zeros((self._num_goals, 2), device=self.device)
        waypoints[0] = torch.tensor([-1.7, 0.0], device=self.device)   # [1.8, -1.8] - [3.5, -1.8]
        waypoints[1] = torch.tensor([-2.8, 0.0], device=self.device)   # [0.7, -1.8] - [3.5, -1.8]
        waypoints[2] = torch.tensor([-2.8, 1.2], device=self.device)   # [0.7, -0.6] - [3.5, -1.8]
        waypoints[3] = torch.tensor([-1.7, 1.2], device=self.device)   # [1.8, -0.6] - [3.5, -1.8]
        waypoints[4] = torch.tensor([-0.5, 1.2], device=self.device)   # [3.0, -0.6] - [3.5, -1.8]
        waypoints[5] = torch.tensor([0.7, 1.2], device=self.device)    # [4.2, -0.6] - [3.5, -1.8]
        waypoints[6] = torch.tensor([0.7, 2.4], device=self.device)    # [4.2, 0.6] - [3.5, -1.8]
        waypoints[7] = torch.tensor([0.7, 3.5], device=self.device)    # [4.2, 1.7] - [3.5, -1.8]
        waypoints[8] = torch.tensor([-0.5, 3.5], device=self.device)   # [3.0, 1.7] - [3.5, -1.8]
        waypoints[9] = torch.tensor([-0.5, 2.4], device=self.device)   # [3.0, 0.6] - [3.5, -1.8]
        waypoints[10] = torch.tensor([-1.7, 2.4], device=self.device)  # [1.8, 0.6] - [3.5, -1.8]
        waypoints[11] = torch.tensor([-1.7, 3.5], device=self.device)  # [1.8, 1.7] - [3.5, -1.8]
        waypoints[12] = torch.tensor([-2.8, 3.5], device=self.device)  # [0.7, 1.7] - [3.5, -1.8]
        waypoints[13] = torch.tensor([-2.8, 2.4], device=self.device)  # [0.7, 0.6] - [3.5, -1.8]
        return waypoints

    def _define_waypoints_terrain_2(self):
        # Origin for terrain 2: [3.5, 3.0]
        # Waypoints (with origin subtraction pre-calculated)
        waypoints = torch.zeros((self._num_goals, 2), device=self.device)
        waypoints[0] = torch.tensor([-1.7, 0.0], device=self.device)   # [1.8, 3.0] - [3.5, 3.0]
        waypoints[1] = torch.tensor([-2.8, 0.0], device=self.device)   # [0.7, 3.0] - [3.5, 3.0]
        waypoints[2] = torch.tensor([-2.8, 1.2], device=self.device)   # [0.7, 4.2] - [3.5, 3.0]
        waypoints[3] = torch.tensor([-1.7, 1.2], device=self.device)   # [1.8, 4.2] - [3.5, 3.0]
        waypoints[4] = torch.tensor([-0.5, 1.2], device=self.device)   # [3.0, 4.2] - [3.5, 3.0]
        waypoints[5] = torch.tensor([0.7, 1.2], device=self.device)    # [4.2, 4.2] - [3.5, 3.0]
        waypoints[6] = torch.tensor([0.7, 2.4], device=self.device)    # [4.2, 5.4] - [3.5, 3.0]
        waypoints[7] = torch.tensor([0.7, 3.5], device=self.device)    # [4.2, 6.5] - [3.5, 3.0]
        waypoints[8] = torch.tensor([-0.5, 3.5], device=self.device)   # [3.0, 6.5] - [3.5, 3.0]
        waypoints[9] = torch.tensor([-0.5, 2.4], device=self.device)   # [3.0, 5.4] - [3.5, 3.0]
        waypoints[10] = torch.tensor([-1.7, 2.4], device=self.device)  # [1.8, 5.4] - [3.5, 3.0]
        waypoints[11] = torch.tensor([-1.7, 3.5], device=self.device)  # [1.8, 6.5] - [3.5, 3.0]
        waypoints[12] = torch.tensor([-2.8, 3.5], device=self.device)  # [0.7, 6.5] - [3.5, 3.0]
        waypoints[13] = torch.tensor([-2.8, 2.4], device=self.device)  # [0.7, 5.4] - [3.5, 3.0]
        return waypoints

    def _define_waypoints_terrain_3(self):
        # Origin for terrain 3: [-1.3, -6.6]
        # Waypoints (with origin subtraction pre-calculated)
        waypoints = torch.zeros((self._num_goals, 2), device=self.device)
        waypoints[0] = torch.tensor([-1.7, 0.0], device=self.device)   # [-3.0, -6.6] - [-1.3, -6.6]
        waypoints[1] = torch.tensor([-2.8, 0.0], device=self.device)   # [-4.1, -6.6] - [-1.3, -6.6]
        waypoints[2] = torch.tensor([-2.8, 1.2], device=self.device)   # [-4.1, -5.4] - [-1.3, -6.6]
        waypoints[3] = torch.tensor([-1.7, 1.2], device=self.device)   # [-3.0, -5.4] - [-1.3, -6.6]
        waypoints[4] = torch.tensor([-0.5, 1.2], device=self.device)   # [-1.8, -5.4] - [-1.3, -6.6]
        waypoints[5] = torch.tensor([0.7, 1.2], device=self.device)    # [-0.6, -5.4] - [-1.3, -6.6]
        waypoints[6] = torch.tensor([0.7, 2.4], device=self.device)    # [-0.6, -4.2] - [-1.3, -6.6]
        waypoints[7] = torch.tensor([0.7, 3.5], device=self.device)    # [-0.6, -3.1] - [-1.3, -6.6]
        waypoints[8] = torch.tensor([-0.5, 3.5], device=self.device)   # [-1.8, -3.1] - [-1.3, -6.6]
        waypoints[9] = torch.tensor([-0.5, 2.4], device=self.device)   # [-1.8, -4.2] - [-1.3, -6.6]
        waypoints[10] = torch.tensor([-1.7, 2.4], device=self.device)  # [-3.0, -4.2] - [-1.3, -6.6]
        waypoints[11] = torch.tensor([-1.7, 3.5], device=self.device)  # [-3.0, -3.1] - [-1.3, -6.6]
        waypoints[12] = torch.tensor([-2.8, 3.5], device=self.device)  # [-4.1, -3.1] - [-1.3, -6.6]
        waypoints[13] = torch.tensor([-2.8, 2.4], device=self.device)  # [-4.1, -4.2] - [-1.3, -6.6]
        return waypoints

    def _define_waypoints_terrain_4(self):
        # Origin for terrain 4: [-1.3, -1.8]
        # Waypoints (with origin subtraction pre-calculated)
        waypoints = torch.zeros((self._num_goals, 2), device=self.device)
        waypoints[0] = torch.tensor([-1.7, 0.0], device=self.device)   # [-3.0, -1.8] - [-1.3, -1.8]
        waypoints[1] = torch.tensor([-2.8, 0.0], device=self.device)   # [-4.1, -1.8] - [-1.3, -1.8]
        waypoints[2] = torch.tensor([-2.8, 1.2], device=self.device)   # [-4.1, -0.6] - [-1.3, -1.8]
        waypoints[3] = torch.tensor([-1.7, 1.2], device=self.device)   # [-3.0, -0.6] - [-1.3, -1.8]
        waypoints[4] = torch.tensor([-0.5, 1.2], device=self.device)   # [-1.8, -0.6] - [-1.3, -1.8]
        waypoints[5] = torch.tensor([0.7, 1.2], device=self.device)    # [-0.6, -0.6] - [-1.3, -1.8]
        waypoints[6] = torch.tensor([0.7, 2.4], device=self.device)    # [-0.6, 0.6] - [-1.3, -1.8]
        waypoints[7] = torch.tensor([0.7, 3.5], device=self.device)    # [-0.6, 1.7] - [-1.3, -1.8]
        waypoints[8] = torch.tensor([-0.5, 3.5], device=self.device)   # [-1.8, 1.7] - [-1.3, -1.8]
        waypoints[9] = torch.tensor([-0.5, 2.4], device=self.device)   # [-1.8, 0.6] - [-1.3, -1.8]
        waypoints[10] = torch.tensor([-1.7, 2.4], device=self.device)  # [-3.0, 0.6] - [-1.3, -1.8]
        waypoints[11] = torch.tensor([-1.7, 3.5], device=self.device)  # [-3.0, 1.7] - [-1.3, -1.8]
        waypoints[12] = torch.tensor([-2.8, 3.5], device=self.device)  # [-4.1, 1.7] - [-1.3, -1.8]
        waypoints[13] = torch.tensor([-2.8, 2.4], device=self.device)  # [-4.1, 0.6] - [-1.3, -1.8]
        return waypoints

    def _define_waypoints_terrain_5(self):
        # Origin for terrain 5: [-1.3, 3.0]
        # Waypoints (with origin subtraction pre-calculated)
        waypoints = torch.zeros((self._num_goals, 2), device=self.device)
        waypoints[0] = torch.tensor([-1.7, 0.0], device=self.device)   # [-3.0, 3.0] - [-1.3, 3.0]
        waypoints[1] = torch.tensor([-2.8, 0.0], device=self.device)   # [-4.1, 3.0] - [-1.3, 3.0]
        waypoints[2] = torch.tensor([-2.8, 1.2], device=self.device)   # [-4.1, 4.2] - [-1.3, 3.0]
        waypoints[3] = torch.tensor([-1.7, 1.2], device=self.device)   # [-3.0, 4.2] - [-1.3, 3.0]
        waypoints[4] = torch.tensor([-0.5, 1.2], device=self.device)   # [-1.8, 4.2] - [-1.3, 3.0]
        waypoints[5] = torch.tensor([0.7, 1.2], device=self.device)    # [-0.6, 4.2] - [-1.3, 3.0]
        waypoints[6] = torch.tensor([0.7, 2.4], device=self.device)    # [-0.6, 5.4] - [-1.3, 3.0]
        waypoints[7] = torch.tensor([0.7, 3.5], device=self.device)    # [-0.6, 6.5] - [-1.3, 3.0]
        waypoints[8] = torch.tensor([-0.5, 3.5], device=self.device)   # [-1.8, 6.5] - [-1.3, 3.0]
        waypoints[9] = torch.tensor([-0.5, 2.4], device=self.device)   # [-1.8, 5.4] - [-1.3, 3.0]
        waypoints[10] = torch.tensor([-1.7, 2.4], device=self.device)  # [-3.0, 5.4] - [-1.3, 3.0]
        waypoints[11] = torch.tensor([-1.7, 3.5], device=self.device)  # [-3.0, 6.5] - [-1.3, 3.0]
        waypoints[12] = torch.tensor([-2.8, 3.5], device=self.device)  # [-4.1, 6.5] - [-1.3, 3.0]
        waypoints[13] = torch.tensor([-2.8, 2.4], device=self.device)  # [-4.1, 5.4] - [-1.3, 3.0]
        return waypoints
