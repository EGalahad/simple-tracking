from active_adaptation.envs.mdp.observations.base import Observation

import torch
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from mjlab.sensor import ContactSensor


def random_noise(x: torch.Tensor, std: float):
    return x + torch.randn_like(x).clamp(-3.0, 3.0) * std


class root_ang_vel_history(Observation):
    def __init__(self, env, noise_std: float = 0.0, history_steps: list[int] = [1]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()

    def reset(self, env_ids):
        root_ang_vel_b = self.asset.data.root_com_ang_vel_b[env_ids]
        root_ang_vel_b = root_ang_vel_b.unsqueeze(1).expand(
            -1, self.buffer.shape[1], -1
        )
        if self.noise_std > 0:
            root_ang_vel_b = random_noise(root_ang_vel_b, self.noise_std)
        self.buffer[env_ids] = root_ang_vel_b

    def update(self):
        root_ang_vel_b = self.asset.data.root_com_ang_vel_b
        if self.noise_std > 0:
            root_ang_vel_b = random_noise(root_ang_vel_b, self.noise_std)
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = root_ang_vel_b

    def compute(self) -> torch.Tensor:
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)


class projected_gravity_history(Observation):
    def __init__(self, env, noise_std: float = 0.0, history_steps: list[int] = [1]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.noise_std = noise_std
        self.history_steps = history_steps
        buffer_size = max(history_steps) + 1
        self.buffer = torch.zeros((self.num_envs, buffer_size, 3), device=self.device)
        self.update()

    def reset(self, env_ids):
        projected_gravity_b = self.asset.data.projected_gravity_b[env_ids]
        projected_gravity_b = projected_gravity_b.unsqueeze(1).expand(
            -1, self.buffer.shape[1], -1
        )
        if self.noise_std > 0:
            projected_gravity_b = random_noise(projected_gravity_b, self.noise_std)
            projected_gravity_b = projected_gravity_b / projected_gravity_b.norm(
                dim=-1, keepdim=True
            )
        self.buffer[env_ids] = self.asset.data.projected_gravity_b[env_ids].unsqueeze(1)

    def update(self):
        projected_gravity_b = self.asset.data.projected_gravity_b
        if self.noise_std > 0:
            projected_gravity_b = random_noise(projected_gravity_b, self.noise_std)
            projected_gravity_b = projected_gravity_b / projected_gravity_b.norm(
                dim=-1, keepdim=True
            )
        self.buffer = self.buffer.roll(1, dims=1)
        self.buffer[:, 0] = projected_gravity_b

    def compute(self):
        return self.buffer[:, self.history_steps].reshape(self.num_envs, -1)


class joint_pos_history(Observation):
    def __init__(
        self,
        env,
        joint_names: str = ".*",
        history_steps: list[int] = [0],
        noise_std: float = 0.0,
        set_to_zero_joint_names: str | None = None,
    ):
        super().__init__(env)
        self.history_steps = history_steps
        self.buffer_size = max(history_steps) + 1
        self.noise_std = max(noise_std, 0.0)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)
        self.num_joints = len(self.joint_ids)
        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        self.joint_mask = torch.ones(self.num_joints, device=self.device)
        if set_to_zero_joint_names is not None:
            from mjlab.utils.lab_api.string import resolve_matching_names

            set_to_zero_joint_ids, _ = resolve_matching_names(
                set_to_zero_joint_names, self.joint_names
            )
            self.joint_mask[set_to_zero_joint_ids] = 0.0
        self.joint_mask = self.joint_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, J]

        shape = (self.num_envs, self.buffer_size, self.num_joints)
        self.buffer = torch.zeros(shape, device=self.device)

    def reset(self, env_ids):
        joint_pos = self.asset.data.joint_pos[
            env_ids.unsqueeze(1), self.joint_ids.unsqueeze(0)
        ]
        self.buffer[env_ids] = joint_pos.unsqueeze(1)

    def update(self):
        self.buffer = self.buffer.roll(1, 1)
        joint_pos = self.asset.data.joint_pos[:, self.joint_ids]
        if self.noise_std > 0:
            joint_pos = random_noise(joint_pos, self.noise_std)
        self.buffer[:, 0] = joint_pos

    def compute(self):
        joint_pos = self.buffer - self.asset.data.encoder_bias[
            :, self.joint_ids
        ].unsqueeze(1)
        joint_pos = joint_pos * self.joint_mask
        joint_pos_selected = joint_pos[:, self.history_steps]
        return joint_pos_selected.reshape(self.num_envs, -1)


class prev_actions(Observation):
    def __init__(
        self, env, steps: int = 1, flatten: bool = True, permute: bool = False
    ):
        super().__init__(env)
        self.steps = steps
        self.flatten = flatten
        self.permute = permute
        self.action_manager = self.env.action_manager

    def compute(self):
        action_buf = self.action_manager.action_buf[:, :, : self.steps].clone()
        if self.permute:
            action_buf = action_buf.permute(0, 2, 1)
        if self.flatten:
            return action_buf.reshape(self.num_envs, -1)
        else:
            return action_buf

    def symmetry_transforms(self):
        assert self.permute
        transform = self.action_manager.symmetry_transforms()
        return transform.repeat(self.steps)


class applied_action(Observation):
    def __init__(self, env):
        super().__init__(env)
        self.action_manager = self.env.action_manager

    def compute(self) -> torch.Tensor:
        return self.action_manager.applied_action

    def symmetry_transforms(self):
        transform = self.action_manager.symmetry_transforms()
        return transform


class applied_torque(Observation):
    def __init__(self, env, joint_names: str = ".*"):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, self.joint_names = self.asset.find_joints(joint_names)

    def compute(self) -> torch.Tensor:
        applied_efforts = self.asset.data.actuator_force
        return applied_efforts[:, self.joint_ids]


class last_contact(Observation):
    def __init__(self, env, body_names: str):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]

        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)

        with torch.device(self.device):
            self.body_ids = torch.as_tensor(self.body_ids)
            self.has_contact = torch.zeros(
                self.num_envs, len(self.body_ids), 1, dtype=bool
            )
            self.last_contact_pos_w = torch.zeros(self.num_envs, len(self.body_ids), 3)
        self.body_link_pos_w = self.asset.data.body_link_pos_w[
            :, self.articulation_body_ids
        ]

    def reset(self, env_ids: torch.Tensor):
        self.has_contact[env_ids] = False

    def update(self):
        first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)[
            :, self.body_ids
        ].unsqueeze(-1)
        self.has_contact.logical_or_(first_contact)
        self.body_link_pos_w = self.asset.data.body_link_pos_w[
            :, self.articulation_body_ids
        ]
        self.last_contact_pos_w = torch.where(
            first_contact, self.body_link_pos_w, self.last_contact_pos_w
        )

    def compute(self):
        distance_xy = (
            self.body_link_pos_w[:, :, :2] - self.last_contact_pos_w[:, :, :2]
        ).norm(dim=-1)
        distance_z = self.body_link_pos_w[:, :, 2] - self.last_contact_pos_w[:, :, 2]
        distance = torch.stack([distance_xy, distance_z], dim=-1)
        return (distance * self.has_contact).reshape(self.num_envs, -1)

    def debug_draw(self):
        return


class random_noise_placeholder(Observation):
    def __init__(self, env, dim: int, noise_std: float = 1.0):
        self.noise_std = noise_std
        super().__init__(env)
        self.dim = dim

    def compute(self) -> torch.Tensor:
        return (
            torch.randn(self.num_envs, self.dim, device=self.device).clamp(-3, 3)
            * self.noise_std
        )
