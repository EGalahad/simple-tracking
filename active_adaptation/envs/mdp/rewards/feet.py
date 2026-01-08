from active_adaptation.envs.mdp.rewards.base import Reward

import torch

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation
    from mjlab.sensor import ContactSensor
    from mjlab.viewer.debug_visualizer import DebugVisualizer


class feet_slip(Reward):
    def __init__(
        self,
        env,
        body_names: str,
        weight: float,
        tolerance: float = 0.0,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.articulation_body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.contact_sensor: "ContactSensor" = self.env.scene["feet_ground_contact"]
        self.tolerance = tolerance

    def compute(self) -> torch.Tensor:
        in_contact = self.contact_sensor.data.current_contact_time > 0.02
        feet_vel = self.asset.data.body_com_lin_vel_w[:, self.articulation_body_ids, :2]
        feet_vel = (feet_vel.norm(dim=-1) - self.tolerance).clamp(min=0.0, max=1.0)
        slip = (in_contact * feet_vel).sum(dim=1, keepdim=True)
        return -slip


class feet_stumble(Reward):
    def __init__(
        self, env, body_names: str | List[str], weight: float, enabled: bool = True
    ):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_forces: ContactSensor = self.env.scene["contact_forces"]
        self.feet_contact_ids = self.contact_forces.find_bodies(body_names)[0]

    def compute(self) -> torch.Tensor:
        in_contact = self.contact_forces.data.force[:, :, :2].norm(dim=2) > 0.5
        return -in_contact.float().mean(1, keepdim=True)


class feet_air_time(Reward):
    def __init__(
        self,
        env,
        body_names: str,
        thres: float,
        weight: float,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.thres = thres
        self.asset: Articulation = self.env.scene["robot"]

        self.articulation_body_ids, self.body_names = self.asset.find_bodies(body_names)
        self.articulation_body_ids = list(reversed(self.articulation_body_ids))
        self.contact_sensor: "ContactSensor" = self.env.scene["feet_ground_contact"]
        num_bodies = self.contact_sensor.data.found.shape[1]
        self.in_contact_last = torch.zeros(
            self.num_envs, num_bodies, dtype=bool, device=self.env.device
        )

    def reset(self, env_ids):
        self.in_contact_last.index_fill_(0, env_ids, False)

    def compute(self):
        in_contact_this = self.contact_sensor.data.found > 0
        self.first_contact = (~in_contact_this) & self.in_contact_last
        self.in_contact_last[:] = in_contact_this

        last_air_time = self.contact_sensor.data.last_air_time
        reward = torch.sum(
            (last_air_time - self.thres).clamp_max(0.0) * self.first_contact,
            dim=1,
            keepdim=True,
        )
        reward *= ~self.env.command_manager.is_standing_env
        return reward

    def debug_draw(self):
        visualizer: "DebugVisualizer" = getattr(self.env, "visualizer", None)
        if visualizer is None:
            return
        env_idx = visualizer.env_idx
        feet_pos_w = self.asset.data.body_link_pos_w[
            env_idx, self.articulation_body_ids
        ]

        in_contact = self.first_contact[env_idx]
        # for i in range(2):
        #     if in_contact[i]:
        #         print(f"[DEBUG]: Foot {i} first contact (manual)")

        # first_contact = self.contact_sensor.compute_first_contact(self.env.step_dt)
        # in_contact = first_contact[env_idx]
        # for i in range(2):
        #     if in_contact[i]:
        #         print(f"[DEBUG]: Foot {i} first contact (sensor)")

        contact_pos_w = feet_pos_w[in_contact]
        radius = 2.0 * visualizer.meansize
        color = (1.0, 0.2, 0.2, 0.8)
        for i in range(contact_pos_w.shape[0]):
            visualizer.add_sphere(center=contact_pos_w[i], radius=radius, color=color)


class feet_air_time_mjlab(Reward):
    def __init__(
        self,
        env,
        min_thres: float,
        max_thres: float,
        weight: float,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.min_thres = min_thres
        self.max_thres = max_thres
        self.asset: Articulation = self.env.scene["robot"]

        self.contact_sensor: "ContactSensor" = self.env.scene.sensors[
            "feet_ground_contact"
        ]

    def compute(self):
        current_air_time = self.contact_sensor.data.current_air_time
        reward = (current_air_time > self.min_thres) & (
            current_air_time < self.max_thres
        )
        reward = reward.float().mean(dim=1, keepdim=True)
        reward *= ~self.env.command_manager.is_standing_env
        return reward

    def debug_draw(self):
        # draw a sphere at each foot, the size of the sphere is proportional to the air time
        visualizer: "DebugVisualizer" = getattr(self.env, "visualizer", None)
        if visualizer is None:
            return
        env_idx = visualizer.env_idx
        feet_ids = self.asset.find_bodies(".*ankle_roll_link")[0]
        feet_pos_w = self.asset.data.body_link_pos_w[env_idx, feet_ids]
        current_air_time = self.contact_sensor.data.current_air_time[env_idx]
        # size = 2.0 * visualizer.meansize * (current_air_time / self.max_thres)
        for i in range(len(feet_ids)):
            size = 2.0 * visualizer.meansize * (current_air_time[i] / self.max_thres)
            size = size.item()
            if self.max_thres > current_air_time[i] < self.min_thres:
                color = (0.2, 0.2, 1.0, 0.8)
            else:
                color = (1.0, 0.2, 0.2, 0.8)
            visualizer.add_sphere(center=feet_pos_w[i], radius=size, color=color)


class max_feet_height(Reward):
    def __init__(
        self,
        env,
        body_names: str,
        target_height: float,
        weight: float,
        enabled: bool = True,
    ):
        super().__init__(env, weight, enabled)
        self.target_height = target_height

        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.device)

        self.asset_body_ids, self.asset_body_names = self.asset.find_bodies(body_names)

        self.in_contact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=bool, device=self.device
        )
        self.impact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=bool, device=self.device
        )
        self.detach = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=bool, device=self.device
        )
        self.has_impact = torch.zeros(
            self.num_envs, len(self.body_ids), dtype=bool, device=self.device
        )
        self.max_height = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.device
        )
        self.impact_point = torch.zeros(
            self.num_envs, len(self.body_ids), 3, device=self.device
        )
        self.detach_point = torch.zeros(
            self.num_envs, len(self.body_ids), 3, device=self.device
        )

    def reset(self, env_ids):
        self.has_impact[env_ids] = False

    def update(self):
        contact_force = self.contact_sensor.data.net_forces_w_history[
            :, :, self.body_ids
        ]
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.asset_body_ids]
        in_contact = (contact_force.norm(dim=-1) > 0.01).any(dim=1)
        self.impact[:] = (~self.in_contact) & in_contact
        self.detach[:] = self.in_contact & (~in_contact)
        self.in_contact[:] = in_contact
        self.has_impact.logical_or_(self.impact)
        self.impact_point[self.impact] = feet_pos_w[self.impact]
        self.detach_point[self.detach] = feet_pos_w[self.detach]
        self.max_height[:] = torch.where(
            self.detach,
            feet_pos_w[:, :, 2],
            torch.maximum(self.max_height, feet_pos_w[:, :, 2]),
        )

    def compute(self) -> torch.Tensor:
        reference_height = torch.maximum(
            self.impact_point[:, :, 2], self.detach_point[:, :, 2]
        )
        max_height = self.max_height - reference_height
        # r = (self.impact * (max_height / self.target_height).clamp_max(1.0)).sum(
        #     dim=1, keepdim=True
        # )
        # this should be penalty, otherwise encourages the feet to contact more often
        penalty = self.impact * (1 - max_height / self.target_height).clamp_min(0.0)
        r = -penalty.sum(dim=1, keepdim=True)
        is_standing = self.env.command_manager.is_standing_env.squeeze(1)
        # sometimes the policy can decied is_standing, so we need to set the mean reward to 0
        # r[~is_standing] -= r[~is_standing].mean()
        r[is_standing] = 0
        return r

    def debug_draw(self):
        feet_pos_w = self.asset.data.body_link_pos_w[:, self.asset_body_ids]
        self.env.debug_draw.point(
            feet_pos_w[self.impact],
            color=(1.0, 0.0, 0.0, 1.0),
            size=30,
        )


class feet_contact_count(Reward):
    def __init__(self, env, body_names: str, weight: float, enabled: bool = True):
        super().__init__(env, weight, enabled)
        self.asset: Articulation = self.env.scene["robot"]
        self.contact_sensor: ContactSensor = self.env.scene["contact_forces"]

        self.articulation_body_ids = self.asset.find_bodies(body_names)[0]
        self.body_ids, self.body_names = self.contact_sensor.find_bodies(body_names)
        self.body_ids = torch.tensor(self.body_ids, device=self.env.device)
        self.first_contact = torch.zeros(
            self.num_envs, len(self.body_ids), device=self.env.device
        )

    def compute(self):
        self.first_contact[:] = self.contact_sensor.compute_first_contact(
            self.env.step_dt
        )[:, self.body_ids]
        return self.first_contact.sum(1, keepdim=True)
