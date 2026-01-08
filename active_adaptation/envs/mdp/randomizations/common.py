import math
import torch
import logging
from typing import Union, TYPE_CHECKING, Dict, Tuple

import mjlab.utils.lab_api.string as string_utils
from mjlab.utils.lab_api.math import quat_apply_inverse


if TYPE_CHECKING:
    from mjlab.entity import Entity as Articulation


def uniform(low: torch.Tensor, high: torch.Tensor):
    r = torch.rand_like(low)
    return low + r * (high - low)


from active_adaptation.envs.mdp.randomizations.base import Randomization

RangeType = Tuple[float, float]
NestedRangeType = Union[RangeType, Dict[str, RangeType]]


class perturb_body_materials(Randomization):
    def __init__(
        self,
        env,
        body_names,
        static_friction_range=(0.6, 1.0),
        dynamic_friction_range=(0.6, 1.0),
        restitution_range=(0.0, 0.2),
        homogeneous: bool = False,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        if len(self.body_ids) == 0:
            raise ValueError(
                "No bodies matched the provided names for material perturbation."
            )

        self.homogeneous = homogeneous
        self.static_friction_range = tuple(static_friction_range)
        self.dynamic_friction_range = tuple(dynamic_friction_range)
        self.restitution_range = tuple(restitution_range)

        # Determine geoms that belong to the selected bodies.
        local_body_ids = torch.as_tensor(
            self.body_ids, device=self.device, dtype=torch.long
        )
        self.global_body_ids = self.asset.indexing.body_ids[local_body_ids]
        selected_body_set = set(self.global_body_ids.cpu().tolist())

        geom_global_ids = self.asset.indexing.geom_ids.cpu().tolist()
        geom_names = self.asset.geom_names
        selected_geom_local: list[int] = []
        selected_geom_global: list[int] = []
        selected_geom_names: list[str] = []

        cpu_model = self.env.sim.mj_model
        for local_idx, global_idx in enumerate(geom_global_ids):
            body_id = int(cpu_model.geom_bodyid[global_idx])
            if body_id in selected_body_set:
                selected_geom_local.append(local_idx)
                selected_geom_global.append(global_idx)
                selected_geom_names.append(geom_names[local_idx])

        if not selected_geom_global:
            raise ValueError(
                "No geoms found for the specified bodies when configuring material perturbation."
            )

        self.geom_local_ids = torch.as_tensor(
            selected_geom_local, device=self.device, dtype=torch.long
        )
        self.geom_global_ids = torch.as_tensor(
            selected_geom_global, device=self.device, dtype=torch.long
        )
        self.geom_names = selected_geom_names
        self._geom_global_ids_cpu = torch.as_tensor(
            selected_geom_global, device="cpu", dtype=torch.long
        )

        model = self.env.sim.model
        self._default_friction = model.geom_friction[:, self.geom_global_ids].clone()
        if hasattr(model, "geom_solref"):
            self._default_solref = model.geom_solref[:, self.geom_global_ids].clone()
        else:
            self._default_solref = None

    def _sample_range(
        self, range_tuple: Tuple[float, float], shape: tuple[int, int]
    ) -> torch.Tensor:
        low, high = range_tuple
        if math.isclose(low, high):
            return torch.full(shape, low, device=self.device, dtype=torch.float32)
        return torch.rand(shape, device=self.device) * (high - low) + low

    def startup(self):
        logging.info(f"Randomize body materials of {self.geom_names} upon startup.")

        num_geoms = self.geom_global_ids.numel()
        sample_cols = 1 if self.homogeneous else num_geoms
        sample_shape = (self.num_envs, sample_cols)

        static_frictions = self._sample_range(self.static_friction_range, sample_shape)
        dynamic_frictions = self._sample_range(
            self.dynamic_friction_range, sample_shape
        )
        if sample_cols == 1:
            static_frictions = static_frictions.expand(-1, num_geoms)
            dynamic_frictions = dynamic_frictions.expand(-1, num_geoms)

        model = self.env.sim.model
        geom_friction = model.geom_friction
        geom_friction[:, self.geom_global_ids, 0] = dynamic_frictions
        # TODO: mujoco do not differentiate static and dynamic friction?
        # if geom_friction.shape[-1] > 1:
        #     geom_friction[:, self.geom_global_ids, 1] = dynamic_frictions
        # if geom_friction.shape[-1] > 2:
        #     geom_friction[:, self.geom_global_ids, 2] = dynamic_frictions

        # TODO: this is not restitution
        # if self._default_solref is not None:
        #     rest_vals = self._sample_range(self.restitution_range, sample_shape)
        #     if sample_cols == 1:
        #         rest_vals = rest_vals.expand(-1, num_geoms)
        #     geom_solref = model.geom_solref
        #     solref = self._default_solref.clone()
        #     # Map restitution to the second solref parameter (damping ratio-like term).
        #     solref[..., 1] = rest_vals
        #     geom_solref[:, self.geom_global_ids] = solref

        # Sync CPU model for viewer consistency using env 0 parameters.
        cpu_model = self.env.sim.mj_model
        cpu_model.geom_friction[self._geom_global_ids_cpu.numpy(), 0] = (
            geom_friction[0, self.geom_global_ids, 0].to(device="cpu").numpy()
        )
        if geom_friction.shape[-1] > 1:
            cpu_model.geom_friction[self._geom_global_ids_cpu.numpy(), 1] = (
                geom_friction[0, self.geom_global_ids, 1].to(device="cpu").numpy()
            )
        if geom_friction.shape[-1] > 2:
            cpu_model.geom_friction[self._geom_global_ids_cpu.numpy(), 2] = (
                geom_friction[0, self.geom_global_ids, 2].to(device="cpu").numpy()
            )

        if self._default_solref is not None:
            cpu_model.geom_solref[self._geom_global_ids_cpu.numpy()] = (
                model.geom_solref[0, self.geom_global_ids].to(device="cpu").numpy()
            )


class perturb_body_mass(Randomization):
    def __init__(self, env, **perturb_ranges: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        if not perturb_ranges:
            raise ValueError(
                "perturb_body_mass requires at least one body range entry."
            )

        body_ids, body_names, values = string_utils.resolve_matching_names_values(
            perturb_ranges, self.asset.body_names
        )
        if len(body_ids) == 0:
            raise ValueError(
                "No bodies matched the provided patterns for mass perturbation."
            )

        device = self.device
        self.body_names = body_names
        self.local_body_ids = torch.as_tensor(body_ids, device=device, dtype=torch.long)
        self.global_body_ids = self.asset.indexing.body_ids[self.local_body_ids]
        self.mass_ranges = torch.as_tensor(values, device=device, dtype=torch.float32)

        # Cache default parameters to re-apply scaling from a clean baseline.
        model = self.env.sim.model
        self._default_mass = model.body_mass[:, self.global_body_ids].clone()
        self._default_inertia = model.body_inertia[:, self.global_body_ids].clone()

        # Also cache CPU model indices for synchronization when needed.
        self._global_body_ids_cpu = self.global_body_ids.to(
            device="cpu", dtype=torch.long
        )

    def startup(self):
        logging.info(f"Randomize body masses of {self.body_names} upon startup.")

        low = self.mass_ranges[:, 0]
        high = self.mass_ranges[:, 1]
        rand = torch.rand(
            self.num_envs, self.local_body_ids.numel(), device=self.device
        )
        scale = low + (high - low) * rand

        model = self.env.sim.model
        new_mass = self._default_mass * scale
        new_inertia = self._default_inertia * scale.unsqueeze(-1)

        model.body_mass[:, self.global_body_ids] = new_mass
        model.body_inertia[:, self.global_body_ids] = new_inertia

        # Keep CPU model (used for passive viewer) in sync with env 0 values.
        cpu_model = self.env.sim.mj_model
        cpu_model.body_mass[self._global_body_ids_cpu.numpy()] = (
            model.body_mass[0, self.global_body_ids].to(device="cpu").numpy()
        )
        cpu_model.body_inertia[self._global_body_ids_cpu.numpy()] = (
            model.body_inertia[0, self.global_body_ids].to(device="cpu").numpy()
        )


class perturb_body_com(Randomization):
    def __init__(self, env, body_names, com_range=(-0.05, 0.05)):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.body_ids, self.body_names = self.asset.find_bodies(body_names)
        if len(self.body_ids) == 0:
            raise ValueError(
                "No bodies matched the provided names for COM perturbation."
            )

        self.com_range = tuple(com_range)
        self.local_body_ids = torch.as_tensor(
            self.body_ids, device=self.device, dtype=torch.long
        )
        self.global_body_ids = self.asset.indexing.body_ids[self.local_body_ids]
        self._global_body_ids_cpu = self.global_body_ids.to(
            device="cpu", dtype=torch.long
        )

        model = self.env.sim.model
        self._default_body_ipos = model.body_ipos[:, self.global_body_ids].clone()

    def startup(self):
        logging.info(f"Randomize COM of bodies {self.body_names} upon startup.")

        num_bodies = self.global_body_ids.numel()
        low, high = self.com_range
        offsets = torch.rand(self.num_envs, num_bodies, 3, device=self.device)
        offsets = low + (high - low) * offsets

        model = self.env.sim.model
        new_ipos = self._default_body_ipos + offsets
        model.body_ipos[:, self.global_body_ids] = new_ipos

        cpu_model = self.env.sim.mj_model
        cpu_model.body_ipos[self._global_body_ids_cpu.numpy()] = (
            model.body_ipos[0, self.global_body_ids].to(device="cpu").numpy()
        )


class random_joint_offset(Randomization):
    def __init__(self, env, **offset_range: Tuple[float, float]):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.joint_ids, _, offset_range = string_utils.resolve_matching_names_values(
            dict(offset_range), self.asset.joint_names
        )

        self.joint_ids = torch.tensor(self.joint_ids, device=self.device)
        offset_range = torch.tensor(offset_range, device=self.device)
        self.offset_range = offset_range.unsqueeze(0).expand(self.num_envs, -1, -1)

    def reset(self, env_ids: torch.Tensor):
        offset = uniform(
            self.offset_range[env_ids, :, 0], self.offset_range[env_ids, :, 1]
        )
        self.asset.data.encoder_bias[env_ids.unsqueeze(1), self.joint_ids] = offset


from active_adaptation.envs.mdp.utils.forces import ImpulseForce, ConstantForce


class impulse(Randomization):
    def __init__(
        self,
        env,
        body_names: str = "pelvis",
        impulse_scale: Tuple[float, float, float] = (100.0, 100.0, 20.0),
        duration_range: Tuple[float, float] = (0.40, 0.60),
        impulse_prob: float = 0.005,
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        self.impulse_scale = impulse_scale
        self.duration_range = duration_range
        self.impulse_prob = impulse_prob
        self.impulse_force = self.__sample_impulse(size=self.num_envs)

        # random sample a body id
        body_ids = self.asset.find_bodies(body_names)[0]
        assert len(body_ids) == 1, "Only one body is supported"
        self.body_id = body_ids[0]
        # self.body_ids = torch.tensor(body_ids, device=self.device)
        # body_id = torch.randint(0, len(self.body_ids), (self.num_envs,), device=self.device)
        # self.body_id = self.body_ids[body_id] # shape: [num_envs]

    def __sample_impulse(self, size: int) -> ImpulseForce:
        return ImpulseForce.sample(
            size, self.device, self.impulse_scale, self.duration_range
        )

    def step(self, substep):
        external_wrench = self.asset.data.body_external_wrench
        forces_w, torques_w = external_wrench[..., 0:3], external_wrench[..., 3:6]
        impulse_force = self.impulse_force.get_force()
        forces_w[:, self.body_id] += impulse_force
        self.asset.write_external_wrench_to_sim(forces_w, torques_w)

    def update(self):
        expire = self.impulse_force.time > self.impulse_force.duration
        r = torch.rand(self.num_envs, 1, device=self.device) < self.impulse_prob
        sample = r & expire

        impulse_force = self.__sample_impulse(size=self.num_envs)

        self.impulse_force.time.add_(self.env.step_dt)
        self.impulse_force: ImpulseForce = impulse_force.where(
            sample, self.impulse_force
        )


class constant_force(Randomization):
    def __init__(
        self, env, force_range, offset_range, body_names=None, duration_range=(1.0, 4.0)
    ):
        super().__init__(env)
        self.asset: Articulation = self.env.scene["robot"]
        if body_names is None:
            self.all_body_ids = torch.tensor([0], device=self.device)
        else:
            self.all_body_ids = torch.tensor(
                self.asset.find_bodies(body_names)[0], device=self.device
            )

        self.force = ConstantForce.sample(self.num_envs, device=self.device)
        self.force.duration.zero_()
        self.body_id = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.resample_interval = 50
        self.resample_prob = 0.2

        self.force_range = torch.tensor(force_range, device=self.device)
        self.offset_range = torch.tensor(offset_range, device=self.device)
        self.duration_range = torch.tensor(duration_range, device=self.device)

        self.arange = torch.arange(self.num_envs, device=self.device)

    def step(self, substep):
        arange = self.arange
        quat = self.asset.data.body_link_quat_w[arange, self.body_id]
        forces_b = quat_apply_inverse(
            quat.reshape(self.num_envs, 4), self.force.get_force()
        )
        self.asset._external_force_b[arange, self.body_id] += forces_b
        self.asset._external_torque_b[arange, self.body_id] += self.force.offset.cross(
            forces_b, dim=-1
        )
        self.asset.has_external_wrench = True

    def reset(self, env_ids: torch.Tensor):
        self.force.duration.data[env_ids] = 0.0

    def update(self):
        resample = self.env.episode_length_buf % self.resample_interval == 0
        expired = self.force.time > self.force.duration
        resample = (
            resample
            & expired.squeeze(-1)
            & (torch.rand(self.num_envs, device=self.device) < self.resample_prob)
        )
        force = ConstantForce.sample(
            self.num_envs,
            self.force_range,
            self.offset_range,
            self.duration_range,
            self.device,
        )
        self.force.time.add_(self.env.step_dt)
        self.force = force.where(resample, self.force)
        body_id = self.all_body_ids[
            torch.randint(
                0, len(self.all_body_ids), (self.num_envs,), device=self.device
            )
        ]
        self.body_id = torch.where(resample, body_id, self.body_id)
