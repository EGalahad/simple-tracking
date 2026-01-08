from active_adaptation.envs.tasks.hdmi.command import RobotObjectTracking
from active_adaptation.envs.mdp.randomizations.base import (
    Randomization as BaseRandomization,
)

import torch
from typing import Tuple, TYPE_CHECKING
from mjlab.utils.lab_api.math import sample_uniform
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity


RobotObjectTrackRandomization = BaseRandomization[RobotObjectTracking]


class object_body_randomization(RobotObjectTrackRandomization):
    def __init__(
        self,
        dynamic_friction_range: Tuple[float, float] = (0.6, 1.0),
        restitution_range: Tuple[float, float] = (0.0, 0.2),
        mass_range: Tuple[float, float] = (1.0, 10.0),
        static_friction_range: Tuple[float, float] | None = None,
        static_dynamic_friction_ratio_range: Tuple[float, float] | None = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if (
            static_friction_range is not None
            and static_dynamic_friction_ratio_range is not None
        ):
            raise ValueError(
                "Cannot specify both static_friction_range and static_dynamic_friction_ratio_range"
            )
        if (
            static_friction_range is None
            and static_dynamic_friction_ratio_range is None
        ):
            raise ValueError(
                "Must specify either static_friction_range or static_dynamic_friction_ratio_range"
            )

        self.object = self.command_manager.object
        self.object_name = self.command_manager.object_asset_name
        self.mass_range = mass_range
        self.dynamic_friction_range = dynamic_friction_range
        self.static_friction_range = static_friction_range
        self.static_dynamic_friction_ratio_range = static_dynamic_friction_ratio_range
        self.restitution_range = restitution_range
        self.asset_cfg = SceneEntityCfg(name=self.object_name)

    def startup(self):
        self.env.sim.expand_model_fields(("body_mass", "body_inertia", "geom_friction"))
        if hasattr(self.env.sim.model, "geom_solref"):
            self.env.sim.expand_model_fields(("geom_solref",))

        entity: Entity = self.env.scene[self.object_name]
        if isinstance(self.asset_cfg.body_ids, list):
            local_body_ids = torch.as_tensor(
                self.asset_cfg.body_ids, device=self.device
            )
            global_body_ids = entity.indexing.body_ids[local_body_ids]
        else:
            global_body_ids = entity.indexing.body_ids

        if isinstance(self.asset_cfg.geom_ids, list):
            local_geom_ids = torch.as_tensor(
                self.asset_cfg.geom_ids, device=self.device
            )
            global_geom_ids = entity.indexing.geom_ids[local_geom_ids]
        else:
            global_geom_ids = entity.indexing.geom_ids

        num_envs = self.env.num_envs
        num_bodies = global_body_ids.numel()
        num_geoms = global_geom_ids.numel()

        default_body_mass = self.env.sim.get_default_field("body_mass")[global_body_ids]
        default_body_inertia = self.env.sim.get_default_field("body_inertia")[
            global_body_ids
        ]

        new_mass = sample_uniform(*self.mass_range, (num_envs, num_bodies), self.device)
        mass_scale = new_mass / default_body_mass.unsqueeze(0)

        self.env.sim.model.body_mass[:, global_body_ids] = new_mass
        self.env.sim.model.body_inertia[:, global_body_ids] = (
            default_body_inertia.unsqueeze(0) * mass_scale.unsqueeze(-1)
        )

        dynamic_friction = sample_uniform(
            *self.dynamic_friction_range, (num_envs, num_geoms), self.device
        )
        if self.static_friction_range is not None:
            static_friction = sample_uniform(
                *self.static_friction_range, (num_envs, num_geoms), self.device
            )
        else:
            ratio = sample_uniform(
                *self.static_dynamic_friction_ratio_range,
                (num_envs, num_geoms),
                self.device,
            )
            static_friction = dynamic_friction * ratio
        friction = torch.maximum(static_friction, dynamic_friction)

        default_geom_friction = self.env.sim.get_default_field("geom_friction")[
            global_geom_ids
        ]
        geom_friction = (
            default_geom_friction.unsqueeze(0).expand(num_envs, -1, -1).clone()
        )
        geom_friction[..., 0] = friction
        self.env.sim.model.geom_friction[:, global_geom_ids] = geom_friction

        if hasattr(self.env.sim.model, "geom_solref"):
            default_solref = self.env.sim.get_default_field("geom_solref")[
                global_geom_ids
            ]
            solref = default_solref.unsqueeze(0).expand(num_envs, -1, -1).clone()
            restitution = sample_uniform(
                *self.restitution_range, (num_envs, num_geoms), self.device
            )
            solref[..., 1] = (1.0 - restitution).clamp_min(0.0)
            self.env.sim.model.geom_solref[:, global_geom_ids] = solref
