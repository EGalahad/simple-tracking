import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase as ModBase
from typing import TYPE_CHECKING
from mjlab.actuator import BuiltinPositionActuatorCfg

if TYPE_CHECKING:
    from mjlab.entity import Entity


def get_asset_meta(asset: "Entity"):
    meta, joint_kp, joint_kd = {}, {}, {}
    meta["default_joint_pos"] = asset.cfg.init_state.joint_pos.copy()
    meta["joint_names"] = asset.joint_names

    for actuator in asset.actuators:
        assert isinstance(actuator.cfg, BuiltinPositionActuatorCfg)
        for joint_name in actuator.cfg.target_names_expr:
            joint_kp.update({joint_name: actuator.cfg.stiffness})
        for joint_name in actuator.cfg.target_names_expr:
            joint_kd.update({joint_name: actuator.cfg.damping})

    meta["joint_kp"] = joint_kp
    meta["joint_kd"] = joint_kd
    return meta


@torch.inference_mode()
def export_onnx(module: ModBase, td: TensorDictBase, path: str, meta=None):
    if not path.endswith(".onnx"):
        raise ValueError(f"Export path must end with .onnx, got {path}.")

    td = td.cpu().select(*module.in_keys, strict=True)
    module = module.cpu()
    print(torch.__version__)
    # breakpoint()
    onnx_program = torch.onnx.dynamo_export(module, **td.to_dict())
    onnx_program.save(path)
    print(f"Exported ONNX model to {path}.")

    import json

    meta_path = path.replace(".onnx", ".json")
    if meta is None:
        meta = {}
    meta["in_keys"] = module.in_keys
    meta["out_keys"] = module.out_keys
    meta["in_shapes"] = ([td[k].shape for k in module.in_keys],)

    json.dump(meta, open(meta_path, "w"), indent=4)
    print(f"Exported metadata to {meta_path}.")

    import onnxruntime as ort

    ort_session = ort.InferenceSession(
        path.replace(".pt", ".onnx"), providers=["CPUExecutionProvider"]
    )

    def to_numpy(tensor):
        return (
            tensor.detach().cpu().numpy()
            if tensor.requires_grad
            else tensor.cpu().numpy()
        )

    onnx_input = tuple(td[k] for k in module.in_keys)
    onnxruntime_input = {
        k.name: to_numpy(v) for k, v in zip(ort_session.get_inputs(), onnx_input)
    }

    ort_output = ort_session.run(None, onnxruntime_input)
    assert len(ort_output) == len(module.out_keys)
