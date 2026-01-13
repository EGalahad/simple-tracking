import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as sRot


UNITREE_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

UNITREE_BODY_NAMES = [
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
]


def select_in_order(original, whitelist):
    original_list = list(original)
    index_map = {}
    for i, v in enumerate(original_list):
        if v not in index_map:
            index_map[v] = i
    selected = [x for x in whitelist if x in index_map]
    indices = [index_map[x] for x in selected]
    return selected, indices


def pad_data(data, pad_before, pad_after):
    if pad_before == 0 and pad_after == 0:
        return data
    slot = np.zeros(
        (pad_before + pad_after + data.shape[0], *data.shape[1:]), dtype=data.dtype
    )
    slot[pad_before : pad_before + data.shape[0]] = data
    if pad_before > 0:
        slot[:pad_before] = slot[pad_before : pad_before + 1]
    if pad_after > 0:
        slot[-pad_after:] = slot[-pad_after - 1 : -pad_after]
    return slot


def finite_difference(data, fps):
    vel = np.zeros_like(data, dtype=np.float32)
    if data.shape[0] >= 2:
        vel[1:-1] = (data[2:] - data[:-2]) * (fps / 2.0)
        vel[0] = (data[1] - data[0]) * fps
        vel[-1] = (data[-1] - data[-2]) * fps
    return vel


def angvel_from_rot(quat, fps, quat_order="wxyz"):
    if fps <= 0:
        raise ValueError("fps must be positive.")
    arr = np.asarray(quat)
    if arr.ndim != 2 or arr.shape[1] != 4:
        raise ValueError("quat must have shape (T, 4)")

    quat_xyzw = arr.astype(np.float64)
    if quat_order.lower() == "wxyz":
        quat_xyzw = np.concatenate([quat_xyzw[:, 1:], quat_xyzw[:, :1]], axis=-1)
    elif quat_order.lower() != "xyzw":
        raise ValueError("quat_order must be 'xyzw' or 'wxyz'.")

    if quat_xyzw.shape[0] == 1:
        return np.zeros((1, 3), dtype=np.float32)

    q_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)
    q_wxyz /= np.linalg.norm(q_wxyz, axis=1, keepdims=True).clip(min=1e-12)

    dots = np.sum(q_wxyz[1:] * q_wxyz[:-1], axis=1)
    flip_idx = np.where(dots < 0)[0] + 1
    if flip_idx.size > 0:
        q_wxyz[flip_idx] *= -1.0

    qdot = np.zeros_like(q_wxyz)
    qdot[1:-1] = (q_wxyz[2:] - q_wxyz[:-2]) * (fps / 2.0)
    qdot[0] = (q_wxyz[1] - q_wxyz[0]) * fps
    qdot[-1] = (q_wxyz[-1] - q_wxyz[-2]) * fps

    def qmul_wxyz(a, b):
        aw, ax, ay, az = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
        bw, bx, by, bz = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
        return np.stack(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            axis=-1,
        )

    q_conj = q_wxyz.copy()
    q_conj[:, 1:] *= -1.0
    omega_quat = qmul_wxyz(qdot, q_conj) * 2.0
    return omega_quat[:, 1:].astype(np.float32)


def preprocess_motion(root_pos, body_pos_w, body_names, always_on_ground):
    offset_xy = root_pos[0, :2].copy()
    root_pos[:, 0] -= offset_xy[0]
    root_pos[:, 1] -= offset_xy[1]
    body_pos_w[:, :, 0] -= offset_xy[0]
    body_pos_w[:, :, 1] -= offset_xy[1]

    foot_names = ["left_ankle_roll_link", "right_ankle_roll_link"]
    if not all(name in body_names for name in foot_names):
        return root_pos, body_pos_w

    foot_idx = [body_names.index(n) for n in foot_names]
    z_l = body_pos_w[:, foot_idx[0], 2]
    z_r = body_pos_w[:, foot_idx[1], 2]

    if always_on_ground:
        z_min = np.min(np.stack([z_l, z_r], axis=1), axis=1)
        dz = -z_min
        root_pos[:, 2] += dz
        body_pos_w[:, :, 2] += dz[:, None]
    else:
        z_min = float(min(z_l.min(), z_r.min()))
        dz = -z_min
        root_pos[:, 2] += dz
        body_pos_w[:, :, 2] += dz
    return root_pos, body_pos_w


def _default_relpath(dataset_root: Path, p: Path) -> str:
    if dataset_root.is_file():
        return p.name
    try:
        return str(p.relative_to(dataset_root))
    except ValueError:
        return str(p)


def load_allowlist(path: str):
    payload = json.loads(Path(path).read_text())
    params = payload.get("params")
    if not isinstance(params, dict):
        raise ValueError("allowlist json missing required object key: params")
    segs = payload.get("segments", {})
    out = set()
    for fname, spans in segs.items():
        for start, end in spans:
            out.add((str(fname), int(start), int(end)))
    required_params = ["target_fps", "pad_before", "pad_after", "segment_len"]
    missing = [k for k in required_params if k not in params]
    if missing:
        raise ValueError(f"allowlist json missing required params keys: {missing}")
    return out, {
        "target_fps": int(params["target_fps"]),
        "pad_before": int(params["pad_before"]),
        "pad_after": int(params["pad_after"]),
        "segment_len": int(params["segment_len"]),
    }


def make_allowlist_filter(allow, dataset_root: Path):
    def _filter(p, start_idx, end_idx):
        rel = _default_relpath(dataset_root, Path(p))
        return (rel, int(start_idx), int(end_idx)) in allow

    return _filter


def compute_body_ang_vel(body_quat_w, fps):
    T, B, _ = body_quat_w.shape
    out = np.zeros((T, B, 3), dtype=np.float32)
    for b in range(B):
        out[:, b] = angvel_from_rot(body_quat_w[:, b], fps, quat_order="wxyz")
    return out


def process_motion(
    m,
    joint_names_keep,
    body_names_keep,
    pad_before,
    pad_after,
    target_fps,
    always_on_ground,
):
    joint_names = m["joint_names"].tolist()
    body_names = m["body_names"].tolist()
    sel_joint_names, sel_joint_idx = select_in_order(joint_names, joint_names_keep)
    sel_body_names, sel_body_idx = select_in_order(body_names, body_names_keep)

    if not sel_joint_names:
        raise RuntimeError("No joint names matched the configured whitelist.")
    if not sel_body_names:
        raise RuntimeError("No body names matched the configured whitelist.")

    root_pos = pad_data(m["root_pos"], pad_before, pad_after).astype(np.float32)
    root_rot_xyzw = pad_data(m["root_rot"], pad_before, pad_after).astype(np.float32)
    dof_pos = pad_data(m["dof_pos"][:, sel_joint_idx], pad_before, pad_after).astype(
        np.float32
    )
    local_body_pos = pad_data(
        m["local_body_pos"][:, sel_body_idx, :], pad_before, pad_after
    ).astype(np.float32)
    local_body_rot_xyzw = pad_data(
        m["local_body_rot"][:, sel_body_idx, :], pad_before, pad_after
    ).astype(np.float32)

    src_fps = int(m.get("mocap_framerate", m.get("frequency", m.get("fps", 0))))
    if src_fps and src_fps != target_fps:
        print(f"Warning: source fps {src_fps} != target fps {target_fps}")

    T = root_pos.shape[0]
    R_root = sRot.from_quat(root_rot_xyzw)
    R_root_m = R_root.as_matrix()
    R_local_m = (
        sRot.from_quat(local_body_rot_xyzw.reshape(-1, 4))
        .as_matrix()
        .reshape(T, -1, 3, 3)
    )
    body_pos_w = (
        np.einsum("tij,tbj->tbi", R_root_m, local_body_pos) + root_pos[:, None, :]
    )
    R_world_m = np.einsum("tij,tbjk->tbik", R_root_m, R_local_m)
    body_quat_w_xyzw = (
        sRot.from_matrix(R_world_m.reshape(-1, 3, 3)).as_quat().reshape(T, -1, 4)
    )
    body_quat_w = np.concatenate(
        [body_quat_w_xyzw[..., 3:4], body_quat_w_xyzw[..., :3]], axis=-1
    )

    root_pos, body_pos_w = preprocess_motion(
        root_pos, body_pos_w, sel_body_names, always_on_ground=always_on_ground
    )

    body_lin_vel_w = finite_difference(body_pos_w, target_fps)
    body_ang_vel_w = compute_body_ang_vel(body_quat_w, target_fps)
    joint_vel = finite_difference(dof_pos, target_fps)

    return {
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
        "joint_pos": dof_pos,
        "joint_vel": joint_vel,
        "fps": float(target_fps),
        "body_names": sel_body_names,
        "joint_names": sel_joint_names,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-root", required=True, help="AMASS npz root (file or directory)"
    )
    ap.add_argument(
        "--allowlist", default=None, help="Allowlist json from precompute step"
    )
    ap.add_argument(
        "--target-fps",
        type=int,
        default=None,
        help="Target fps when allowlist is not provided",
    )
    ap.add_argument(
        "--output-dir", required=True, help="Output directory for motion segments"
    )
    ap.add_argument("--no-always-on-ground", action="store_true", default=False)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.allowlist:
        if args.target_fps is not None:
            print("Warning: --target-fps is ignored when --allowlist is set")
        allow, params = load_allowlist(args.allowlist)
        allow_filter = make_allowlist_filter(allow, dataset_root)
    else:
        params = {
            "target_fps": None if args.target_fps is None else int(args.target_fps),
            "pad_before": 0,
            "pad_after": 0,
        }

    if dataset_root.is_file() and dataset_root.suffix == ".npz":
        paths = [dataset_root]
    else:
        paths = list(dataset_root.rglob("*.npz"))
    if not paths:
        raise RuntimeError(f"No motions found in {dataset_root}")

    meta_payload = None

    for p in paths:
        m = dict(np.load(p, allow_pickle=True))
        target_fps = params["target_fps"]
        if target_fps is None:
            target_fps = int(
                m.get("mocap_framerate", m.get("frequency", m.get("fps", 0)))
            )
            if target_fps <= 0:
                raise ValueError(f"Missing target fps for {p}; pass --target-fps")

        motion = process_motion(
            m,
            UNITREE_JOINT_NAMES,
            UNITREE_BODY_NAMES,
            params["pad_before"],
            params["pad_after"],
            target_fps,
            always_on_ground=not args.no_always_on_ground,
        )

        meta_candidate = {
            "body_names": motion["body_names"],
            "joint_names": motion["joint_names"],
            "fps": motion["fps"],
        }
        if meta_payload is None:
            meta_payload = meta_candidate
        elif meta_payload != meta_candidate:
            raise RuntimeError(f"meta.json mismatch for {p}")

        rel = _default_relpath(dataset_root, p)
        rel_base = Path(rel).with_suffix("")

        if args.allowlist:
            T = motion["joint_pos"].shape[0]
            for start_idx in range(0, T, params["segment_len"]):
                end_idx = min(start_idx + params["segment_len"], T)
                if not allow_filter(p, start_idx, end_idx):
                    continue

                seg_dir = output_dir / rel_base / f"{start_idx}_{end_idx}"
                seg_dir.mkdir(parents=True, exist_ok=True)

                segment = {
                    "body_pos_w": motion["body_pos_w"][start_idx:end_idx],
                    "body_quat_w": motion["body_quat_w"][start_idx:end_idx],
                    "body_lin_vel_w": motion["body_lin_vel_w"][start_idx:end_idx],
                    "body_ang_vel_w": motion["body_ang_vel_w"][start_idx:end_idx],
                    "joint_pos": motion["joint_pos"][start_idx:end_idx],
                    "joint_vel": motion["joint_vel"][start_idx:end_idx],
                }

                np.savez(seg_dir / "motion.npz", **segment)
                (seg_dir / "meta.json").write_text(json.dumps(meta_payload))
                print(f"Saved segment: {seg_dir}")
        else:
            seg_dir = output_dir / rel_base
            seg_dir.mkdir(parents=True, exist_ok=True)
            segment = {
                "body_pos_w": motion["body_pos_w"],
                "body_quat_w": motion["body_quat_w"],
                "body_lin_vel_w": motion["body_lin_vel_w"],
                "body_ang_vel_w": motion["body_ang_vel_w"],
                "joint_pos": motion["joint_pos"],
                "joint_vel": motion["joint_vel"],
            }
            np.savez(seg_dir / "motion.npz", **segment)
            (seg_dir / "meta.json").write_text(json.dumps(meta_payload))
            print(f"Saved motion: {seg_dir}")


if __name__ == "__main__":
    main()
