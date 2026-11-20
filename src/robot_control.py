"""Módulo de control articular básico e inspección del modelo MuJoCo."""

from pathlib import Path
from typing import Sequence
import mujoco
import numpy as np


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "robot.xml"

JOINT_NAMES = (
    "joint_base_yaw",
    "joint_shoulder_pitch",
    "joint_elbow_pitch",
)

ACTUATOR_NAMES = (
    "servo_base_yaw",
    "servo_shoulder_pitch",
    "servo_elbow_pitch",
)

EE_SITE_NAME = "site_end_effector"
EE_BODY_NAME = "forearm_link"
TARGET_MOCAP_BODY_NAME = "target_marker"


class RobotArmController:
    """Controlador de bajo nivel para el manipulador de 3 GDL en MuJoCo."""

    def __init__(self, model_path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo MJCF en: {self.model_path}")

        xml_content = self.model_path.read_text(encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_string(xml_content)
        self.data = mujoco.MjData(self.model)

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in JOINT_NAMES
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ACTUATOR_NAMES
        ]
        self.ee_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, EE_SITE_NAME
        )
        self.ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, EE_BODY_NAME
        )
        target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, TARGET_MOCAP_BODY_NAME
        )
        self.target_mocap_id = (
            int(self.model.body_mocapid[target_body_id]) if target_body_id >= 0 else -1
        )

        if any(jid < 0 for jid in self.joint_ids):
            raise ValueError(f"Faltan articulaciones esperadas en el modelo: {JOINT_NAMES}")
        if any(aid < 0 for aid in self.actuator_ids):
            raise ValueError(f"Faltan actuadores esperados en el modelo: {ACTUATOR_NAMES}")
        if self.ee_site_id < 0:
            raise ValueError(f"No se encontró el site del efector final: {EE_SITE_NAME}")

        self.reset_to_home()

    def get_joint_limits_deg(self) -> np.ndarray:
        """Devuelve una matriz (3, 2) con los límites [min, max] en grados de cada articulación."""
        limits_rad = self.model.jnt_range[self.joint_ids]
        return np.rad2deg(limits_rad)

    def clip_angles_deg(self, angles_deg: Sequence[float]) -> np.ndarray:
        """Recorta los ángulos en grados dentro de los límites articulares definidos."""
        limits_deg = self.get_joint_limits_deg()
        angles = np.asarray(angles_deg, dtype=float)
        return np.clip(angles, limits_deg[:, 0], limits_deg[:, 1])

    def reset_to_home(self) -> None:
        """Restaura el robot a su postura robótica Home ([0°, +45°, -30°])."""
        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.set_target_marker_position(self.get_end_effector_position())

    def reset_to_origin(self) -> None:
        """Restaura el robot a la posición Origen (cero matemático DH: [0°, 0°, 0°])."""
        origin_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "origen")
        if origin_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, origin_key_id)
        else:
            mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.set_target_marker_position(self.get_end_effector_position())

    def set_target_marker_position(self, pos_xyz: Sequence[float]) -> None:
        """Actualiza la posición de la esfera visual de objetivo en el visor 3D."""
        if self.target_mocap_id >= 0:
            self.data.mocap_pos[self.target_mocap_id] = np.asarray(pos_xyz, dtype=float)

    def set_target_angles_deg(
        self, angles_deg: Sequence[float], direct_qpos: bool = False
    ) -> np.ndarray:
        """
        Asigna las referencias de ángulo para las 3 articulaciones (en grados).
        Si `direct_qpos=True`, actualiza directamente `qpos` además de `ctrl` y recalcula la cinemática.
        """
        clipped_deg = self.clip_angles_deg(angles_deg)
        clipped_rad = np.deg2rad(clipped_deg)

        for idx, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = clipped_rad[idx]

        if direct_qpos:
            for idx, jnt_id in enumerate(self.joint_ids):
                qpos_adr = self.model.jnt_qposadr[jnt_id]
                qvel_adr = self.model.jnt_dofadr[jnt_id]
                self.data.qpos[qpos_adr] = clipped_rad[idx]
                self.data.qvel[qvel_adr] = 0.0
            mujoco.mj_forward(self.model, self.data)

        return clipped_deg

    def set_single_joint_deg(
        self, joint_index: int, angle_deg: float, direct_qpos: bool = False
    ) -> np.ndarray:
        """Modifica únicamente una articulación (0, 1 o 2) manteniendo las demás referencias."""
        if joint_index not in (0, 1, 2):
            raise IndexError("El índice de articulación debe ser 0, 1 o 2.")
        current_targets = self.get_target_angles_deg()
        current_targets[joint_index] = angle_deg
        return self.set_target_angles_deg(current_targets, direct_qpos=direct_qpos)

    def get_joint_angles_deg(self) -> np.ndarray:
        """Devuelve los ángulos actuales medidos (`qpos`) de las 3 articulaciones en grados."""
        qpos_indices = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        return np.rad2deg(self.data.qpos[qpos_indices])

    def get_target_angles_deg(self) -> np.ndarray:
        """Devuelve las consignas actuales (`ctrl`) de los 3 actuadores en grados."""
        return np.rad2deg(self.data.ctrl[self.actuator_ids])

    def get_end_effector_position(self) -> np.ndarray:
        """Devuelve la posición cartesiana [x, y, z] en metros del efector final (`site_end_effector`)."""
        return self.data.site_xpos[self.ee_site_id].copy()

    def step_simulation(self, steps: int = 1) -> None:
        """Avanza la simulación física el número de pasos indicado."""
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
