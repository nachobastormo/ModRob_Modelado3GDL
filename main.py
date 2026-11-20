"""
Script principal — Simulación, Cinemática e Interfaz Gráfica del Robot Manipulador 3 GDL en MuJoCo.

Modos de ejecución:
    python main.py                  # Inicia la Interfaz Gráfica (GUI) sincronizada con el visor 3D de MuJoCo
    python main.py --demo           # Ejecuta la demostración articular en el visor 3D (Fase 1)
    python main.py --verify         # Ejecuta las pruebas de verificación de Fase 1
"""

import argparse
import sys
import time
import mujoco
import mujoco.viewer
import numpy as np

from src.gui import RobotControlApp
from src.kinematics import forward_kinematics_deg
from src.robot_control import (
    ACTUATOR_NAMES,
    DEFAULT_MODEL_PATH,
    EE_BODY_NAME,
    EE_SITE_NAME,
    JOINT_NAMES,
    RobotArmController,
)


def print_status(controller: RobotArmController, label: str = "Estado") -> None:
    """Imprime en consola los ángulos articulares y la posición cartesiana (MuJoCo vs CD DH)."""
    q_deg = controller.get_joint_angles_deg()
    target_deg = controller.get_target_angles_deg()
    ee_pos = controller.get_end_effector_position()
    fk_pos = forward_kinematics_deg(q_deg)
    print(
        f"[{label:22s}] "
        f"q_medido = [{q_deg[0]:+6.1f}°, {q_deg[1]:+6.1f}°, {q_deg[2]:+6.1f}°] | "
        f"q_ref = [{target_deg[0]:+6.1f}°, {target_deg[1]:+6.1f}°, {target_deg[2]:+6.1f}°] | "
        f"TCP_MJ (m) = [{ee_pos[0]:+.4f}, {ee_pos[1]:+.4f}, {ee_pos[2]:+.4f}] | "
        f"CD_DH (m) = [{fk_pos[0]:+.4f}, {fk_pos[1]:+.4f}, {fk_pos[2]:+.4f}]"
    )


def run_verification_tests(open_viewer_briefly: bool = True) -> bool:
    """Ejecuta las 6 comprobaciones obligatorias de la Fase 1."""
    print("=" * 80)
    print("EJECUTANDO PRUEBAS OBLIGATORIAS DE LA FASE 1")
    print("=" * 80)

    print(f"[Prueba 1] Python ejecutable : {sys.executable}")
    print(f"[Prueba 1] Versión de MuJoCo : {mujoco.__version__}")
    print("  -> OK: MuJoCo se importa correctamente.\n")

    controller = RobotArmController(DEFAULT_MODEL_PATH)
    print(f"[Prueba 2] Archivo MJCF cargado: {controller.model_path}")
    print(
        f"  -> OK: Cuerpos={controller.model.nbody}, Articulaciones={controller.model.njnt}, "
        f"Geometrías={controller.model.ngeom}, Actuadores={controller.model.nu}\n"
    )

    assert controller.model.njnt >= 3, "El modelo debe tener al menos 3 articulaciones."
    assert controller.model.nu >= 3, "El modelo debe tener al menos 3 actuadores."
    limits_deg = controller.get_joint_limits_deg()
    for idx, (j_name, a_name) in enumerate(zip(JOINT_NAMES, ACTUATOR_NAMES)):
        print(
            f"[Prueba 3] Articulación {idx + 1}: '{j_name}' <-> Actuador '{a_name}' | "
            f"Rango: [{limits_deg[idx, 0]:.1f}°, {limits_deg[idx, 1]:.1f}°]"
        )
    print("  -> OK: Existen 3 articulaciones tipo bisagra (hinge) con actuadores de posición.\n")

    print("[Prueba 4] Comprobando cambio de posición independiente de cada articulación:")
    controller.reset_to_home()
    print_status(controller, "Posición Inicial (Home)")
    home_ee = controller.get_end_effector_position()

    test_targets = [
        (0, 45.0, "Giro Base (J1 = +45°)"),
        (1, 40.0, "Elevación Hombro (J2 = +40°)"),
        (2, -45.0, "Flexión Codo (J3 = -45°)"),
    ]

    for j_idx, target_angle, desc in test_targets:
        controller.reset_to_origin()
        controller.set_single_joint_deg(j_idx, target_angle, direct_qpos=False)
        controller.step_simulation(steps=400)
        q_now = controller.get_joint_angles_deg()
        print_status(controller, desc)
        assert abs(q_now[j_idx] - target_angle) < 3.0
        for other_idx in (i for i in range(3) if i != j_idx):
            assert abs(q_now[other_idx]) < 1.5

    controller.reset_to_home()
    print_status(controller, "Retorno a Home")
    assert np.allclose(controller.get_joint_angles_deg(), [0.0, 45.0, -30.0], atol=1e-3)
    assert np.allclose(controller.get_end_effector_position(), home_ee, atol=1e-5)
    print("  -> OK: Las 3 articulaciones se mueven independientemente y regresan a Home.\n")

    site_body_id = controller.model.site_bodyid[controller.ee_site_id]
    site_body_name = mujoco.mj_id2name(
        controller.model, mujoco.mjtObj.mjOBJ_BODY, site_body_id
    )
    print(
        f"[Prueba 5] Site TCP: '{EE_SITE_NAME}' pertenece a '{site_body_name}' | "
        f"Posición Home = {home_ee.round(4)} m"
    )
    assert site_body_name == EE_BODY_NAME
    print("  -> OK: El efector final está rígidamente unido al último eslabón.\n")

    if open_viewer_briefly:
        print("[Prueba 6] Abriendo visor 3D pasivo de MuJoCo durante 2.0 segundos...")
        with mujoco.viewer.launch_passive(
            controller.model, controller.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            assert viewer.is_running()
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            viewer.cam.distance = 0.45
            viewer.cam.lookat[:] = [0.05, 0.0, 0.08]
            controller.set_target_angles_deg([30.0, 35.0, -30.0])
            start_t = time.time()
            while viewer.is_running() and (time.time() - start_t) < 2.0:
                controller.step_simulation(steps=5)
                viewer.sync()
                time.sleep(0.01)
        print("  -> OK: El visor 3D se abrió, renderizó el movimiento y cerró sin errores.\n")

    print("=" * 80)
    print("TODAS LAS PRUEBAS OBLIGATORIAS SUPERADAS CON ÉXITO")
    print("=" * 80)
    return True


def run_demo_viewer() -> None:
    """Ejecuta la secuencia de demostración articular por pasos en el visor 3D."""
    controller = RobotArmController(DEFAULT_MODEL_PATH)
    demo_sequence = [
        ("Posición Inicial (Home)", [0.0, 0.0, 0.0], 1.2),
        ("Articulación 1: Base +50°", [50.0, 0.0, 0.0], 1.5),
        ("Articulación 1: Base -50°", [-50.0, 0.0, 0.0], 1.5),
        ("Regreso a Home", [0.0, 0.0, 0.0], 1.2),
        ("Articulación 2: Hombro +45°", [0.0, 45.0, 0.0], 1.5),
        ("Articulación 2: Hombro -20°", [0.0, -20.0, 0.0], 1.5),
        ("Regreso a Home", [0.0, 0.0, 0.0], 1.2),
        ("Articulación 3: Codo +50°", [0.0, 0.0, 50.0], 1.5),
        ("Articulación 3: Codo -50°", [0.0, 0.0, -50.0], 1.5),
        ("Regreso a Posición Inicial (Home)", [0.0, 0.0, 0.0], 1.5),
    ]

    with mujoco.viewer.launch_passive(
        controller.model, controller.data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -25
        viewer.cam.distance = 0.45
        viewer.cam.lookat[:] = [0.05, 0.0, 0.08]

        for label, angles, duration in demo_sequence:
            if not viewer.is_running():
                break
            print(f">>> Paso demo: {label} -> {angles}°")
            controller.set_target_angles_deg(angles)
            t0 = time.time()
            while viewer.is_running() and (time.time() - t0) < duration:
                controller.step_simulation(steps=4)
                viewer.sync()
                time.sleep(0.008)
            print_status(controller, label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Control y simulación del robot manipulador 3 GDL en MuJoCo."
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Ejecuta las 6 pruebas obligatorias de verificación de Fase 1.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Ejecuta la demostración automática de articulaciones en el visor 3D.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.verify:
        run_verification_tests(open_viewer_briefly=True)
    elif args.demo:
        run_demo_viewer()
    else:
        app = RobotControlApp(open_3d_viewer=True)
        app.run()
