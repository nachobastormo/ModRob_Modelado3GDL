"""
Pruebas automatizadas de la Fase 2: Cinemática Directa e Inversa contrastadas con MuJoCo.

Ejecución:
    python -m unittest discover -s tests -v
    (o directamente: python tests/test_kinematics.py)
"""

import sys
from pathlib import Path
import unittest
import numpy as np

# Asegurar que la raíz del proyecto esté en sys.path al ejecutar directamente
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kinematics import (
    a2,
    a3,
    check_joint_limits_deg,
    d1,
    dh_matrix,
    forward_kinematics_deg,
    forward_kinematics_matrix_deg,
    get_dh_table_rad,
    inverse_kinematics_deg,
)
from src.robot_control import RobotArmController


# Tolerancias numéricas justificadas:
# - TOL_FK_MUJOCO_M (1e-6 m = 1 micrómetro): diferencia máxima admisible entre la fórmula
#   analítica en doble precisión y la cinemática calculada por MuJoCo (`mj_forward`).
#   Nota: MuJoCo compila el XML convirtiendo grados a radianes, por lo que una tolerancia de 1e-6 m
#   cubre sobradamente cualquier redondeo interno.
TOL_FK_MUJOCO_M = 1e-6

# - TOL_IK_POS_M (1e-5 m = 0.01 milímetros): error máximo de reconstrucción espacial
#   entre el objetivo (x, y, z) pedido a la CI y la posición alcanzada en CD y en MuJoCo.
TOL_IK_POS_M = 1e-5


class TestKinematicsPhase2(unittest.TestCase):
    """Batería de pruebas automáticas para validar CD, CI y coherencia con MuJoCo."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.controller = RobotArmController()

    def test_01_forward_kinematics_vs_mujoco(self) -> None:
        """
        Comprueba que la Cinemática Directa analítica coincide exactamente con la posición
        del efector final (`site_end_effector`) calculada por MuJoCo en múltiples posturas.
        """
        test_angles_list = [
            [0.0, 0.0, 0.0],          # Posición Home extendida horizontalmente
            [90.0, 0.0, 0.0],         # Giro de base +90° (apuntando a +Y)
            [-90.0, 0.0, 0.0],        # Giro de base -90° (apuntando a -Y)
            [0.0, 90.0, 0.0],         # Brazo completamente vertical hacia arriba (+Z)
            [0.0, 45.0, -45.0],       # Hombro a +45°, codo a -45° (antebrazo horizontal)
            [30.0, 45.0, -30.0],      # Postura general 3D en el primer octante
            [-45.0, 20.0, 35.0],      # Postura general 3D con codo hacia arriba
            [60.0, -30.0, -40.0],     # Postura inclinada hacia abajo
            [-75.0, 80.0, -85.0],     # Postura cercana a los límites articulares
        ]

        print("\n--- [Test 1] Comparación Cinemática Directa Analítica vs MuJoCo ---")
        for q_deg in test_angles_list:
            pos_math = forward_kinematics_deg(q_deg)
            self.controller.set_target_angles_deg(q_deg, direct_qpos=True)
            pos_mujoco = self.controller.get_end_effector_position()

            error_m = float(np.linalg.norm(pos_math - pos_mujoco))
            print(
                f"  q={str(q_deg):22s} -> CD={pos_math.round(5)} m | "
                f"MuJoCo={pos_mujoco.round(5)} m | Error={error_m:.2e} m"
            )
            self.assertLess(
                error_m,
                TOL_FK_MUJOCO_M,
                f"Discrepancia excesiva entre CD y MuJoCo para {q_deg}: {error_m} m",
            )

    def test_02_inverse_kinematics_roundtrip_and_mujoco(self) -> None:
        """
        Ciclo completo:
        1. Parte de ángulos conocidos -> calcula posición cartesiana con CD.
        2. Usa esa posición como objetivo de la Cinemática Inversa (CI).
        3. Comprueba que los ángulos recuperados cumplen límites y reconstruyen la misma posición
           tanto en la ecuación matemática como dentro del simulador MuJoCo.
        """
        reference_angles_list = [
            [0.0, 0.0, 0.0],
            [30.0, 45.0, -30.0],
            [-40.0, 60.0, -50.0],
            [45.0, 20.0, 40.0],
            [-60.0, -15.0, 30.0],
            [15.0, 75.0, -70.0],
            [0.0, 50.0, -50.0],
        ]

        print("\n--- [Test 2] Ciclo CD -> CI -> Reconstrucción en MuJoCo ---")
        for q_orig in reference_angles_list:
            target_xyz = forward_kinematics_deg(q_orig)

            # Elegimos la rama preferida acorde al signo original de q3 para verificar también ángulos
            preferred_branch = "down" if q_orig[2] <= 0 else "up"
            q_ik = inverse_kinematics_deg(target_xyz, elbow_config=preferred_branch)

            self.assertTrue(
                check_joint_limits_deg(q_ik),
                f"Los ángulos devueltos por CI {q_ik} violan los límites articulares.",
            )

            # Posición reconstruida matemáticamente
            pos_reconstructed = forward_kinematics_deg(q_ik)
            err_math = float(np.linalg.norm(pos_reconstructed - target_xyz))

            # Posición real en el simulador MuJoCo aplicando q_ik
            self.controller.set_target_angles_deg(q_ik, direct_qpos=True)
            pos_mujoco = self.controller.get_end_effector_position()
            err_mujoco = float(np.linalg.norm(pos_mujoco - target_xyz))

            print(
                f"  q_orig={str(q_orig):20s} -> Objetivo={target_xyz.round(4)} m -> "
                f"q_IK={q_ik.round(2).tolist()} -> Err_MuJoCo={err_mujoco:.2e} m"
            )

            self.assertLess(err_math, TOL_IK_POS_M)
            self.assertLess(err_mujoco, TOL_IK_POS_M)

    def test_03_multiple_solutions_elbow_up_down(self) -> None:
        """
        Verifica que para un punto interior alcanzable por ambas configuraciones
        ('down' y 'up'), las dos soluciones son distintas pero sitúan el efector final
        exactamente en el mismo punto (x, y, z) en MuJoCo.
        """
        target = np.array([0.14, 0.05, 0.12])
        q_down = inverse_kinematics_deg(target, elbow_config="only_down")
        q_up = inverse_kinematics_deg(target, elbow_config="only_up")

        print("\n--- [Test 3] Soluciones Múltiples (Codo Abajo vs Codo Arriba) ---")
        print(f"  Objetivo: {target} m")
        print(f"  Solución 'only_down' (q3 <= 0): {q_down.round(2)}°")
        print(f"  Solución 'only_up'   (q3 >= 0): {q_up.round(2)}°")

        # Deben tener signos opuestos en q3
        self.assertLessEqual(q_down[2], 0.0)
        self.assertGreaterEqual(q_up[2], 0.0)
        self.assertFalse(np.allclose(q_down, q_up))

        # Ambas alcanzan el mismo punto en MuJoCo
        for q_sol in (q_down, q_up):
            self.controller.set_target_angles_deg(q_sol, direct_qpos=True)
            pos_mj = self.controller.get_end_effector_position()
            self.assertLess(float(np.linalg.norm(pos_mj - target)), TOL_IK_POS_M)

    def test_04_unreachable_targets_and_joint_limits(self) -> None:
        """
        Verifica que la Cinemática Inversa rechaza (lanza ValueError) objetivos:
        1. Fuera del radio máximo del brazo (D > L1 + L2 = 0.200 m).
        2. Por encima de la altura máxima (z > d1 + L1 + L2 = 0.260 m).
        3. Dentro de la esfera pero en regiones prohibidas por los límites [-90°, +90°]
           (por ejemplo, puntos detrás de la base con x = -0.15 m, o con codo plegado > 90°).
        """
        unreachable_cases = [
            ([0.35, 0.00, 0.06], "Fuera de alcance radial frontal (x = 0.35 m > 0.20 m)"),
            ([0.00, 0.00, 0.35], "Por encima de la altura máxima (z = 0.35 m > 0.26 m)"),
            ([-0.15, 0.00, 0.06], "Detrás de la base (requiere |q1| > 90° o |q2| > 90°)"),
            ([0.02, 0.00, 0.06], "Demasiado cerca del hombro (requiere flexión de codo |q3| > 90°)"),
        ]

        print("\n--- [Test 4] Detección de Objetivos Inalcanzables y Límites Articulares ---")
        for target_xyz, description in unreachable_cases:
            with self.assertRaises(ValueError) as ctx:
                inverse_kinematics_deg(target_xyz)
            print(f"  [Rechazado OK] {description} -> {ctx.exception}")

    def test_05_boundary_and_singularity_targets(self) -> None:
        """
        Verifica puntos límite exactos del espacio de trabajo:
        - Máxima extensión frontal: [0.200, 0.000, 0.060] m -> q = [0°, 0°, 0°].
        - Máxima elevación vertical (singularidad x=0, y=0): [0.000, 0.000, 0.260] m -> q = [0°, 90°, 0°].
        - Lateral derecho máximo (q1 = -90°): [0.000, -0.200, 0.060] m.
        """
        boundary_targets = [
            np.array([a2 + a3, 0.0, d1]),
            np.array([0.0, 0.0, d1 + a2 + a3]),
            np.array([0.0, -(a2 + a3), d1]),
        ]

        print("\n--- [Test 5] Puntos Frontera y Singularidad Vertical ---")
        for target in boundary_targets:
            q_sol = inverse_kinematics_deg(target)
            self.controller.set_target_angles_deg(q_sol, direct_qpos=True)
            pos_mj = self.controller.get_end_effector_position()
            err = float(np.linalg.norm(pos_mj - target))
            print(f"  Frontera {target.round(4)} m -> q_IK={q_sol.round(2)}° | Err_MuJoCo={err:.2e} m")
            self.assertLess(err, TOL_IK_POS_M)

    def test_06_dh_homogeneous_matrices_step_by_step(self) -> None:
        """
        Verifica explícitamente la construcción de las matrices homogéneas A_1, A_2, A_3
        y T_3^0 = A_1 * A_2 * A_3 según el Método de Denavit-Hartenberg de los apuntes (UD1 M11).
        """
        q_deg = [30.0, 45.0, -30.0]
        q_rad = np.deg2rad(q_deg)
        dh_rows = get_dh_table_rad(q_rad)

        A1 = dh_matrix(*dh_rows[0])
        A2 = dh_matrix(*dh_rows[1])
        A3 = dh_matrix(*dh_rows[2])
        T_0_3 = forward_kinematics_matrix_deg(q_deg)

        self.assertTrue(np.allclose(T_0_3, A1 @ A2 @ A3))
        self.assertEqual(T_0_3.shape, (4, 4))

        # Verificar que el vector de posición o_3^0 (última columna de T_3^0) coincide con MuJoCo
        o_0_3 = T_0_3[:3, 3]
        self.controller.set_target_angles_deg(q_deg, direct_qpos=True)
        pos_mj = self.controller.get_end_effector_position()
        print("\n--- [Test 6] Matriz de Transformación Homogénea DH T_3^0 = A_1 * A_2 * A_3 ---")
        print(f"  Parámetros DH (d1={d1} m, a2={a2} m, a3={a3} m) para q={q_deg}°:")
        print(f"  Vector posición o_3^0 (DH) : {o_0_3.round(5)} m")
        print(f"  Vector posición MuJoCo     : {pos_mj.round(5)} m")
        self.assertLess(float(np.linalg.norm(o_0_3 - pos_mj)), TOL_FK_MUJOCO_M)


if __name__ == "__main__":
    unittest.main(verbosity=2)
