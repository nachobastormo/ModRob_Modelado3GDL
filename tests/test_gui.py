"""
Pruebas automatizadas de la Fase 3: Interfaz Gráfica de Usuario (GUI) sincronizada con MuJoCo y Cinemática.

Ejecución:
    python -m unittest discover -s tests -v
"""

import sys
from pathlib import Path
import unittest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.gui import RobotControlApp
from src.kinematics import forward_kinematics_deg


class TestGUIPhase3(unittest.TestCase):
    """Verifica todos los controles de la interfaz gráfica y su conexión real con MuJoCo."""

    def setUp(self) -> None:
        # Iniciamos con visor 3D desactivado en los tests unitarios rápidos y un test específico con visor 3D real
        self.app = RobotControlApp(open_3d_viewer=False)

    def tearDown(self) -> None:
        self.app.close()

    def test_01_joint_sliders_and_entries_move_mujoco_robot(self) -> None:
        """
        Comprueba que al modificar deslizadores y campos de texto articulares en la GUI:
        1. El modelo de MuJoCo mueve realmente sus articulaciones.
        2. Se actualiza la posición cartesiana (X, Y, Z) mostrada en la interfaz.
        3. Se aplican los límites articulares [-90°, +90°] cuando el usuario introduce p.ej. 135°.
        """
        # Mover deslizador de Articulación 1 a +30°
        self.app.on_slider_changed(0, 30.0)
        self.app.controller.step_simulation(350)
        q_mj = self.app.controller.get_joint_angles_deg()
        self.assertAlmostEqual(q_mj[0], 30.0, places=1)

        # Introducir ángulos por teclado en las 3 cajas de texto: [30°, 45°, -30°]
        self.app.var_joint_entries[0].set("30.0")
        self.app.var_joint_entries[1].set("45.0")
        self.app.var_joint_entries[2].set("-30.0")
        self.app.on_apply_all_joint_entries()
        self.app.controller.step_simulation(350)

        q_mj = self.app.controller.get_joint_angles_deg()
        np.testing.assert_allclose(q_mj, [30.0, 45.0, -30.0], atol=0.6)

        expected_xyz = forward_kinematics_deg(q_mj)
        actual_xyz = self.app.controller.get_end_effector_position()
        np.testing.assert_allclose(actual_xyz, expected_xyz, atol=1e-5)

        # Probar recorte de límite articular: escribir 140° en articulación 2 (máximo +90°)
        self.app.var_joint_entries[1].set("140.0")
        self.app.on_joint_entry_applied(1)
        self.app.controller.step_simulation(350)
        q_mj_clipped = self.app.controller.get_joint_angles_deg()
        self.assertAlmostEqual(q_mj_clipped[1], 90.0, delta=0.5)
        self.assertEqual(self.app.var_joint_entries[1].get(), "90.00")

    def test_02_cartesian_ik_calculation_and_execution(self) -> None:
        """
        Comprueba que al introducir coordenadas objetivo X, Y, Z en la GUI:
        1. El botón 'Calcular Cinemática Inversa' calcula y muestra los ángulos sin error.
        2. El botón 'Ejecutar Movimiento al Objetivo' mueve el robot en MuJoCo hasta (X, Y, Z).
        3. El botón 'Volver a Posición Inicial (Home)' devuelve el robot a [0°, +45°, -30°].
        """
        target = [0.1400, 0.0500, 0.1200]
        self.app.var_target_x.set(str(target[0]))
        self.app.var_target_y.set(str(target[1]))
        self.app.var_target_z.set(str(target[2]))
        self.app.var_elbow_config.set("down")

        # 1. Calcular CI
        q_ik = self.app.on_calculate_ik()
        self.assertIsNotNone(q_ik)
        self.assertIn("°", self.app.var_info_ik_result.get())

        # 2. Ejecutar movimiento hacia el objetivo
        moved_ok = self.app.on_execute_cartesian_move()
        self.assertTrue(moved_ok)
        self.app.controller.step_simulation(400)

        pos_mj = self.app.controller.get_end_effector_position()
        np.testing.assert_allclose(pos_mj, target, atol=1.5e-3)

        # 3. Volver a Posición Home (postura de codo [0°, +45°, -30°]) y Origen ([0°, 0°, 0°])
        self.app.on_reset_home()
        q_home = self.app.controller.get_joint_angles_deg()
        np.testing.assert_allclose(q_home, [0.0, 45.0, -30.0], atol=1e-3)
        expected_home_xyz = forward_kinematics_deg([0.0, 45.0, -30.0])
        np.testing.assert_allclose(
            self.app.controller.get_end_effector_position(),
            expected_home_xyz,
            atol=1e-5,
        )
        self.assertIn("COINCIDENCIA EXACTA", self.app.var_info_error_mm.get())

        # 4. Probar botón Ir a Origen [0°, 0°, 0°]
        self.app.on_reset_origin()
        q_orig = self.app.controller.get_joint_angles_deg()
        np.testing.assert_allclose(q_orig, [0.0, 0.0, 0.0], atol=1e-4)
        np.testing.assert_allclose(
            self.app.controller.get_end_effector_position(),
            [0.2000, 0.0000, 0.0600],
            atol=1e-5,
        )
        self.assertEqual(self.app.var_target_x.get(), "0.2000")
        self.assertEqual(self.app.var_target_y.get(), "0.0000")
        self.assertEqual(self.app.var_target_z.get(), "0.0600")
        self.assertIn("COINCIDENCIA EXACTA", self.app.var_info_error_mm.get())

    def test_03_unreachable_cartesian_target_shows_clear_error(self) -> None:
        """
        Comprueba que si se introduce un objetivo fuera del espacio de trabajo (ej. X=0.45 m),
        la GUI informa claramente del error, no mueve el robot a una postura falsa y mantiene su estado.
        """
        self.app.on_reset_home()
        initial_pos = self.app.controller.get_end_effector_position().copy()

        self.app.var_target_x.set("0.45")
        self.app.var_target_y.set("0.00")
        self.app.var_target_z.set("0.10")

        moved = self.app.on_execute_cartesian_move()
        self.assertFalse(moved)
        self.assertIn("SIN SOLUCIÓN", self.app.var_info_ik_result.get())
        self.assertIn("Error Cinemática Inversa", self.app.var_status_msg.get())

        # El robot no debe haberse movido de su posición previa
        np.testing.assert_allclose(
            self.app.controller.get_end_effector_position(), initial_pos, atol=1e-6
        )

    def test_04_gui_and_3d_viewer_synchronization(self) -> None:
        """
        Abre conjuntamente la GUI de Tkinter y el visor 3D de MuJoCo en el mismo hilo,
        ejecuta pasos de simulación y verifica que ambos se actualizan sin bloqueos.
        """
        app_3d = RobotControlApp(open_3d_viewer=True)
        try:
            self.assertIsNotNone(app_3d.viewer)
            self.assertTrue(app_3d.viewer.is_running())

            # Enviar un objetivo cartesiano y avanzar 25 ciclos de simulación + renderizado
            app_3d.var_target_x.set("0.1449")
            app_3d.var_target_y.set("0.0837")
            app_3d.var_target_z.set("0.1566")
            app_3d.on_execute_cartesian_move()

            for _ in range(25):
                alive = app_3d.step_once(physics_steps=10)
                app_3d.root.update()
                self.assertTrue(alive)

            # Comprobar que el robot físico se ha desplazado hacia la referencia
            q_now = app_3d.controller.get_joint_angles_deg()
            self.assertGreater(q_now[0], 20.0)
            self.assertGreater(q_now[1], 30.0)
        finally:
            app_3d.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
