"""
Módulo de Interfaz Gráfica de Usuario (GUI) — Fase 3.

Proporciona un panel de control claro y ordenado en Tkinter (tema oscuro de alto contraste)
sincronizado en tiempo real con el visor 3D de MuJoCo (`show_left_ui=False`,
`show_right_ui=False`) en el mismo hilo principal.
"""

from typing import Sequence
import tkinter as tk
from tkinter import ttk
import mujoco
import mujoco.viewer
import numpy as np

from src.kinematics import (
    forward_kinematics_deg,
    inverse_kinematics_deg,
)
from src.robot_control import RobotArmController


# Paleta de colores estilo panel de robótica (Dark Slate Dashboard)
BG_MAIN = "#0f172a"        # Fondo principal oscuro (Slate 900)
BG_CARD = "#1e293b"        # Fondo de tarjetas (Slate 800)
BG_INPUT = "#0b1120"       # Fondo de cajas de texto y lecturas digitales
BORDER_CARD = "#334155"    # Borde sutil de tarjetas (Slate 700)
FG_PRIMARY = "#f8fafc"     # Texto blanco principal
FG_SECONDARY = "#94a3b8"   # Texto secundario gris azulado
FG_DIGITAL = "#38bdf8"     # Azul cian para pantallas digitales de telemetría
ACCENT_BLUE = "#0284c7"    # Botón de cálculo azul
ACCENT_GREEN = "#059669"   # Botón de ejecución verde esmeralda
ACCENT_HOME = "#d97706"    # Botón Home ámbar/naranja
ACCENT_SUBTLE = "#334155"  # Botones secundarios
JOINT_COLORS = ("#38bdf8", "#fb923c", "#4ade80")  # J1: Azul, J2: Naranja, J3: Verde

# Tolerancia máxima (en milímetros) para considerar coincidencia exacta entre MuJoCo y DH
TOL_MATCH_MM = 0.05


class RobotControlApp:
    """Interfaz gráfica de control articular y cartesiano para el manipulador 3 GDL."""

    def __init__(
        self,
        controller: RobotArmController | None = None,
        open_3d_viewer: bool = True,
    ) -> None:
        self.controller = controller if controller is not None else RobotArmController()
        self.open_3d_viewer = open_3d_viewer
        self.viewer = None

        self.root = tk.Tk()
        self.root.title("Panel de Control Robótico 3 GDL — MuJoCo (UPV)")
        self.root.geometry("820x830")
        self.root.minsize(780, 790)
        self.root.configure(bg=BG_MAIN)

        self._configure_styles()

        self.limits_deg = self.controller.get_joint_limits_deg()
        self._updating_sliders = False
        self.current_target_xyz: np.ndarray | None = None

        # Variables de control articular (inicializadas en la postura del controlador: Home [0°, +45°, -30°])
        init_q = self.controller.get_target_angles_deg()
        self.var_joint_sliders = [tk.DoubleVar(value=round(float(init_q[i]), 2)) for i in range(3)]
        self.var_joint_entries = [tk.StringVar(value=f"{float(init_q[i]):.2f}") for i in range(3)]

        # Variables de control cartesiano
        home_xyz = self.controller.get_end_effector_position()
        self.var_target_x = tk.StringVar(value=f"{home_xyz[0]:.4f}")
        self.var_target_y = tk.StringVar(value=f"{home_xyz[1]:.4f}")
        self.var_target_z = tk.StringVar(value=f"{home_xyz[2]:.4f}")
        self.var_elbow_config = tk.StringVar(value="down")
        self.var_show_joint_axes = tk.BooleanVar(value=False)
        self.var_show_frames = tk.BooleanVar(value=False)

        # Variables de telemetría en tiempo real
        self.var_info_q_measured = tk.StringVar(value="[ +0.00°, +45.00°, -30.00°]")
        self.var_info_q_ref = tk.StringVar(value="[ +0.00°, +45.00°, -30.00°]")
        self.var_info_xyz_mujoco = tk.StringVar(
            value=f"X = {home_xyz[0]:+.4f} m   Y = {home_xyz[1]:+.4f} m   Z = {home_xyz[2]:+.4f} m"
        )
        self.var_info_xyz_dh = tk.StringVar(
            value=f"X = {home_xyz[0]:+.4f} m   Y = {home_xyz[1]:+.4f} m   Z = {home_xyz[2]:+.4f} m"
        )
        self.var_info_error_mm = tk.StringVar(
            value="0.0000 mm   [✓ COINCIDENCIA EXACTA (< 0.05 mm)]"
        )
        self.var_info_target_xyz = tk.StringVar(
            value=f"X = {home_xyz[0]:+.4f} m   Y = {home_xyz[1]:+.4f} m   Z = {home_xyz[2]:+.4f} m  (Home)"
        )
        self.var_info_ik_result = tk.StringVar(
            value=f"q₁ = {init_q[0]:+6.2f}°   q₂ = {init_q[1]:+6.2f}°   q₃ = {init_q[2]:+6.2f}°"
        )
        self.var_status_msg = tk.StringVar(
            value="LISTO — Simulación física y visor 3D sincronizados en tiempo real."
        )

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        if self.open_3d_viewer:
            self._launch_mujoco_viewer()

        self._refresh_telemetry_labels()

    def _configure_styles(self) -> None:
        """Configura el tema visual oscuro para los widgets ttk."""
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure(
            "TScale",
            background=BG_CARD,
            troughcolor=BG_INPUT,
            bordercolor=BORDER_CARD,
            lightcolor=ACCENT_BLUE,
            darkcolor=ACCENT_BLUE,
            sliderlength=24,
        )
        style.configure(
            "TCombobox",
            fieldbackground=BG_INPUT,
            background=ACCENT_SUBTLE,
            foreground=FG_PRIMARY,
            arrowcolor=FG_PRIMARY,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", BG_INPUT)],
            foreground=[("readonly", FG_PRIMARY)],
            selectbackground=[("readonly", ACCENT_BLUE)],
            selectforeground=[("readonly", FG_PRIMARY)],
        )

    def _launch_mujoco_viewer(self) -> None:
        """
        Inicia el visor 3D de MuJoCo ocultando los paneles laterales de diagnóstico
        (`show_left_ui=False`, `show_right_ui=False`) para mostrar la escena 3D limpia.
        """
        self.viewer = mujoco.viewer.launch_passive(
            self.controller.model,
            self.controller.data,
            show_left_ui=False,
            show_right_ui=False,
        )
        self.viewer.cam.azimuth = 135
        self.viewer.cam.elevation = -22
        self.viewer.cam.distance = 0.42
        self.viewer.cam.lookat[:] = [0.06, 0.0, 0.08]

    def _create_card(self, parent: tk.Widget, title: str, subtitle: str = "") -> tk.Frame:
        """Crea un contenedor tipo tarjeta con borde y cabecera."""
        outer = tk.Frame(
            parent,
            bg=BG_CARD,
            highlightbackground=BORDER_CARD,
            highlightthickness=1,
            padx=16,
            pady=11,
        )
        outer.pack(fill=tk.X, pady=6)

        header = tk.Frame(outer, bg=BG_CARD)
        header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            header,
            text=title,
            bg=BG_CARD,
            fg=FG_PRIMARY,
            font=("Segoe UI", 11, "bold"),
        ).pack(side=tk.LEFT)

        if subtitle:
            tk.Label(
                header,
                text=subtitle,
                bg=BG_CARD,
                fg=FG_SECONDARY,
                font=("Segoe UI", 9, "bold"),
            ).pack(side=tk.RIGHT)

        tk.Frame(outer, bg=BORDER_CARD, height=1).pack(fill=tk.X, pady=(0, 8))
        return outer

    def _create_button(
        self,
        parent: tk.Widget,
        text: str,
        command,
        bg_color: str = ACCENT_BLUE,
    ) -> tk.Button:
        """Crea un botón estilizado plano de alto contraste."""
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=bg_color,
            fg=FG_PRIMARY,
            activebackground=FG_PRIMARY,
            activeforeground=BG_MAIN,
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=6,
            cursor="hand2",
            font=("Segoe UI", 9, "bold"),
        )

    def _build_ui(self) -> None:
        """Construye y distribuye los paneles del panel de control."""
        container = tk.Frame(self.root, bg=BG_MAIN, padx=18, pady=12)
        container.pack(fill=tk.BOTH, expand=True)

        # ==================== CABECERA PRINCIPAL ====================
        top_bar = tk.Frame(container, bg=BG_MAIN)
        top_bar.pack(fill=tk.X, pady=(0, 6))

        title_box = tk.Frame(top_bar, bg=BG_MAIN)
        title_box.pack(side=tk.LEFT)

        tk.Label(
            title_box,
            text="MANIPULADOR ROBÓTICO 3 GDL (RRR)",
            bg=BG_MAIN,
            fg=FG_PRIMARY,
            font=("Segoe UI", 15, "bold"),
        ).pack(anchor="w")

        tk.Label(
            title_box,
            text="Simulación Física en MuJoCo  •  Cinemática DH y Geométrica (UPV)",
            bg=BG_MAIN,
            fg=FG_SECONDARY,
            font=("Segoe UI", 9),
        ).pack(anchor="w")

        btn_home = self._create_button(
            top_bar,
            text="⌂  Posición Home (Codo)",
            command=self.on_reset_home,
            bg_color=ACCENT_HOME,
        )
        btn_home.pack(side=tk.RIGHT, pady=4)

        btn_origin = self._create_button(
            top_bar,
            text="⊙  Ir a Origen [0°, 0°, 0°]",
            command=self.on_reset_origin,
            bg_color=ACCENT_SUBTLE,
        )
        btn_origin.pack(side=tk.RIGHT, padx=(0, 8), pady=4)

        # ==================== TARJETA 1: CONTROL ARTICULAR ====================
        card_joints = self._create_card(
            container,
            "1. CONTROL ARTICULAR DIRECTO (GRADOS)",
            "Límites articulares de cada eje: [-90.0°, +90.0°]",
        )

        # Nombres claros y diferenciados para las tres articulaciones
        joint_titles = [
            "J1 — Giro de la Base (q₁ = θ₁):",
            "J2 — Movimiento del Hombro (q₂ = θ₂):",
            "J3 — Movimiento del Codo (q₃ = θ₃):",
        ]

        joints_grid = tk.Frame(card_joints, bg=BG_CARD)
        joints_grid.pack(fill=tk.X)
        joints_grid.columnconfigure(3, weight=1)  # El deslizador se expande sin cortar etiquetas

        for idx in range(3):
            low, high = self.limits_deg[idx]

            # Columna 0: barra de color identificativa del eslabón
            color_bar = tk.Frame(joints_grid, bg=JOINT_COLORS[idx], width=5, height=26)
            color_bar.grid(row=idx, column=0, padx=(0, 10), pady=5, sticky="ns")

            # Columna 1: etiqueta de la articulación (sin ancho fijo que recorte el texto)
            tk.Label(
                joints_grid,
                text=joint_titles[idx],
                bg=BG_CARD,
                fg=FG_PRIMARY,
                font=("Segoe UI", 10, "bold"),
                anchor="w",
            ).grid(row=idx, column=1, sticky="w", padx=(0, 12), pady=5)

            # Columna 2: límite inferior (-90°)
            tk.Label(
                joints_grid,
                text=f"{low:+.0f}°",
                bg=BG_CARD,
                fg=FG_SECONDARY,
                font=("Consolas", 9, "bold"),
            ).grid(row=idx, column=2, padx=(4, 6), pady=5)

            # Columna 3: deslizador horizontal
            slider = ttk.Scale(
                joints_grid,
                from_=float(low),
                to=float(high),
                orient=tk.HORIZONTAL,
                variable=self.var_joint_sliders[idx],
                command=lambda val, i=idx: self.on_slider_changed(i, float(val)),
            )
            slider.grid(row=idx, column=3, sticky="ew", padx=4, pady=5)

            # Columna 4: límite superior (+90°)
            tk.Label(
                joints_grid,
                text=f"{high:+.0f}°",
                bg=BG_CARD,
                fg=FG_SECONDARY,
                font=("Consolas", 9, "bold"),
            ).grid(row=idx, column=4, padx=(6, 12), pady=5)

            # Columna 5: campo numérico editable
            entry = tk.Entry(
                joints_grid,
                textvariable=self.var_joint_entries[idx],
                width=7,
                justify=tk.RIGHT,
                bg=BG_INPUT,
                fg=JOINT_COLORS[idx],
                insertbackground=FG_PRIMARY,
                relief=tk.FLAT,
                highlightbackground=BORDER_CARD,
                highlightthickness=1,
                font=("Consolas", 11, "bold"),
            )
            entry.grid(row=idx, column=5, ipady=3, pady=5)
            entry.bind("<Return>", lambda _evt, i=idx: self.on_joint_entry_applied(i))

            # Columna 6: unidad y rango explícito
            tk.Label(
                joints_grid,
                text="°   [-90°, +90°]",
                bg=BG_CARD,
                fg=FG_SECONDARY,
                font=("Consolas", 9),
            ).grid(row=idx, column=6, sticky="w", padx=(6, 0), pady=5)

        # Fila inferior de control articular
        joint_footer = tk.Frame(card_joints, bg=BG_CARD)
        joint_footer.pack(fill=tk.X, pady=(8, 0))

        chk_joints = tk.Checkbutton(
            joint_footer,
            text="Mostrar ejes de giro (J1, J2, J3)",
            variable=self.var_show_joint_axes,
            command=self.on_toggle_3d_axes,
            bg=BG_CARD,
            fg=FG_DIGITAL,
            selectcolor=BG_INPUT,
            activebackground=BG_CARD,
            activeforeground=FG_PRIMARY,
            font=("Segoe UI", 9, "bold"),
        )
        chk_joints.pack(side=tk.LEFT, padx=(0, 16))

        chk_frames = tk.Checkbutton(
            joint_footer,
            text="Mostrar sistemas XYZ",
            variable=self.var_show_frames,
            command=self.on_toggle_3d_axes,
            bg=BG_CARD,
            fg=FG_SECONDARY,
            selectcolor=BG_INPUT,
            activebackground=BG_CARD,
            activeforeground=FG_PRIMARY,
            font=("Segoe UI", 9),
        )
        chk_frames.pack(side=tk.LEFT)

        btn_apply_joints = self._create_button(
            joint_footer,
            text="Aplicar Ángulos Escritos",
            command=self.on_apply_all_joint_entries,
            bg_color=ACCENT_BLUE,
        )
        btn_apply_joints.pack(side=tk.RIGHT)

        # ==================== TARJETA 2: CONTROL CARTESIANO (CI) ====================
        card_cart = self._create_card(
            container,
            "2. CONTROL CARTESIANO — CINEMÁTICA INVERSA (METROS)",
            "Alcance máx: 0.200 m desde hombro (z = 0.060 m)",
        )

        inputs_row = tk.Frame(card_cart, bg=BG_CARD)
        inputs_row.pack(fill=tk.X, pady=2)

        coord_specs = [
            ("Coordenada X (m):", self.var_target_x),
            ("Coordenada Y (m):", self.var_target_y),
            ("Coordenada Z (m):", self.var_target_z),
        ]

        for label_txt, tk_var in coord_specs:
            box = tk.Frame(inputs_row, bg=BG_CARD)
            box.pack(side=tk.LEFT, padx=(0, 18))
            tk.Label(
                box,
                text=label_txt,
                bg=BG_CARD,
                fg=FG_SECONDARY,
                font=("Segoe UI", 9, "bold"),
            ).pack(anchor="w", pady=(0, 3))
            ent = tk.Entry(
                box,
                textvariable=tk_var,
                width=11,
                justify=tk.RIGHT,
                bg=BG_INPUT,
                fg=FG_DIGITAL,
                insertbackground=FG_PRIMARY,
                relief=tk.FLAT,
                highlightbackground=BORDER_CARD,
                highlightthickness=1,
                font=("Consolas", 11, "bold"),
            )
            ent.pack(ipady=3)

        # Selector de configuración de codo con explicación clara
        elbow_box = tk.Frame(inputs_row, bg=BG_CARD)
        elbow_box.pack(side=tk.LEFT, padx=(6, 0))
        tk.Label(
            elbow_box,
            text="Configuración de Codo (rama):",
            bg=BG_CARD,
            fg=FG_SECONDARY,
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", pady=(0, 3))
        elbow_combo = ttk.Combobox(
            elbow_box,
            textvariable=self.var_elbow_config,
            values=("down", "up"),
            state="readonly",
            width=12,
            font=("Segoe UI", 10),
        )
        elbow_combo.pack(anchor="w", ipady=2)

        # Nota explicativa de las configuraciones 'down' y 'up'
        tk.Label(
            card_cart,
            text="Nota:  'down' = codo flexionado hacia abajo (q₃ ≤ 0°, hombro elevado)   |   "
                 "'up' = codo flexionado hacia arriba (q₃ ≥ 0°, hombro bajo).",
            bg=BG_CARD,
            fg=FG_SECONDARY,
            font=("Segoe UI", 8, "italic"),
            anchor="w",
        ).pack(fill=tk.X, pady=(6, 4))

        # Botones de acción cartesiana claramente diferenciados
        cart_actions = tk.Frame(card_cart, bg=BG_CARD)
        cart_actions.pack(fill=tk.X, pady=(6, 2))

        self._create_button(
            cart_actions,
            text="1. Calcular Cinemática Inversa (CI)",
            command=self.on_calculate_ik,
            bg_color=ACCENT_BLUE,
        ).pack(side=tk.LEFT, padx=(0, 12))

        self._create_button(
            cart_actions,
            text="▶  2. Ejecutar Movimiento al Objetivo",
            command=self.on_execute_cartesian_move,
            bg_color=ACCENT_GREEN,
        ).pack(side=tk.LEFT)

        self._create_button(
            cart_actions,
            text="Copiar Posición Actual",
            command=self.on_copy_current_tcp,
            bg_color=ACCENT_SUBTLE,
        ).pack(side=tk.RIGHT)

        # ==================== TARJETA 3: TELEMETRÍA E INFORMACIÓN ====================
        card_info = self._create_card(
            container,
            "3. TELEMETRÍA Y ESTADO DEL ROBOT EN TIEMPO REAL",
            "Comparativa MuJoCo vs Denavit-Hartenberg (T₃⁰)",
        )

        grid_frame = tk.Frame(card_info, bg=BG_CARD)
        grid_frame.pack(fill=tk.X)
        grid_frame.columnconfigure(1, weight=1)

        telemetry_items = [
            ("Ángulos reales medidos en MuJoCo (q):", self.var_info_q_measured, FG_DIGITAL),
            ("Ángulos articulares de referencia:", self.var_info_q_ref, FG_PRIMARY),
            ("Posición real del efector final (MuJoCo):", self.var_info_xyz_mujoco, "#4ade80"),
            ("Posición calculada por Cinemática Directa (DH):", self.var_info_xyz_dh, "#4ade80"),
            ("Error de posición |MuJoCo − DH| (mm):", self.var_info_error_mm, "#a7f3d0"),
            ("Posición cartesiana objetivo (X, Y, Z):", self.var_info_target_xyz, "#fbbf24"),
            ("Resultado de la Cinemática Inversa (CI):", self.var_info_ik_result, "#c084fc"),
        ]

        self.error_readout_label = None
        for row_idx, (label_text, var_obj, color_hex) in enumerate(telemetry_items):
            tk.Label(
                grid_frame,
                text=label_text,
                bg=BG_CARD,
                fg=FG_SECONDARY,
                font=("Segoe UI", 9, "bold"),
                anchor="w",
            ).grid(row=row_idx, column=0, sticky="w", pady=3, padx=(0, 14))

            readout = tk.Label(
                grid_frame,
                textvariable=var_obj,
                bg=BG_INPUT,
                fg=color_hex,
                font=("Consolas", 10, "bold"),
                anchor="w",
                padx=10,
                pady=3,
            )
            readout.grid(row=row_idx, column=1, sticky="ew", pady=3)
            if var_obj is self.var_info_error_mm:
                self.error_readout_label = readout

        # ==================== BARRA DE ESTADO INFERIOR ====================
        self.status_label = tk.Label(
            container,
            textvariable=self.var_status_msg,
            anchor="w",
            padx=14,
            pady=8,
            bg="#064e3b",
            fg="#a7f3d0",
            font=("Segoe UI", 9, "bold"),
            wraplength=760,
            justify=tk.LEFT,
        )
        self.status_label.pack(fill=tk.X, pady=(6, 0))

    def _set_status(self, message: str, is_error: bool = False) -> None:
        """Actualiza la barra de estado con colores de alto contraste (verde éxito / rojo error)."""
        self.var_status_msg.set(message)
        if is_error:
            self.status_label.configure(bg="#7f1d1d", fg="#fecaca")
        else:
            self.status_label.configure(bg="#064e3b", fg="#a7f3d0")

    def _sync_controls_to_angles(self, angles_deg: Sequence[float]) -> None:
        """Sincroniza sliders y campos de texto articulares sin activar eventos en bucle."""
        self._updating_sliders = True
        try:
            for idx in range(3):
                val = float(angles_deg[idx])
                self.var_joint_sliders[idx].set(round(val, 2))
                self.var_joint_entries[idx].set(f"{val:.2f}")
        finally:
            self._updating_sliders = False

    # ==================== ACCIONES DE CONTROL ARTICULAR ====================

    def on_slider_changed(self, joint_idx: int, value_deg: float) -> None:
        """Callback al arrastrar cualquiera de los 3 deslizadores."""
        if self._updating_sliders:
            return
        clipped = self.controller.set_single_joint_deg(
            joint_idx, value_deg, direct_qpos=False
        )
        self.var_joint_entries[joint_idx].set(f"{clipped[joint_idx]:.2f}")
        fk_xyz = forward_kinematics_deg(clipped)
        self.controller.set_target_marker_position(fk_xyz)
        self._set_status(
            f"Control articular: Articulación J{joint_idx + 1} movida a {clipped[joint_idx]:+.2f}°."
        )
        self._refresh_telemetry_labels()

    def on_joint_entry_applied(self, joint_idx: int) -> None:
        """Callback al pulsar Enter en un campo numérico de ángulo articular."""
        raw_text = self.var_joint_entries[joint_idx].get().strip().replace(",", ".")
        try:
            requested_val = float(raw_text)
        except ValueError:
            self._set_status(
                f"Error de formato: '{raw_text}' no es un ángulo numérico válido.",
                is_error=True,
            )
            return

        low, high = self.limits_deg[joint_idx]
        was_clipped = requested_val < low or requested_val > high
        clipped = self.controller.set_single_joint_deg(
            joint_idx, requested_val, direct_qpos=False
        )
        self._sync_controls_to_angles(clipped)
        self.controller.set_target_marker_position(forward_kinematics_deg(clipped))

        if was_clipped:
            self._set_status(
                f"Límite articular aplicado: {requested_val:.1f}° excedía [{low:.0f}°, +{high:.0f}°] "
                f"-> recortado a {clipped[joint_idx]:+.2f}°."
            )
        else:
            self._set_status(
                f"Ángulo aplicado en J{joint_idx + 1}: {clipped[joint_idx]:+.2f}°."
            )
        self._refresh_telemetry_labels()

    def on_apply_all_joint_entries(self) -> None:
        """Lee los 3 campos de texto articulares, aplica límites y actualiza MuJoCo."""
        requested = []
        for idx in range(3):
            raw = self.var_joint_entries[idx].get().strip().replace(",", ".")
            try:
                requested.append(float(raw))
            except ValueError:
                self._set_status(
                    f"Error: Valor no numérico '{raw}' en la articulación J{idx + 1}.",
                    is_error=True,
                )
                return

        clipped = self.controller.set_target_angles_deg(requested, direct_qpos=False)
        self._sync_controls_to_angles(clipped)
        self.controller.set_target_marker_position(forward_kinematics_deg(clipped))

        if not np.allclose(requested, clipped, atol=1e-3):
            self._set_status(
                f"Ángulos ajustados a los límites [-90°, +90°]: "
                f"[{clipped[0]:+.1f}°, {clipped[1]:+.1f}°, {clipped[2]:+.1f}°]."
            )
        else:
            self._set_status(
                f"Consigna articular aplicada: "
                f"[{clipped[0]:+.2f}°, {clipped[1]:+.2f}°, {clipped[2]:+.2f}°]."
            )
        self._refresh_telemetry_labels()

    def on_toggle_3d_axes(self) -> None:
        """Activa o desactiva la visualización de las flechas de ejes articulares y sistemas XYZ en MuJoCo."""
        if self.viewer is not None and self.viewer.is_running():
            self.viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_JOINT] = bool(
                self.var_show_joint_axes.get()
            )
            self.viewer.opt.frame = (
                mujoco.mjtFrame.mjFRAME_SITE
                if bool(self.var_show_frames.get())
                else mujoco.mjtFrame.mjFRAME_NONE
            )
            self.viewer.sync()
        estado_j = "visibles" if self.var_show_joint_axes.get() else "ocultos"
        self._set_status(f"Visualización 3D actualizada: ejes de giro articulares {estado_j}.")

    def on_reset_home(self) -> None:
        """Restaura el robot a la postura robótica Home ([0°, +45°, -30°])."""
        self.controller.reset_to_home()
        q_home = self.controller.get_target_angles_deg()
        self._sync_controls_to_angles(q_home)
        self.current_target_xyz = None
        home_xyz = self.controller.get_end_effector_position()
        self.var_target_x.set(f"{home_xyz[0]:.4f}")
        self.var_target_y.set(f"{home_xyz[1]:.4f}")
        self.var_target_z.set(f"{home_xyz[2]:.4f}")
        self.var_info_target_xyz.set(
            f"X = {home_xyz[0]:+.4f} m   Y = {home_xyz[1]:+.4f} m   Z = {home_xyz[2]:+.4f} m  (Home)"
        )
        self.var_info_ik_result.set(
            f"q₁ = {q_home[0]:+6.2f}°   q₂ = {q_home[1]:+6.2f}°   q₃ = {q_home[2]:+6.2f}°"
        )
        self._set_status("Robot restaurado a la Posición Home (postura de codo [0°, +45°, -30°]).")
        self._refresh_telemetry_labels()

    def on_reset_origin(self) -> None:
        """Lleva el robot a la posición Origen matemático DH ([0°, 0°, 0°])."""
        self.controller.reset_to_origin()
        self._sync_controls_to_angles([0.0, 0.0, 0.0])
        self.current_target_xyz = None
        origin_xyz = self.controller.get_end_effector_position()
        self.var_target_x.set(f"{origin_xyz[0]:.4f}")
        self.var_target_y.set(f"{origin_xyz[1]:.4f}")
        self.var_target_z.set(f"{origin_xyz[2]:.4f}")
        self.var_info_target_xyz.set(
            f"X = {origin_xyz[0]:+.4f} m   Y = {origin_xyz[1]:+.4f} m   Z = {origin_xyz[2]:+.4f} m  (Origen)"
        )
        self.var_info_ik_result.set("q₁ =  +0.00°   q₂ =  +0.00°   q₃ =  +0.00°")
        self._set_status("Robot situado en la Posición Origen matemático ([0°, 0°, 0°]).")
        self._refresh_telemetry_labels()

    # ==================== ACCIONES DE CONTROL CARTESIANO ====================

    def _read_target_xyz_inputs(self) -> np.ndarray | None:
        """Lee y valida las entradas X, Y, Z en metros."""
        try:
            x = float(self.var_target_x.get().strip().replace(",", "."))
            y = float(self.var_target_y.get().strip().replace(",", "."))
            z = float(self.var_target_z.get().strip().replace(",", "."))
            return np.array([x, y, z], dtype=float)
        except ValueError:
            self._set_status(
                "Error: Las coordenadas objetivo X, Y, Z deben ser números reales válidos.",
                is_error=True,
            )
            return None

    def on_calculate_ik(self) -> np.ndarray | None:
        """Calcula la Cinemática Inversa para (X, Y, Z) y muestra la solución articular."""
        target_xyz = self._read_target_xyz_inputs()
        if target_xyz is None:
            return None

        self.current_target_xyz = target_xyz
        self.var_info_target_xyz.set(
            f"X = {target_xyz[0]:+.4f} m   Y = {target_xyz[1]:+.4f} m   Z = {target_xyz[2]:+.4f} m"
        )

        elbow_mode = self.var_elbow_config.get()
        try:
            q_sol = inverse_kinematics_deg(target_xyz, elbow_config=elbow_mode)
        except ValueError as exc:
            self.var_info_ik_result.set("SIN SOLUCIÓN VÁLIDA (Fuera de límites/alcance)")
            self._set_status(f"Error Cinemática Inversa: {exc}", is_error=True)
            return None

        self.controller.set_target_marker_position(target_xyz)
        self.var_info_ik_result.set(
            f"q₁ = {q_sol[0]:+6.2f}°   q₂ = {q_sol[1]:+6.2f}°   q₃ = {q_sol[2]:+6.2f}°"
        )
        self._set_status(
            f"Cinemática Inversa OK: q = [{q_sol[0]:+.2f}°, {q_sol[1]:+.2f}°, {q_sol[2]:+.2f}°]. "
            f"Pulsa '2. Ejecutar Movimiento' para desplazar el brazo."
        )
        return q_sol

    def on_execute_cartesian_move(self) -> bool:
        """Calcula la CI y ejecuta el movimiento hacia el objetivo (X, Y, Z) en MuJoCo."""
        q_sol = self.on_calculate_ik()
        if q_sol is None:
            return False

        clipped = self.controller.set_target_angles_deg(q_sol, direct_qpos=False)
        self._sync_controls_to_angles(clipped)
        self._set_status(
            f"Movimiento en ejecución hacia [{self.current_target_xyz[0]:+.3f}, "
            f"{self.current_target_xyz[1]:+.3f}, {self.current_target_xyz[2]:+.3f}] m "
            f"-> q = [{clipped[0]:+.2f}°, {clipped[1]:+.2f}°, {clipped[2]:+.2f}°]."
        )
        self._refresh_telemetry_labels()
        return True

    def on_copy_current_tcp(self) -> None:
        """Copia las coordenadas actuales del TCP a los campos de entrada X, Y, Z."""
        tcp = self.controller.get_end_effector_position()
        self.var_target_x.set(f"{tcp[0]:.4f}")
        self.var_target_y.set(f"{tcp[1]:.4f}")
        self.var_target_z.set(f"{tcp[2]:.4f}")
        self._set_status("Coordenadas actuales del efector final copiadas a X, Y, Z.")

    # ==================== BUCLE DE SIMULACIÓN Y TELEMETRÍA ====================

    def _refresh_telemetry_labels(self) -> None:
        """Refresca las lecturas digitales con el estado real de MuJoCo, la CD (DH) y el error en mm."""
        q_meas = self.controller.get_joint_angles_deg()
        q_ref = self.controller.get_target_angles_deg()
        pos_mj = self.controller.get_end_effector_position()
        pos_dh = forward_kinematics_deg(q_meas)

        # Error euclídeo entre la posición real de MuJoCo y la calculada por DH (expresado en mm)
        err_mm = float(np.linalg.norm(pos_mj - pos_dh) * 1000.0)
        is_exact_match = err_mm < TOL_MATCH_MM

        self.var_info_q_measured.set(
            f"q₁ = {q_meas[0]:+6.2f}°   q₂ = {q_meas[1]:+6.2f}°   q₃ = {q_meas[2]:+6.2f}°"
        )
        self.var_info_q_ref.set(
            f"q₁ = {q_ref[0]:+6.2f}°   q₂ = {q_ref[1]:+6.2f}°   q₃ = {q_ref[2]:+6.2f}°"
        )
        self.var_info_xyz_mujoco.set(
            f"X = {pos_mj[0]:+.4f} m   Y = {pos_mj[1]:+.4f} m   Z = {pos_mj[2]:+.4f} m"
        )
        self.var_info_xyz_dh.set(
            f"X = {pos_dh[0]:+.4f} m   Y = {pos_dh[1]:+.4f} m   Z = {pos_dh[2]:+.4f} m"
        )

        if is_exact_match:
            self.var_info_error_mm.set(
                f"{err_mm:.4f} mm   [✓ COINCIDENCIA EXACTA (< {TOL_MATCH_MM:.2f} mm)]"
            )
            if self.error_readout_label is not None:
                self.error_readout_label.configure(fg="#4ade80")
        else:
            self.var_info_error_mm.set(
                f"{err_mm:.4f} mm   [⚠ DESVIACIÓN (> {TOL_MATCH_MM:.2f} mm)]"
            )
            if self.error_readout_label is not None:
                self.error_readout_label.configure(fg="#f87171")

    def step_once(self, physics_steps: int = 8) -> bool:
        """Avanza un ciclo de simulación física y sincroniza visor 3D + interfaz."""
        if self.viewer is not None and not self.viewer.is_running():
            self.close()
            return False

        self.controller.step_simulation(steps=physics_steps)

        if self.viewer is not None and self.viewer.is_running():
            self.viewer.sync()

        self._refresh_telemetry_labels()
        return True

    def _periodic_loop(self) -> None:
        """Bucle periódico (~60 Hz) en el hilo principal de Tkinter."""
        if not self.step_once(physics_steps=8):
            return
        self.root.after(16, self._periodic_loop)

    def run(self) -> None:
        """Inicia la interfaz gráfica y la simulación."""
        self.root.after(16, self._periodic_loop)
        self.root.mainloop()

    def close(self) -> None:
        """Cierra limpiamente el visor 3D de MuJoCo y la ventana de control."""
        if self.viewer is not None:
            try:
                self.viewer.close()
            except Exception:
                pass
            self.viewer = None
        try:
            self.root.destroy()
        except Exception:
            pass
