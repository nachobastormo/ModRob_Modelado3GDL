"""
Script de generación de gráficas científicas para el informe de la práctica
(Sección 4: Resultados obtenidos, reflejo de los Procedimientos 1, 2 y 3).

Genera tres figuras de alta resolución en `docs/figuras/`:
1. `fig_proc1_control_articular.png`:
   Respuesta temporal de las 3 articulaciones (q1, q2, q3) y trayectoria cartesiana del TCP
   en MuJoCo ante consignas escalón secuenciales bajo gravedad (servos PD, kp=6.0, kv=0.4).
2. `fig_proc2_validacion_dh_espacio.png`:
   Comparativa entre la posición analítica de Denavit-Hartenberg (T_3^0) y la posición de MuJoCo
   en 9 configuraciones de prueba, junto con el espacio de trabajo alcanzable en el plano r - z_0.
3. `fig_proc3_cinematica_inversa.png`:
   Comparativa geométrica planar de las dos soluciones de Cinemática Inversa ('down' vs 'up')
   para un mismo objetivo cartesiano [0.14, 0.05, 0.12] m y convergencia dinámica del TCP en MuJoCo.
"""

from pathlib import Path
import sys
import matplotlib.pyplot as plt
import numpy as np

# Añadir la raíz del proyecto al path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.kinematics import (
    a2,
    a3,
    d1,
    dh_matrix,
    forward_kinematics_deg,
    get_dh_table_rad,
    inverse_kinematics_deg,
)
from src.robot_control import RobotArmController

FIGURES_DIR = PROJECT_ROOT / "docs" / "figuras"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Configuración estética general para informe académico (LaTeX)
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 200,
        "savefig.dpi": 220,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "grid.linestyle": "--",
    }
)


def generar_grafica_procedimiento_1() -> Path:
    """
    Prueba 1: Respuesta separada y clara de cada articulación (J1, J2, J3) en MuJoCo.
    Partimos de Origen [0°, 0°, 0°] y movemos un motor cada vez para que se entienda a simple vista:
      - En t = 0.5 s: movemos J1 (Base) a +45°
      - En t = 1.5 s: movemos J2 (Hombro) a +45°
      - En t = 2.5 s: movemos J3 (Codo) a -30° (llegando a la postura Home [0/45°, +45°, -30°])
    """
    ctrl = RobotArmController()
    ctrl.reset_to_origin()

    dt = ctrl.model.opt.timestep  # 0.002 s
    total_time = 3.6
    n_steps = int(total_time / dt)

    times = np.zeros(n_steps)
    q_ref_hist = np.zeros((n_steps, 3))
    q_meas_hist = np.zeros((n_steps, 3))

    for k in range(n_steps):
        t = k * dt
        if t < 0.5:
            target = [0.0, 0.0, 0.0]
        elif t < 1.5:
            target = [45.0, 0.0, 0.0]
        elif t < 2.5:
            target = [45.0, 45.0, 0.0]
        else:
            target = [45.0, 45.0, -30.0]

        ctrl.set_target_angles_deg(target, direct_qpos=False)
        ctrl.step_simulation(1)

        times[k] = t
        q_ref_hist[k] = ctrl.get_target_angles_deg()
        q_meas_hist[k] = ctrl.get_joint_angles_deg()

    fig, axes = plt.subplots(3, 1, figsize=(8.2, 5.8), sharex=True)

    titles = [
        "Articulación 1 (J1 — Giro de la Base): orden de 0° a +45° en t = 0.5 s",
        "Articulación 2 (J2 — Elevación del Hombro): orden de 0° a +45° en t = 1.5 s",
        "Articulación 3 (J3 — Flexión del Codo): orden de 0° a -30° en t = 2.5 s",
    ]
    colors = ["#0284c7", "#ea580c", "#16a34a"]

    for i, ax in enumerate(axes):
        ax.plot(
            times,
            q_ref_hist[:, i],
            linestyle="--",
            color="#475569",
            linewidth=1.5,
            label="Ángulo ordenado (Referencia)",
        )
        ax.plot(
            times,
            q_meas_hist[:, i],
            linestyle="-",
            color=colors[i],
            linewidth=2.3,
            label=f"Ángulo real en MuJoCo (J{i + 1})",
        )
        ax.set_title(titles[i], fontsize=10, fontweight="bold")
        ax.set_ylabel("Grados (°)")
        ax.set_ylim(-45, 60)
        ax.legend(loc="lower right" if i < 2 else "upper right", framealpha=0.95)

    axes[-1].set_xlabel("Tiempo de simulación (segundos)")
    fig.suptitle(
        "Prueba de movimiento articular en MuJoCo (un motor por gráfica)",
        fontsize=12,
        fontweight="bold",
        y=0.99,
    )

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig_control_motores.png"
    fig.savefig(out_path)
    fig.savefig(FIGURES_DIR / "fig_proc1_control_articular.png")
    plt.close(fig)
    return out_path


def generar_grafica_procedimiento_2() -> Path:
    """
    Prueba 2: Dibujo sencillo de la zona de alcance del robot vista de perfil (plano X - Z)
    mostrando claramente las 3 posturas principales de la tabla:
      1. Origen [0°, 0°, 0°] (brazo estirado horizontal -> punta en X=0.20 m, Z=0.06 m)
      2. Home [0°, +45°, -30°] (postura de codo -> punta en X=0.167 m, Z=0.157 m)
      3. Vertical [0°, +90°, 0°] (brazo estirado hacia arriba -> punta en X=0.00 m, Z=0.26 m)
    """
    q2_vals = np.linspace(-90.0, 90.0, 60)
    q3_vals = np.linspace(-90.0, 90.0, 60)
    r_cloud = []
    z_cloud = []
    for q2 in q2_vals:
        for q3 in q3_vals:
            p = forward_kinematics_deg([0.0, q2, q3])
            r_cloud.append(p[0])
            z_cloud.append(p[2])

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 5.4))

    # Zona azul de puntos alcanzables
    ax.scatter(
        r_cloud,
        z_cloud,
        s=8,
        color="#93c5fd",
        alpha=0.45,
        label="Zona de alcance (puntos a los que llega la punta)",
    )

    # Pedestal de la base (de Z=0 a Z=0.06 m)
    ax.plot(
        [0.0, 0.0],
        [0.0, d1],
        color="#334155",
        lw=5,
        label="Base fija (altura d₁ = 0.06 m)",
    )

    # Postura 1: Origen [0°, 0°, 0°]
    ax.plot(
        [0.0, a2, a2 + a3],
        [d1, d1, d1],
        "o-",
        color="#ea580c",
        lw=2.8,
        markersize=7,
        label="1. Postura Origen [0°, 0°, 0°] → Punta en (0.20, 0.06) m",
    )

    # Postura 2: Home [0°, +45°, -30°]
    q2_r, q3_r = np.deg2rad(45.0), np.deg2rad(-30.0)
    elbow_x = a2 * np.cos(q2_r)
    elbow_z = d1 + a2 * np.sin(q2_r)
    tcp_x = elbow_x + a3 * np.cos(q2_r + q3_r)
    tcp_z = elbow_z + a3 * np.sin(q2_r + q3_r)
    ax.plot(
        [0.0, elbow_x, tcp_x],
        [d1, elbow_z, tcp_z],
        "s-",
        color="#16a34a",
        lw=2.8,
        markersize=7,
        label="2. Postura Home [0°, +45°, -30°] → Punta en (0.167, 0.157) m",
    )

    # Postura 3: Vertical [0°, +90°, 0°]
    ax.plot(
        [0.0, 0.0, 0.0],
        [d1, d1 + a2, d1 + a2 + a3],
        "^-",
        color="#2563eb",
        lw=2.8,
        markersize=7,
        label="3. Postura Vertical [0°, +90°, 0°] → Punta en (0.00, 0.26) m",
    )

    ax.axhline(0.0, color="#64748b", lw=1.2, ls="-")
    ax.set_title("Espacio de trabajo del robot (vista de perfil) y posturas de prueba")
    ax.set_xlabel("Distancia hacia delante X (metros)")
    ax.set_ylabel("Altura desde el suelo Z (metros)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.04, 0.24)
    ax.set_ylim(-0.06, 0.29)
    ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig_espacio_trabajo.png"
    fig.savefig(out_path)
    fig.savefig(FIGURES_DIR / "fig_proc2_validacion_dh_espacio.png")
    plt.close(fig)
    return out_path


def _calcular_articulaciones_plano_rz(angles_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve las coordenadas (r, z) de [base, hombro, codo, TCP] para representar las ramas CI."""
    q_rad = np.deg2rad(angles_deg)
    dh_rows = get_dh_table_rad(q_rad)
    A1 = dh_matrix(*dh_rows[0])
    A12 = A1 @ dh_matrix(*dh_rows[1])
    A123 = A12 @ dh_matrix(*dh_rows[2])

    pts_xyz = np.array(
        [
            [0.0, 0.0, 0.0],
            A1[:3, 3],
            A12[:3, 3],
            A123[:3, 3],
        ]
    )
    r_pts = np.sqrt(pts_xyz[:, 0] ** 2 + pts_xyz[:, 1] ** 2)
    z_pts = pts_xyz[:, 2]
    return r_pts, z_pts


def generar_grafica_procedimiento_3() -> Path:
    """
    Prueba 3: Dibujo único y limpio de las dos soluciones geométricas de Cinemática Inversa
    ('down' vs 'up') incluyendo los dos círculos de radio a2 y a3 (como en el vídeo de ejemplo)
    cuyos dos cortes definen las dos posiciones posibles del codo.
    """
    target_xyz = np.array([0.14, 0.05, 0.12])
    sol_down = inverse_kinematics_deg(target_xyz, elbow_config="only_down")
    sol_up = inverse_kinematics_deg(target_xyz, elbow_config="only_up")

    r_down, z_down = _calcular_articulaciones_plano_rz(sol_down)
    r_up, z_up = _calcular_articulaciones_plano_rz(sol_up)
    r_target = float(np.sqrt(target_xyz[0] ** 2 + target_xyz[1] ** 2))
    z_target = float(target_xyz[2])

    fig, ax = plt.subplots(1, 1, figsize=(7.4, 5.5))

    # Dibujar los dos círculos (del hombro con radio a2 y del objetivo con radio a3) como en el vídeo
    ang = np.linspace(0, 2 * np.pi, 300)
    ax.plot(
        0.0 + a2 * np.cos(ang),
        d1 + a2 * np.sin(ang),
        linestyle=":",
        color="#38bdf8",
        lw=1.4,
        alpha=0.85,
        label="Círculo del brazo desde el hombro (radio a₂ = 0.10 m)",
    )
    ax.plot(
        r_target + a3 * np.cos(ang),
        z_target + a3 * np.sin(ang),
        linestyle=":",
        color="#fb923c",
        lw=1.4,
        alpha=0.85,
        label="Círculo del antebrazo desde el objetivo (radio a₃ = 0.10 m)",
    )

    # Base fija
    ax.plot(
        [0.0, 0.0],
        [0.0, d1],
        color="#334155",
        lw=5.0,
        label="Base fija (altura d₁ = 0.06 m)",
    )

    # Rama 'down'
    ax.plot(
        r_down[1:],
        z_down[1:],
        "o-",
        color="#2563eb",
        lw=3.0,
        markersize=8,
        label=f"Corte 1 — Opción 'down' (θ₂={sol_down[1]:+.1f}°, θ₃={sol_down[2]:+.1f}°)",
    )

    # Rama 'up'
    ax.plot(
        r_up[1:],
        z_up[1:],
        "s--",
        color="#ea580c",
        lw=2.8,
        markersize=8,
        label=f"Corte 2 — Opción 'up' (θ₂={sol_up[1]:+.1f}°, θ₃={sol_up[2]:+.1f}°)",
    )

    # Punto objetivo (estrella roja)
    ax.scatter(
        [r_target],
        [z_target],
        color="#dc2626",
        s=160,
        zorder=6,
        marker="*",
        label="Punto objetivo [0.14, 0.05, 0.12] m",
    )

    ax.axhline(0.0, color="#64748b", lw=1.2)
    ax.set_title("Dos soluciones de Cinemática Inversa: intersección de los dos círculos")
    ax.set_xlabel("Distancia horizontal desde la base r (metros)")
    ax.set_ylabel("Altura desde el suelo Z (metros)")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-0.04, 0.21)
    ax.set_ylim(-0.02, 0.20)
    ax.legend(loc="lower left", fontsize=8.0, framealpha=0.95)

    fig.tight_layout()
    out_path = FIGURES_DIR / "fig_cinematica_inversa.png"
    fig.savefig(out_path)
    fig.savefig(FIGURES_DIR / "fig_proc3_cinematica_inversa.png")
    plt.close(fig)
    return out_path


def main() -> None:
    p1 = generar_grafica_procedimiento_1()
    p2 = generar_grafica_procedimiento_2()
    p3 = generar_grafica_procedimiento_3()
    print("Gráficas generadas correctamente:")
    print(f" - {p1}")
    print(f" - {p2}")
    print(f" - {p3}")


if __name__ == "__main__":
    main()
