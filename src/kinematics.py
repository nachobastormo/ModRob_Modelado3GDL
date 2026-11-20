"""
Módulo de Cinemática Directa e Inversa (Fase 2) para el manipulador de 3 GDL (RRR).

Implementado siguiendo los apuntes de la asignatura:
"UD1. Fundamentos del modelado de robots — M11. Cinemática directa e inversa"
(Prof. J. Ernesto Solanes, Universitat Politècnica de València - UPV).

================================================================================
1. CINEMÁTICA DIRECTA: MÉTODO DE DENAVIT-HARTENBERG (DH) (Diapositivas 15 a 51)
================================================================================
Matriz de transformación homogénea de cada eslabón i:
    A_i = Rot_{z, theta_i} * Trans_{z, d_i} * Trans_{x, a_i} * Rot_{x, alpha_i}

Matriz de transformación homogénea del extremo respecto a la base (o_0 x_0 y_0 z_0):
    T_3^0(q_1, q_2, q_3) = A_1(q_1) * A_2(q_2) * A_3(q_3) = [ R_3^0   o_3^0 ]
                                                            [ 0_{1x3}   1   ]

Tabla de parámetros DH de nuestro manipulador 3 GDL:
    +---+---------------+-----------+---------------+---------------+
    | i |      d_i      |  theta_i  |      a_i      |    alpha_i    |
    +---+---------------+-----------+---------------+---------------+
    | 1 | d_1 = 0.060 m |   theta_1 |       0       |    +pi / 2    |
    | 2 |       0       |   theta_2 | a_2 = 0.100 m |       0       |
    | 3 |       0       |   theta_3 | a_3 = 0.100 m |       0       |
    +---+---------------+-----------+---------------+---------------+

================================================================================
2. CINEMÁTICA INVERSA: MÉTODO GEOMÉTRICO (Diapositivas 61 a 71)
================================================================================
Dado el punto objetivo del efector final o_c = [x_c, y_c, z_c]^T:
1) Primera variable articular (q_1 = theta_1) proyectando sobre x_0 - y_0 (Diap. 63-65):
       theta_1 = atan2(y_c, x_c)   (o bien pi + atan2(y_c, x_c))
2) Proyección sobre el plano z_0 - r (Diap. 66-68):
       r^2 = x_c^2 + y_c^2,    s = z_c - d_1
3) Tercera variable articular (q_3 = theta_3) por Teorema del Coseno (Diap. 67-70):
       D := (x_c^2 + y_c^2 + (z_c - d_1)^2 - a_2^2 - a_3^2) / (2 * a_2 * a_3)
       cos(theta_3) = D,    sin(theta_3) = +-sqrt(1 - D^2)
       theta_3 = atan2(sin(theta_3), cos(theta_3))
4) Segunda variable articular (q_2 = theta_2) (Diap. 71):
       tan(beta) = (a_3 * sin(theta_3)) / (a_2 + a_3 * cos(theta_3))
       tan(theta_2 + beta) = s / r
       theta_2 = atan2(s, r) - atan2(a_3 * sin(theta_3), a_2 + a_3 * cos(theta_3))
"""

from typing import Sequence
import numpy as np


# Parámetros de Denavit-Hartenberg (notación exacta de los apuntes UPV, en metros)
d1: float = 0.060  # Distancia entre o_0 y o_1 a lo largo del eje z_0
a2: float = 0.100  # Longitud del eslabón 2 (hombro -> codo)
a3: float = 0.100  # Longitud del eslabón 3 hasta el TCP (0.080 m antebrazo + 0.020 m puntero)

# Límites articulares [min, max] en grados para q_1, q_2, q_3
JOINT_LIMITS_DEG: np.ndarray = np.array(
    [
        [-90.0, 90.0],
        [-90.0, 90.0],
        [-90.0, 90.0],
    ],
    dtype=float,
)


def dh_matrix(d_i: float, theta_i: float, a_i: float, alpha_i: float) -> np.ndarray:
    """
    Construye la matriz de transformación homogénea 4x4 de Denavit-Hartenberg A_i(q_i)
    según la diapositiva 15 de los apuntes:
        A_i = Rot_{z, theta_i} * Trans_{z, d_i} * Trans_{x, a_i} * Rot_{x, alpha_i}
    """
    ct = float(np.cos(theta_i))
    st = float(np.sin(theta_i))
    ca = float(np.cos(alpha_i))
    sa = float(np.sin(alpha_i))

    return np.array(
        [
            [ct, -st * ca,  st * sa, a_i * ct],
            [st,  ct * ca, -ct * sa, a_i * st],
            [0.0,      sa,       ca,      d_i],
            [0.0,     0.0,      0.0,      1.0],
        ],
        dtype=float,
    )


def get_dh_table_rad(angles_rad: Sequence[float]) -> list[tuple[float, float, float, float]]:
    """
    Devuelve las filas de la tabla de parámetros DH [(d_i, theta_i, a_i, alpha_i)]
    para i = 1, 2, 3 (Paso 6 de la metodología DH, diapositivas 49-50).
    """
    theta1, theta2, theta3 = angles_rad
    return [
        (d1, float(theta1), 0.0, np.pi / 2.0),
        (0.0, float(theta2), a2, 0.0),
        (0.0, float(theta3), a3, 0.0),
    ]


def forward_kinematics_matrix_deg(angles_deg: Sequence[float]) -> np.ndarray:
    """
    Calcula la matriz de transformación homogénea T_3^0 = A_1 * A_2 * A_3 (4x4)
    mediante el método de Denavit-Hartenberg (diapositivas 12, 13 y 15).
    """
    q = np.asarray(angles_deg, dtype=float)
    if q.shape != (3,):
        raise ValueError("Se esperaban exactamente 3 ángulos articulares [q1, q2, q3].")

    q_rad = np.deg2rad(q)
    dh_rows = get_dh_table_rad(q_rad)

    A1 = dh_matrix(*dh_rows[0])
    A2 = dh_matrix(*dh_rows[1])
    A3 = dh_matrix(*dh_rows[2])

    return A1 @ A2 @ A3


def forward_kinematics_deg(angles_deg: Sequence[float]) -> np.ndarray:
    """
    Obtiene el vector de posición del efector final respecto a la base o_3^0 = [x_c, y_c, z_c]
    (en metros) a partir de la última columna de la matriz DH T_3^0 (diapositiva 13).
    """
    T_0_3 = forward_kinematics_matrix_deg(angles_deg)
    return T_0_3[:3, 3].copy()


def check_joint_limits_deg(
    angles_deg: Sequence[float], tol_deg: float = 1e-4
) -> bool:
    """Comprueba si los ángulos [theta_1, theta_2, theta_3] están dentro de [-90°, +90°]."""
    q = np.asarray(angles_deg, dtype=float)
    if q.shape != (3,):
        raise ValueError("Se esperaban exactamente 3 ángulos articulares [q1, q2, q3].")
    lower_ok = np.all(q >= (JOINT_LIMITS_DEG[:, 0] - tol_deg))
    upper_ok = np.all(q <= (JOINT_LIMITS_DEG[:, 1] + tol_deg))
    return bool(lower_ok and upper_ok)


def inverse_kinematics_deg(
    target_xyz: Sequence[float],
    elbow_config: str = "down",
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Calcula la Cinemática Inversa mediante el Método Geométrico de los apuntes (diapositivas 61 a 71).
    Devuelve los ángulos [theta_1, theta_2, theta_3] en grados dentro de [-90°, +90°],
    o lanza ValueError si el objetivo no es alcanzable.
    """
    pos = np.asarray(target_xyz, dtype=float)
    if pos.shape != (3,):
        raise ValueError("El objetivo debe ser un vector de 3 coordenadas [x_c, y_c, z_c].")

    xc, yc, zc = float(pos[0]), float(pos[1]), float(pos[2])

    # 1) Proyección en el plano z_0 - r (Diapositiva 68)
    r_sq = xc**2 + yc**2
    r = float(np.sqrt(r_sq))
    s = zc - d1

    # 2) Parámetro D por el Teorema del Coseno (Diapositiva 70)
    D = (r_sq + s**2 - a2**2 - a3**2) / (2.0 * a2 * a3)

    if abs(D) > 1.0 + tol:
        dist = float(np.sqrt(r_sq + s**2))
        raise ValueError(
            f"Objetivo fuera del espacio de trabajo: D={D:.3f} (|D|>1), "
            f"distancia al hombro {dist:.3f} m > alcance máximo {a2 + a3:.3f} m."
        )
    D_clipped = float(np.clip(D, -1.0, 1.0))

    # Dos soluciones para theta_3: codo abajo / codo arriba (Diapositiva 70)
    if elbow_config == "down":
        elbow_signs = [-1.0, 1.0]
    elif elbow_config == "up":
        elbow_signs = [1.0, -1.0]
    elif elbow_config == "only_down":
        elbow_signs = [-1.0]
    elif elbow_config == "only_up":
        elbow_signs = [1.0]
    else:
        raise ValueError(
            f"Configuración de codo no válida: '{elbow_config}'. Usa 'down', 'up', 'only_down' u 'only_up'."
        )

    # 3) Primera variable articular theta_1 sobre el plano x_0 - y_0 (Diapositiva 64-65)
    if r < 1e-9:
        theta1_candidates = [(0.0, 0.0)]
    else:
        theta1_sol1 = float(np.arctan2(yc, xc))
        theta1_sol2 = float(np.arctan2(-yc, -xc))
        theta1_candidates = [(theta1_sol1, r), (theta1_sol2, -r)]

    sin_theta3_abs = float(np.sqrt(max(0.0, 1.0 - D_clipped**2)))
    valid_solutions: list[np.ndarray] = []

    for theta1_rad, r_proj in theta1_candidates:
        for sign_e in elbow_signs:
            sin_theta3 = sign_e * sin_theta3_abs
            cos_theta3 = D_clipped
            theta3_rad = float(np.arctan2(sin_theta3, cos_theta3))

            # 4) Segunda variable articular theta_2 (Diapositiva 71)
            beta = float(np.arctan2(a3 * sin_theta3, a2 + a3 * cos_theta3))
            theta2_rad = float(np.arctan2(s, r_proj) - beta)

            sol_deg = np.rad2deg([theta1_rad, theta2_rad, theta3_rad])

            pos_reconstructed = forward_kinematics_deg(sol_deg)
            if np.linalg.norm(pos_reconstructed - pos) > 1e-4:
                continue

            if check_joint_limits_deg(sol_deg, tol_deg=1e-3):
                sol_deg = np.clip(
                    sol_deg, JOINT_LIMITS_DEG[:, 0], JOINT_LIMITS_DEG[:, 1]
                )
                valid_solutions.append(sol_deg)

    if not valid_solutions:
        raise ValueError(
            f"Objetivo [{xc:.3f}, {yc:.3f}, {zc:.3f}] m inalcanzable: requiere ángulos fuera "
            f"de los límites [-90°, +90°]."
        )

    return valid_solutions[0]
