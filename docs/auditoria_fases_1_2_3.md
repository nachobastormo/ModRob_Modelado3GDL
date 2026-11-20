# Auditoría y Validación del Proyecto — Fases 1, 2 y 3

**Asignatura:** Fundamentos del Modelado de Robots (UD1 — M11: Cinemática directa e inversa, UPV)  
**Entorno:** Windows + Python 3.13.14 (`.venv`) + MuJoCo 3.17.0 + NumPy 2.5.4 + Tkinter  

---

## 1. Verificación de Requisitos Académicos

| Requisito de la práctica | Estado | Archivo donde se implementa |
| :--- | :---: | :--- |
| Robot manipulador con al menos 3 GDL | **CUMPLIDO** | `model/robot.xml` (3 articulaciones revolutas: base, hombro, codo) |
| Modelo tridimensional articulado en MuJoCo | **CUMPLIDO** | `model/robot.xml` y `src/robot_control.py` |
| Cinemática directa según apuntes de clase (Método DH) | **CUMPLIDO** | `src/kinematics.py` (`dh_matrix`, `get_dh_table_rad`, `forward_kinematics_deg`) |
| Cinemática inversa según apuntes de clase (Método Geométrico) | **CUMPLIDO** | `src/kinematics.py` (`inverse_kinematics_deg`) |
| Interfaz gráfica (GUI) conectada al simulador | **CUMPLIDO** | `src/gui.py` (`RobotControlApp`) |
| Control mediante ángulos articulares ($q_1, q_2, q_3$) | **CUMPLIDO** | `src/gui.py` (deslizadores + campos numéricos con límites $[-90^\circ, +90^\circ]$) |
| Control mediante coordenadas cartesianas ($X, Y, Z$) | **CUMPLIDO** | `src/gui.py` (cálculo de CI, selector de codo y ejecución del movimiento) |
| Coherencia entre cálculos matemáticos, GUI y MuJoCo | **CUMPLIDO** | `tests/test_kinematics.py` y `tests/test_gui.py` |

---

## 2. Coherencia con los Apuntes de la Asignatura (UD1 — M11)

Se ha contrastado todo el módulo `src/kinematics.py` con las transparencias de clase del Prof. J. Ernesto Solanes (*UD1. Fundamentos del modelado de robots — Módulo M11: Cinemática directa e inversa*):

1. **Nomenclatura idéntica a la de clase**:
   - Variables articulares revolutas: $q_i = \theta_i$ ($i = 1, 2, 3$).
   - Parámetros geométricos DH: $d_1 = 0.060\text{ m}$, $a_2 = 0.100\text{ m}$, $a_3 = 0.100\text{ m}$.
   - Coordenadas del punto extremo: $(x_c, y_c, z_c)$ y proyecciones $r^2 = x_c^2 + y_c^2$, $s = z_c - d_1$.
2. **Cinemática Directa por el Método de Denavit-Hartenberg (Diapositivas 15 a 51)**:
   - Se calcula cada matriz homogénea $A_i = \text{Rot}_{z,\theta_i}\cdot \text{Trans}_{z,d_i}\cdot \text{Trans}_{x,a_i}\cdot \text{Rot}_{x,\alpha_i}$.
   - Se obtiene $T_3^0 = A_1(q_1)\, A_2(q_2)\, A_3(q_3)$ y se extrae el vector de posición $o_3^0 = [x_c, y_c, z_c]^T$ de la última columna.
3. **Cinemática Inversa por el Método Geométrico (Diapositivas 61 a 71)**:
   - Primera articulación proyectando en $x_0 - y_0$: $\theta_1 = \text{atan2}(y_c, x_c)$ (y caso singular $x_c = y_c = 0$).
   - Tercera articulación por el Teorema del Coseno en el plano $z_0 - r$:
     $$D := \frac{x_c^2 + y_c^2 + (z_c - d_1)^2 - a_2^2 - a_3^2}{2\, a_2\, a_3}, \qquad \cos\theta_3 = D, \qquad \sin\theta_3 = \pm\sqrt{1 - D^2}$$
     Contemplando las dos soluciones (**codo arriba y codo abajo**).
   - Segunda articulación mediante el ángulo auxiliar $\beta$:
     $$\tan\beta = \frac{a_3\sin\theta_3}{a_2 + a_3\cos\theta_3}, \qquad \theta_2 = \text{atan2}(s, r) - \beta$$

---

## 3. Revisión del Modelo en MuJoCo (`model/robot.xml`)

- **Jerarquía padre-hijo verificada**:
  `worldbody` $\to$ `base_link` $\to$ `turret_link` (`joint_base_yaw`, eje $0\;0\;1$) $\to$ `upper_arm_link` (`joint_shoulder_pitch`, eje $0\;-1\;0$) $\to$ `forearm_link` (`joint_elbow_pitch`, eje $0\;-1\;0$) $\to$ `end_effector_link` (contiene `site_end_effector`).
- **Controlabilidad real**: Tres actuadores de posición (`servo_base_yaw`, `servo_shoulder_pitch`, `servo_elbow_pitch`) con límites $[-90^\circ, +90^\circ]$.
- **Comprobación experimental (`python main.py --verify`)**:
  - Posición inicial (`Home` $= [0^\circ, 0^\circ, 0^\circ]$): $\text{TCP} = [+0.2000, +0.0000, +0.0600]\text{ m}$.
  - Giro independiente de $J_1 \to +45^\circ$: $\text{TCP} = [+0.1414, +0.1414, +0.0582]\text{ m}$.
  - Elevación independiente de $J_2 \to +40^\circ$: $\text{TCP} = [+0.1541, +0.0000, +0.1875]\text{ m}$.
  - Flexión independiente de $J_3 \to -45^\circ$: $\text{TCP} = [+0.1701, +0.0000, -0.0121]\text{ m}$.
  - Retorno a `Home`: $\text{TCP} = [+0.2000, +0.0000, +0.0600]\text{ m}$.

---

## 4. Resultados de las Pruebas Automatizadas de Cinemática y GUI

Comando ejecutado:
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```
**Resultado: 10 tests ejecutados en 1.43 s — TODOS OK.**

### 4.1 Tabla de Contraste: Cinemática Directa (DH) vs MuJoCo (`test_01`)

| Ángulos articulares $[q_1, q_2, q_3]$ | Posición DH $o_3^0\text{ (m)}$ | Posición MuJoCo `site_xpos` $\text{(m)}$ | Error Euclídeo $\text{(m)}$ |
| :--- | :--- | :--- | :---: |
| $[0^\circ, 0^\circ, 0^\circ]$ | $[0.20000, 0.00000, 0.06000]$ | $[0.20000, 0.00000, 0.06000]$ | $2.78 \times 10^{-17}$ |
| $[90^\circ, 0^\circ, 0^\circ]$ | $[0.00000, 0.20000, 0.06000]$ | $[0.00000, 0.20000, 0.06000]$ | $5.68 \times 10^{-17}$ |
| $[0^\circ, 90^\circ, 0^\circ]$ | $[0.00000, 0.00000, 0.26000]$ | $[0.00000, 0.00000, 0.26000]$ | $5.82 \times 10^{-17}$ |
| $[0^\circ, 45^\circ, -45^\circ]$ | $[0.17071, 0.00000, 0.13071]$ | $[0.17071, 0.00000, 0.13071]$ | $5.57 \times 10^{-17}$ |
| $[30^\circ, 45^\circ, -30^\circ]$ | $[0.14489, 0.08365, 0.15659]$ | $[0.14489, 0.08365, 0.15659]$ | $2.78 \times 10^{-17}$ |
| $[-45^\circ, 20^\circ, 35^\circ]$ | $[0.10700, -0.10700, 0.17612]$ | $[0.10700, -0.10700, 0.17612]$ | $7.97 \times 10^{-17}$ |
| $[-75^\circ, 80^\circ, -85^\circ]$ | $[0.03028, -0.11300, 0.14977]$ | $[0.03028, -0.11300, 0.14977]$ | $3.94 \times 10^{-17}$ |

### 4.2 Validación de Cinemática Inversa (`test_02`, `test_03`, `test_04`, `test_05`)
- **Reconstrucción exacta**: En las 7 posturas de prueba, los ángulos obtenidos por `inverse_kinematics_deg` colocan el efector final en MuJoCo con un error inferior a $10^{-16}\text{ m}$.
- **Múltiples soluciones**: Para el objetivo $[0.14, 0.05, 0.12]\text{ m}$, se obtienen las dos configuraciones válidas:
  - Codo abajo (`only_down`): $[+19.65^\circ, +58.70^\circ, -73.44^\circ]$
  - Codo arriba (`only_up`): $[+19.65^\circ, -14.74^\circ, +73.44^\circ]$
- **Rechazo de objetivos inalcanzables**:
  - $[0.35, 0.00, 0.06]\text{ m} \to$ Rechazado ($D = 5.125 > 1$, fuera de alcance).
  - $[0.00, 0.00, 0.35]\text{ m} \to$ Rechazado ($D = 3.205 > 1$, por encima de la altura máxima).
  - $[-0.15, 0.00, 0.06]\text{ m} \to$ Rechazado (viola límites $[-90^\circ, +90^\circ]$).
  - $[0.02, 0.00, 0.06]\text{ m} \to$ Rechazado ($D = -0.980$, exige flexión de codo $|q_3| > 90^\circ$).

### 4.3 Validación de la Interfaz Gráfica (`test_gui.py`)
- Verificado el movimiento real de las articulaciones mediante deslizadores y cajas de texto, el recorte automático a $[-90^\circ, +90^\circ]$, el cálculo y ejecución de objetivos cartesianos $(X, Y, Z)$, el aviso de error ante puntos inalcanzables, el botón de retorno a `Home` y la sincronización en vivo con el visor 3D limpio (`show_left_ui=False`, `show_right_ui=False`).

---

## 5. Limpieza Realizada en el Código

Durante esta auditoría se han eliminado elementos redundantes para dejar el código limpio y directo:
1. En `src/kinematics.py`: eliminado el parámetro `enforce_limits` que no se utilizaba fuera del módulo (los límites $[-90^\circ, +90^\circ]$ del robot siempre se aplican).
2. En `src/gui.py`: eliminados el atributo no utilizado `self.last_ik_solution_deg`, el parámetro innecesario `width` en `_create_button` y las tuplas redundantes en `joint_titles`.
3. No existe duplicación de fórmulas cinemáticas: tanto `main.py` como `src/gui.py` delegan todos los cálculos exclusivamente en `src/kinematics.py`.

---

## 6. Singularidades y Limitaciones Relevantes del Manipulador

1. **Singularidad interior en el eje vertical de la base ($x_c = 0,\; y_c = 0$)**:
   - Tal como se explica en la **diapositiva 65** de los apuntes de clase, cuando el efector final se sitúa exactamente sobre el eje $z_0$ ($r = \sqrt{x_c^2 + y_c^2} = 0$, por ejemplo en el punto $[0, 0, 0.26]\text{ m}$ con $q_2 = 90^\circ, q_3 = 0^\circ$), cualquier giro $\theta_1$ de la base mantiene el punto extremo en el mismo lugar (infinitas soluciones para $\theta_1$). En `src/kinematics.py` se resuelve fijando por convención $\theta_1 = 0^\circ$.
2. **Singularidad de frontera por extensión completa ($q_3 = 0^\circ$)**:
   - Cuando el brazo y el antebrazo están completamente alineados ($D = \cos\theta_3 = 1$, distancia al hombro igual a $a_2 + a_3 = 0.200\text{ m}$), las dos soluciones de codo (`up` y `down`) convergen en una única solución ($\theta_3 = 0^\circ$).
3. **Limitación del espacio de trabajo por límites articulares $[-90^\circ, +90^\circ]$**:
   - Debido al recorrido típico de $\pm 90^\circ$ de cada servo:
     - El robot no puede alcanzar puntos con distancia al hombro menor que $\sqrt{a_2^2 + a_3^2} = \sqrt{0.10^2 + 0.10^2} \approx 0.1414\text{ m}$ (ya que requeriría $|\theta_3| > 90^\circ$).
     - Tampoco puede alcanzar puntos situados en el semiespacio trasero ($x_c < 0$ cuando la proyección radial es positiva, pues exigiría $|\theta_1| > 90^\circ$).
4. **Efecto estacionario de la gravedad en modo dinámico vs Modo Cinemático Exacto**:
   - Cuando la casilla *"Modo Cinemático Exacto"* está desmarcada, los servos proporcionales (`position` con `kp="6.0"`) sostienen el peso del brazo bajo gravedad ($-9.81\text{ m/s}^2$), presentando una pequeña desviación elástica natural ($\approx 0.4^\circ$). En ambos modos, el error entre la posición de MuJoCo y la calculada por DH para los ángulos medidos es de $0.0000\text{ mm}$ (`[✓ COINCIDENCIA EXACTA (< 0.05 mm)]`).
