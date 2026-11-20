# ModRob_Modelado3GDL — Robot Manipulador de 3 GDL en Python + MuJoCo

Proyecto de la asignatura **Modelado y Control de Robots (UPV)** que implementa el diseño, simulación física 3D, **Cinemática Directa (Denavit-Hartenberg)**, **Cinemática Inversa (Método Geométrico)** e **Interfaz Gráfica de Control** de un robot manipulador articulado de 3 grados de libertad (configuración RRR).

---

## 🚀 Guía Rápida para el Profesor (Descargar y Ejecutar en 1 minuto)

Requisitos previos: tener instalado **Python 3.10 o superior** en Windows (o Linux/macOS).

### Paso 1. Clonar o descargar el repositorio
Abra una terminal (**PowerShell** o **CMD**) y ejecute:
```powershell
git clone https://github.com/nachobastormo/ModRob_Modelado3GDL.git
cd ModRob_Modelado3GDL
```
*(Si lo ha descargado como archivo `.zip` desde GitHub, descomprímalo y abra la terminal dentro de la carpeta).*

### Paso 2. Crear el entorno virtual e instalar dependencias
Ejecute estos dos comandos para crear un entorno limpio e instalar `mujoco`, `numpy` y `matplotlib`:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### Paso 3. Iniciar el Simulador 3D y la Interfaz Gráfica
```powershell
.\.venv\Scripts\python.exe main.py
```
Al ejecutarlo se abrirán automáticamente dos ventanas sincronizadas en tiempo real:
1. **El visor 3D de MuJoCo** con el robot manipulador en su postura inicial *Home* (`[0°, +45°, -30°]`) y la esfera verde transparente que indica el punto objetivo.
2. **El Panel de Control interactivo** desde donde se puede mover el robot.

---

## 🧪 Guía de Pruebas Rápidas desde la Interfaz

Una vez abierto el programa con `.\.venv\Scripts\python.exe main.py`, puede comprobar todas las funcionalidades en pocos segundos:

1. **Control Articular (Cinemática Directa):**
   - Mueva los deslizadores **J1 (Base)**, **J2 (Hombro)** y **J3 (Codo)** o escriba ángulos entre `-90°` y `+90°` y pulse **`Aplicar Ángulos Escritos`**.
   - En el panel inferior (**3. Telemetría**) verá cómo se actualizan las coordenadas $(X, Y, Z)$ medidas en MuJoCo y las calculadas por **Denavit-Hartenberg ($T_3^0$)** con un error de **`0.0000 mm [✓ COINCIDENCIA EXACTA]`**.
   - Marque las casillas **`Mostrar ejes de giro (J1, J2, J3)`** o **`Mostrar sistemas XYZ`** para visualizar en 3D los ejes de las articulaciones y los sistemas de referencia DH.
   - Use los botones superiores **`⊙ Ir a Origen [0°, 0°, 0°]`** (brazo estirado horizontal) y **`⌂ Posición Home (Codo)`** (`[0°, +45°, -30°]`).

2. **Control Cartesiano (Cinemática Inversa):**
   - En el panel **2. Control Cartesiano**, introduzca por ejemplo:
     - **`X = 0.14`**, **`Y = 0.05`**, **`Z = 0.12`** con configuración **`down`**.
   - Pulse **`1. Calcular Cinemática Inversa (CI)`** (verá los ángulos calculados `[+19.65°, +58.70°, -73.44°]` y la esfera verde colocarse en el destino) y luego pulse **`▶ 2. Ejecutar Movimiento al Objetivo`**.
   - Cambie el desplegable de configuración de codo a **`up`** y vuelva a pulsar **`▶ 2. Ejecutar Movimiento al Objetivo`**: el robot cambiará a la solución de codo alternativo (`[+19.65°, -14.74°, +73.44°]`) manteniendo la punta exactamente en el mismo punto `[0.14, 0.05, 0.12] m`.

3. **Comprobación de Límites y Objetivos Inalcanzables:**
   - Introduzca una coordenada fuera del alcance máximo de $0.20\text{ m}$ (por ejemplo **`X = 0.30`**, **`Y = 0.00`**, **`Z = 0.10`**) o fuera de los límites articulares (por ejemplo **`X = 0.03`**, **`Y = 0.00`**, **`Z = 0.10`**) y pulse **`1. Calcular Cinemática Inversa (CI)`**.
   - La interfaz detectará que no existe solución válida, mostrará un aviso en rojo explicando la causa física y bloqueará el movimiento.

---

## ✅ Ejecución de Tests Automáticos (Opcional)

Para verificar automáticamente por consola la coincidencia entre Denavit-Hartenberg y MuJoCo, la Cinemática Inversa y la Interfaz Gráfica (10 tests unitarios):
```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

---

## 📐 Parámetros de Denavit-Hartenberg del Robot

El robot está modelado en [`model/robot.xml`](model/robot.xml) y su cinemática está implementada en [`src/kinematics.py`](src/kinematics.py) siguiendo la notación de los apuntes de la asignatura (*UD1 — M11: Cinemática directa e inversa*):

| Eslabón $i$ | $d_i\text{ (m)}$ | $\theta_i\text{ (rad)}$ | $a_i\text{ (m)}$ | $\alpha_i\text{ (rad)}$ | Límites |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Base – Hombro)** | $d_1 = 0.060$ | $\theta_1^*$ | $0$ | $+\pi/2$ | $[-90^\circ, +90^\circ]$ |
| **2 (Brazo superior)** | $0$ | $\theta_2^*$ | $a_2 = 0.100$ | $0$ | $[-90^\circ, +90^\circ]$ |
| **3 (Antebrazo + Punta)** | $0$ | $\theta_3^*$ | $a_3 = 0.100$ | $0$ | $[-90^\circ, +90^\circ]$ |

---

## 📁 Estructura del Repositorio

```text
ModRob_Modelado3GDL/
├── model/
│   └── robot.xml               # Modelo 3D MJCF del manipulador de 3 GDL
├── src/
│   ├── robot_control.py        # Controlador de bajo nivel en MuJoCo (RobotArmController)
│   ├── kinematics.py           # Cinemática Directa (DH) e Inversa (Geométrica)
│   └── gui.py                  # Interfaz Gráfica de Usuario en Tkinter sincronizada con MuJoCo
├── scripts/
│   └── generar_graficas.py     # Script que genera las gráficas de resultados de la memoria
├── tests/
│   ├── test_kinematics.py      # Tests unitarios de Cinemática Directa e Inversa vs MuJoCo
│   └── test_gui.py             # Tests unitarios de la Interfaz Gráfica
├── docs/
│   ├── informe_trabajo.tex     # Código fuente LaTeX del informe técnico
│   └── figuras/                # Gráficas y capturas del informe
├── main.py                     # Punto de entrada principal (abre GUI + Visor 3D)
├── requirements.txt            # Dependencias del proyecto (mujoco, numpy, matplotlib)
└── README.md                   # Instrucciones de instalación y uso
```
