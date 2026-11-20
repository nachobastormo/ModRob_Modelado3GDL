# ModRob_Modelado3GDL — Robot Manipulador de 3 GDL en Python + MuJoCo

Proyecto de la asignatura **Modelado y Control de Robots (UPV)** que implementa el diseño, simulación física 3D, **Cinemática Directa (Denavit-Hartenberg)**, **Cinemática Inversa (Método Geométrico)** e **Interfaz Gráfica de Control** de un robot manipulador articulado de 3 grados de libertad (configuración RRR).

---

## Guía Rápida para el Profesor (Descargar y Ejecutar en 1 minuto)

Requisitos previos: tener instalado **Python 3.10 o superior** en Windows (o Linux/macOS).

### Paso 1. Clonar el repositorio y entrar en la carpeta
Abra una terminal (**PowerShell** o **CMD**), copie y ejecute cada comando pulse `Enter`:

```powershell
git clone https://github.com/nachobastormo/ModRob_Modelado3GDL.git
```
```powershell
cd ModRob_Modelado3GDL
```
*(Si lo ha descargado como archivo `.zip` desde GitHub, descomprímalo y abra la terminal dentro de la carpeta).*

### Paso 2. Crear el entorno virtual e instalar las librerías
Ejecute primero este comando para crear el entorno virtual `.venv`:
```powershell
python -m venv .venv
```
Y a continuación instale las dependencias (`mujoco`, `numpy` y `matplotlib`):
```powershell
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

## Guía de Pruebas Rápidas desde la Interfaz

Una vez abierto el programa con `.\.venv\Scripts\python.exe main.py`, puede comprobar todas las funcionalidades en pocos segundos:

1. **Control Articular (Cinemática Directa):**
   - Mueva los deslizadores **J1 (Base)**, **J2 (Hombro)** y **J3 (Codo)** o escriba ángulos entre `-90°` y `+90°` y pulse **`Aplicar Ángulos Escritos`**.
   - En el panel inferior (**3. Telemetría**) verá cómo se actualizan las coordenadas $(X, Y, Z)$ medidas en MuJoCo y las calculadas por **Denavit-Hartenberg ($T_3^0$)** con un error de **`0.0000 mm [COINCIDENCIA EXACTA]`**.
   - Marque las casillas **`Mostrar ejes de giro (J1, J2, J3)`** o **`Mostrar sistemas XYZ`** para visualizar en 3D los ejes de las articulaciones y los sistemas de referencia DH.
   - Use los botones superiores **`Ir a Origen [0°, 0°, 0°]`** (brazo estirado horizontal) y **`Posición Home (Codo)`** (`[0°, +45°, -30°]`).

2. **Control Cartesiano (Cinemática Inversa):**
   - En el panel **2. Control Cartesiano**, introduzca por ejemplo:
     - **`X = 0.14`**, **`Y = 0.05`**, **`Z = 0.12`** con configuración **`down`**.
   - Pulse **`1. Calcular Cinemática Inversa (CI)`** (verá los ángulos calculados `[+19.65°, +58.70°, -73.44°]` y la esfera verde colocarse en el destino) y luego pulse **`2. Ejecutar Movimiento al Objetivo`**.
   - Cambie el desplegable de configuración de codo a **`up`** y vuelva a pulsar **`2. Ejecutar Movimiento al Objetivo`**: el robot cambiará a la solución de codo alternativo (`[+19.65°, -14.74°, +73.44°]`) manteniendo la punta exactamente en el mismo punto `[0.14, 0.05, 0.12] m`.

3. **Comprobación de Límites y Objetivos Inalcanzables:**
   - Introduzca una coordenada fuera del alcance máximo de $0.20\text{ m}$ (por ejemplo **`X = 0.30`**, **`Y = 0.00`**, **`Z = 0.10`**) o fuera de los límites articulares (por ejemplo **`X = 0.03`**, **`Y = 0.00`**, **`Z = 0.10`**) y pulse **`1. Calcular Cinemática Inversa (CI)`**.
   - La interfaz detectará que no existe solución válida, mostrará un aviso en rojo explicando la causa física y bloqueará el movimiento.

