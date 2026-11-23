# Verificación de reproducibilidad (Fase 3)

Comprueba que el modelo desplegable reproduce sus resultados fuera de la máquina que lo entrenó
(issue [#41](https://github.com/CarlosHeinze/MLOps_Team2/issues/41)). La referencia es el
snapshot `snapshot-373aa06d7db0` (run `ea0f152256134089863214a3cdd2e4b4`), copiado en
`tests/data/reference.json`:

|Referencia|Valor|
|-|-|
|Modelos|`models:/mibici-departures/1` y `models:/mibici-arrivals/1` (`lgbm\\\_tweedie`, alias `champion`)|
|Digest de la tabla de features|`636e73295b9df173482041fd5cd04ede`|
|Digest de los perfiles|`51c85bee5c1166854a6da55a41631c28`|
|Partición de test|2025-10 a 2026-09 (3,576,680 estación-horas)|
|Imagen|`racj90120hplmnv09/mibici-ml-service:0.1.1` (amd64 y arm64)|
|Tolerancias|`METRIC\\\_RTOL` en `modeling/reproducibility.py`; justificación en el ML Canvas §8|

Cada entorno corre `make reproduce`
(`python -m team2\\\_mlops2026.modeling.reproducibility compare`), que:

1. revisa que la tabla local tenga los digests del snapshot;
2. reentrena cada champion con sus propios parámetros y compara sus métricas de test con la
referencia, y puntúa también el champion exportado;
3. con `--train-runs`, compara las corridas `train-\\\*` locales (`make train`) con las de
referencia;
4. con `--url`, manda las filas de referencia al servicio y compara los pronósticos;
5. guarda el resultado completo con `--save` en `docs/reproducibility/`.

## 1\. Entorno limpio en contenedor

Ejecutado por Javier el 18 de noviembre de 2026. Un contenedor nuevo de
`python:3.12.12-slim-trixie` (la misma base fijada por digest que la imagen) sin nada del equipo
anfitrión: clona la rama, instala con el lock, corre las pruebas y la comparación.
Solo recibe, de solo lectura, la tabla de features (verificada por digest) y los modelos
exportados. El servicio es la imagen `0.1.1` descargada de Docker Hub, en otro contenedor.
El script es `docs/reproducibility/clean\\\_env.sh` y los resultados completos están en
`docs/reproducibility/contenedor-linux-\\\*.json`.

```bash
docker run -d --name svc -p 8000:8000 racj90120hplmnv09/mibici-ml-service:0.1.1
docker run --rm -e BRANCH=<rama> -e SERVICE\\\_URL=http://host.docker.internal:8000 \\\\
    -v "$PWD":/repo:ro -v "$PWD"/data/processed/mibici/features:/features:ro \\\\
    -v "$PWD"/models:/models:ro -v "$PWD"/docs/reproducibility:/out \\\\
    -v "$PWD"/docs/reproducibility/clean\\\_env.sh:/clean\\\_env.sh:ro \\\\
    python:3.12.12-slim-trixie@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c \\\\
    bash /clean\\\_env.sh
```

|Dato|Resultado|
|-|-|
|Anfitrión|MacBook Pro, Apple M4 Pro (12 núcleos); Docker Desktop con 12 CPU y 8 GB|
|Entorno|Linux 7.0.12 (linuxkit) aarch64, glibc 2.41, Python 3.12.12, contenedor|
|Commit|`29448beb3d7a85ed6d0be00987b36c80cb6bf01c`|
|`make requirements` (con `requirements-lock.txt`)|Correcto en 44 s; `.venv` de 1.1 GB; `pip check` sin conflictos|
|`make test`|Correcto (código de salida 0) en 52 s|
|Digests de la tabla|Iguales a los del snapshot|
|Reentrenamiento `departures`|**Reproducido**: métricas dentro de 8.4e-15 (tolerancia `1e-4`), pronósticos dentro de 8.2e-12 viajes; el champion exportado da exactamente las métricas registradas; 112 s|
|Reentrenamiento `arrivals`|**Reproducido**: métricas dentro de 1.5e-13, pronósticos dentro de 6.3e-11 viajes; champion exportado exacto; 41 s|
|Servicio desde la imagen `0.1.1`|**Reproducido**: los pronósticos de referencia con diferencia 0|

En el Mac anfitrión, `make reproduce` dio lo mismo (métricas dentro de 1.4e-14 y 1.1e-13) y las
14 corridas `train-\\\*` coincidieron, aunque ahí se comparan consigo mismas. El reentrenamiento con
4 hilos en lugar de 12 movió los pronósticos hasta 0.003 viajes y las métricas hasta 4.3e-6
(notebook `3.02`), dentro de la tolerancia.

## 2\. Otra máquina (integrante distinto del autor)

Pasos para quien verifique. Necesita Docker, Python 3.12 y la tabla de features del snapshot:

```bash
git checkout master \\\&\\\& git pull
make create\\\_environment \\\&\\\& make requirements      # instala con requirements-lock.txt
make test                                         # la suite completa, sin datos ni MLflow

# La tabla de features con los digests del snapshot. Si ya tienes data/processed de las fases
# 1-2 hasta 2026-09, basta con `make features`; compare revisa los digests y avisa si difieren.
make features

# Los champions exportados, sacados de la imagen publicada (no hace falta el servidor MLflow)
make docker-models                                # copia /app/models de la imagen 0.1.1

# El servicio, desde la imagen de Docker Hub
docker run -d --name mibici-ml-service -p 8000:8000 racj90120hplmnv09/mibici-ml-service:0.1.1

# El pipeline: make mlflow en otra terminal, luego el entrenamiento (\\\~10 min)
make train

# La comparación con la referencia; guarda el resultado para el reporte
.venv/bin/python -m team2\\\_mlops2026.modeling.reproducibility compare \\\\
    --train-runs --url http://127.0.0.1:8000 \\\\
    --save docs/reproducibility/maquina-<nombre>-<objetivo>.json
```

En Windows, `.venv\\\\Scripts\\\\python` en lugar de `.venv/bin/python` y `PYTHON\\\_INTERPRETER` en las
llamadas a `make`. Si la tabla tiene otro digest (por ejemplo, porque incluye octubre de 2026),
hay que usar la del snapshot: los artefactos `snapshot/` del run del snapshot, con acceso al
servidor MLflow de referencia, o la carpeta compartida por el equipo, y pasar `--src` con su ruta.


|Dato|Resultado|
|-|-|
|Integrante, máquina, CPU, RAM||
|Sistema operativo||
|Commit (`git rev-parse HEAD`)||
|`make test`||
|Digests de la tabla (los revisa `compare`)||
|Reentrenamiento `departures` (máx. diferencia relativa de métricas, tolerancia `1e-4`)||
|Reentrenamiento `arrivals`||
|`make train` contra las 14 corridas de referencia||
|Servicio desde la imagen (`--url`)||

## 3\. Integración de Fase 3 (issue 43)

Verificación local del 22 de noviembre de 2026 desde `origin/master` en `a485bb8`,
en macOS 27.2.1 arm64 con Python 3.12.13. El entorno de prueba se creó vacío y se
seleccionó con `PYTHON\_INTERPRETER`.

|Comprobación|Resultado|
|-|-|
|`make requirements` y `python -m pip check`|Instalación limpia correcta. Los 186 paquetes de ejecución instalados coinciden con el lock.|
|`make lint` y `make test`|Ruff pasó. 307 pruebas aprobadas en 72.47 s. Los 35 avisos fueron `Pandas4Warning` de MLflow al serializar ejemplos.|
|Linaje del export|La prueba con SQLite temporal verificó para ambos targets la versión asociada a `champion`, el ID del run y los digests de features y perfiles en `model\_card.json`.|
|Servicio local|`/health`, `/model` y `/openapi.json` respondieron como se documenta. `/predict` y `serving.app check` reprodujeron las tres filas de referencia con diferencia máxima de 0.|
|Comparación ligera|`make compare REPRODUCE\_ARGS="--no-track --no-retrain --url http://127.0.0.1:8043"` aprobó con FastAPI local en el puerto alternativo.|
|Make y monitoreo|`make help` mostró los siete targets de Fase 3 revisados. `make monitor MONITOR\_ARGS=--help` y la prueba de monitoreo con datos sintéticos pasaron.|
|Docker|Se revisaron las recetas de build, run, smoke y push con `make -n`. Falta ejecutar el smoke de la imagen publicada en un equipo con Docker.|
|Windows x64|PyPI resolvió los cinco paquetes exclusivos de Windows para CPython 3.12 x64 en un dry-run. Quedan pendientes la instalación nativa con `make requirements`, `pip check` y la suite en Windows.|

Los pins de Windows agregados al lock son `colorama` 0.4.6, `pywin32` 312,
`pywinpty` 3.1.2, `waitress` 3.0.2 y `win32-setctime` 1.2.1.
El cambio conserva los demás pins y las dependencias directas.
Los pasos para verificar una instalación nativa de Windows están en la sección
[Environment](../README.md#1-environment).

|Dato|Resultado|
|-|-|
|Integrante, máquina, CPU, RAM|Carlos Chavero, HP OmniBook (WSL2), Intel Core Ultra 9 288V (8 CPUs), 15 GiB RAM|
|Sistema operativo|Linux 6.18.33.2-microsoft-standard-WSL2 x86\_64 GNU/Linux (Ubuntu on Windows / WSL2, Python 3.12)|
|Commit (`git rev-parse HEAD`)|59b56cd3b9619852a5e5bde2116ddec35e2cb491|
|`make test`|Correcto (código de salida 0) — 291 passed, 1 skipped en 66.83 s|
|Digests de la tabla (los revisa `compare`)|features\_digest: 75857847f53a78a330c347114f0c8987, profiles\_digest: c9578c9c9963778247a15ce9afb8a9d0 (difieren del snapshot ea0f152256134089863214a3cdd2e4b4: 636e73295b9df173482041fd5cd04ede / 51c85bee5c1166854a6da55a41631c28)|
|Reentrenamiento `departures` (máx. diferencia relativa de métricas, tolerancia `1e-4`)|No ejecutado (--no-retrain por digest de tabla distinto al snapshot ea0f152256134089863214a3cdd2e4b4)|
|Reentrenamiento `arrivals`|No ejecutado (--no-retrain por digest de tabla distinto al snapshot ea0f152256134089863214a3cdd2e4b4)|
|`make train` contra las 14 corridas de referencia|14 corridas completadas localmente; en reproducibility compare --train-runs marcan FAIL (diferencias \~0.019–0.037 frente a tolerancias 1e-09–0.005) debido a que los datos locales tienen digests distintos al snapshot|
|Servicio desde la imagen (`--url`)|Reproducido (OK: diferencia 0 en \[http://127.0.0.1:8000](http://127.0.0.1:8000) con racj90120hplmnv09/mibici-ml-service:0.1.1; reporte en docs/reproducibility/maquina-carlos-wsl.json)|




