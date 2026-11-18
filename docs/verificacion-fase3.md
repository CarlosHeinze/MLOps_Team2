# Verificación de reproducibilidad (Fase 3)

Comprueba que el modelo desplegable reproduce sus resultados fuera de la máquina que lo entrenó
(issue [#41](https://github.com/CarlosHeinze/MLOps_Team2/issues/41)). La referencia es el
snapshot `snapshot-373aa06d7db0` (run `ea0f152256134089863214a3cdd2e4b4`), copiado en
`tests/data/reference.json`:

| Referencia | Valor |
|---|---|
| Modelos | `models:/mibici-departures/1` y `models:/mibici-arrivals/1` (`lgbm_tweedie`, alias `champion`) |
| Digest de la tabla de features | `636e73295b9df173482041fd5cd04ede` |
| Digest de los perfiles | `51c85bee5c1166854a6da55a41631c28` |
| Partición de test | 2025-10 a 2026-09 (3,576,680 estación-horas) |
| Imagen | `racj90120hplmnv09/mibici-ml-service:0.1.1` (amd64 y arm64) |
| Tolerancias | `METRIC_RTOL` en `modeling/reproducibility.py`; justificación en el ML Canvas §8 |

Cada entorno corre `make reproduce`
(`python -m team2_mlops2026.modeling.reproducibility compare`), que:

1. revisa que la tabla local tenga los digests del snapshot;
2. reentrena cada champion con sus propios parámetros y compara sus métricas de test con la
   referencia, y puntúa también el champion exportado;
3. con `--train-runs`, compara las corridas `train-*` locales (`make train`) con las de
   referencia;
4. con `--url`, manda las filas de referencia al servicio y compara los pronósticos;
5. guarda el resultado completo con `--save` en `docs/reproducibility/`.

## 1. Entorno limpio en contenedor

Ejecutado por Javier el 18 de noviembre de 2026. Un contenedor nuevo de
`python:3.12.12-slim-trixie` (la misma base fijada por digest que la imagen) sin nada del equipo
anfitrión: clona la rama, instala con el lock, corre las pruebas y la comparación.
Solo recibe, de solo lectura, la tabla de features (verificada por digest) y los modelos
exportados. El servicio es la imagen `0.1.1` descargada de Docker Hub, en otro contenedor.
El script es `docs/reproducibility/clean_env.sh` y los resultados completos están en
`docs/reproducibility/contenedor-linux-*.json`.

```bash
docker run -d --name svc -p 8000:8000 racj90120hplmnv09/mibici-ml-service:0.1.1
docker run --rm -e BRANCH=<rama> -e SERVICE_URL=http://host.docker.internal:8000 \
    -v "$PWD":/repo:ro -v "$PWD"/data/processed/mibici/features:/features:ro \
    -v "$PWD"/models:/models:ro -v "$PWD"/docs/reproducibility:/out \
    -v "$PWD"/docs/reproducibility/clean_env.sh:/clean_env.sh:ro \
    python:3.12.12-slim-trixie@sha256:f3fa41d74a768c2fce8016b98c191ae8c1bacd8f1152870a3f9f87d350920b7c \
    bash /clean_env.sh
```

| Dato | Resultado |
|---|---|
| Anfitrión | MacBook Pro, Apple M4 Pro (12 núcleos); Docker Desktop con 12 CPU y 8 GB |
| Entorno | Linux 7.0.12 (linuxkit) aarch64, glibc 2.41, Python 3.12.12, contenedor |
| Commit | `29448beb3d7a85ed6d0be00987b36c80cb6bf01c` |
| `make requirements` (con `requirements-lock.txt`) | Correcto en 44 s; `.venv` de 1.1 GB; `pip check` sin conflictos |
| `make test` | Correcto (código de salida 0) en 52 s |
| Digests de la tabla | Iguales a los del snapshot |
| Reentrenamiento `departures` | **Reproducido**: métricas dentro de 8.4e-15 (tolerancia `1e-4`), pronósticos dentro de 8.2e-12 viajes; el champion exportado da exactamente las métricas registradas; 112 s |
| Reentrenamiento `arrivals` | **Reproducido**: métricas dentro de 1.5e-13, pronósticos dentro de 6.3e-11 viajes; champion exportado exacto; 41 s |
| Servicio desde la imagen `0.1.1` | **Reproducido**: los pronósticos de referencia con diferencia 0 |

En el Mac anfitrión, `make reproduce` dio lo mismo (métricas dentro de 1.4e-14 y 1.1e-13) y las
14 corridas `train-*` coincidieron, aunque ahí se comparan consigo mismas. El reentrenamiento con
4 hilos en lugar de 12 movió los pronósticos hasta 0.003 viajes y las métricas hasta 4.3e-6
(notebook `3.02`), dentro de la tolerancia.

## 2. Otra máquina (integrante distinto del autor)

Pasos para quien verifique. Necesita Docker, Python 3.12 y la tabla de features del snapshot:

```bash
git checkout master && git pull
make create_environment && make requirements      # instala con requirements-lock.txt
make test                                         # la suite completa, sin datos ni MLflow

# La tabla de features con los digests del snapshot. Si ya tienes data/processed de las fases
# 1-2 hasta 2026-09, basta con `make features`; compare revisa los digests y avisa si difieren.
make features

# Los champions exportados, sacados de la imagen publicada (no hace falta el servidor MLflow)
make docker-models                                # copia /app/models de la imagen 0.1.1

# El servicio, desde la imagen de Docker Hub
docker run -d --name mibici-ml-service -p 8000:8000 racj90120hplmnv09/mibici-ml-service:0.1.1

# El pipeline: make mlflow en otra terminal, luego el entrenamiento (~10 min)
make train

# La comparación con la referencia; guarda el resultado para el reporte
.venv/bin/python -m team2_mlops2026.modeling.reproducibility compare \
    --train-runs --url http://127.0.0.1:8000 \
    --save docs/reproducibility/maquina-<nombre>-<objetivo>.json
```

En Windows, `.venv\Scripts\python` en lugar de `.venv/bin/python` y `PYTHON_INTERPRETER` en las
llamadas a `make`. Si la tabla tiene otro digest (por ejemplo, porque incluye octubre de 2026),
hay que usar la del snapshot: los artefactos `snapshot/` del run del snapshot, con acceso al
servidor MLflow de referencia, o la carpeta compartida por el equipo, y pasar `--src` con su ruta.

| Dato | Resultado |
|---|---|
| Integrante, máquina, CPU, RAM | |
| Sistema operativo | |
| Commit (`git rev-parse HEAD`) | |
| `make test` | |
| Digests de la tabla (los revisa `compare`) | |
| Reentrenamiento `departures` (máx. diferencia relativa de métricas, tolerancia `1e-4`) | |
| Reentrenamiento `arrivals` | |
| `make train` contra las 14 corridas de referencia | |
| Servicio desde la imagen (`--url`) | |
