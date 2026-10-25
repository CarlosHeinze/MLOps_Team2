# Verificación local de la contribución de Julian

Entorno previsto: MacBook Air M4, 16 GB de RAM, Python 3.12.13. No se descargó el
historial MiBici ni se ejecutaron notebooks o procesos sobre los 39 millones de viajes.

| Comando | Resultado | Alcance |
|---|---|---|
| `git rev-parse HEAD` antes de crear la rama | `2b74f5bea2fb77f61e50e6dd348de8de5346a739` | Confirma el punto de partida de `master` |
| `python3.12 --version` | `Python 3.12.13` | Intérprete disponible |
| `make create_environment` | Correcto | Creó `.venv` aislada |
| `make requirements` | Correcto | Instaló las dependencias directas fijadas y el paquete editable; `.venv` ocupó aproximadamente 902 MB |
| `make -n data` | Correcto | Mostró extracción → normalización → limpieza; no ejecutó ni descargó datos |
| `make help` | Correcto | Incluyó `create_environment`, `requirements`, `mlflow`, `data`, `test` y `lint` |
| `make lint` | Correcto | `ruff format --check`: 16 archivos formateados; `ruff check`: sin hallazgos |
| `make test` (primera corrida) | 162 aprobadas, 1 fallida | La prueba de mensaje de error MLflow esperaba el comando antiguo de puerto 5000 |
| Corrección de `tests/test_tracking.py` | Correcta | Espera la nueva instrucción `make mlflow` |
| `make test` (segunda corrida) | **163 aprobadas, 1 advertencia**, 6.62 s | Pruebas sintéticas; sin datos históricos ni servidor MLflow externo |
| `.venv/bin/python -m pip check` | Correcto | Sin requisitos incompatibles en la `.venv` |
| `python3.12` con `json.loads` | Correcto | Los seis notebooks son JSON válido tras las ediciones |
| `python3.12` con `ast.parse` | Correcto | Los 16 módulos Python del paquete tienen sintaxis válida |
| `git diff --check` | Correcto | Sin errores de espacios en el diff |
| Cinco CLI con `--help` | Correcto | Extracción, normalización, limpieza, variables y PCA terminan con código 0; no procesan datos |

**Advertencia de la suite:** `SADeprecationWarning` emitida dentro de
`mlflow/store/tracking/sqlalchemy_store.py` por una estrategia de SQLAlchemy obsoleta.
No hubo fallo de prueba asociado. Los recuentos históricos citados en el material de
presentación proceden de PR de otros integrantes y no se verificaron con esta rama.

**Pendiente:** reproducción en otra máquina y ejecución con el historial completo.
`requirements.txt` fija dependencias directas; las transitivas aún no están bloqueadas.
