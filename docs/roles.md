# Roles, actividades e interacción

## Fase 3: pruebas, serving, Docker, reproducibilidad y drift

Carlos abrió los issues #38 a #49 con su alcance y sus criterios de aceptación; #38 a #44 eran
obligatorios y #45 a #49 opcionales (quedaron abiertos).

| Integrante | Rol | Issues | Pull requests | Entrega a la siguiente etapa |
|---|---|---|---|---|
| CarlosHeinze | ML Engineer y QA | [#38](https://github.com/CarlosHeinze/MLOps_Team2/issues/38), [#44](https://github.com/CarlosHeinze/MLOps_Team2/issues/44) | [#50](https://github.com/CarlosHeinze/MLOps_Team2/pull/50), [#57](https://github.com/CarlosHeinze/MLOps_Team2/pull/57); revisó e integró #55 y #56 | Prueba de integración y `pytest -q`; verificación nativa en Windows; reporte final |
| javieramirez90 | MLOps Engineer | [#39](https://github.com/CarlosHeinze/MLOps_Team2/issues/39), [#40](https://github.com/CarlosHeinze/MLOps_Team2/issues/40), [#41](https://github.com/CarlosHeinze/MLOps_Team2/issues/41) | [#51](https://github.com/CarlosHeinze/MLOps_Team2/pull/51), [#52](https://github.com/CarlosHeinze/MLOps_Team2/pull/52), [#53](https://github.com/CarlosHeinze/MLOps_Team2/pull/53), [#54](https://github.com/CarlosHeinze/MLOps_Team2/pull/54) | Campeones exportados con `model_card.json`, servicio FastAPI, imagen `0.1.1` en Docker Hub, run snapshot y `tests/data/reference.json` |
| JulianMaloRomeroTEC | ML Engineer de monitoreo y DevOps | [#42](https://github.com/CarlosHeinze/MLOps_Team2/issues/42), [#43](https://github.com/CarlosHeinze/MLOps_Team2/issues/43) | [#55](https://github.com/CarlosHeinze/MLOps_Team2/pull/55), [#56](https://github.com/CarlosHeinze/MLOps_Team2/pull/56) | Monitoreo de drift (`make monitor`, runs `drift-*`) y la Fase 3 integrada en el `Makefile` y el README |
| chaverocarlos | QA de reproducibilidad | [#41](https://github.com/CarlosHeinze/MLOps_Team2/issues/41) | commit `d8d2d2d` | Reproducción en una segunda máquina (WSL2): `docs/reproducibility/maquina-carlos-wsl.json` |

### Entregas entre roles

1. **ML Engineer → MLOps Engineer:** el campeón con alias `champion` y la prueba de integración,
   que debe pasar con cada cambio.
2. **MLOps Engineer → monitoreo y QA:** `models/` con `model_card.json` (versión, run, digests,
   commit), la imagen con tag y el set de referencia del snapshot.
3. **Monitoreo → equipo:** cada run `drift-*` registra los digests de sus entradas, las alertas
   y la acción propuesta.
4. **QA → documentación:** un JSON comparable por entorno en `docs/reproducibility/`.

## Fase 2: construcción, ajuste, evaluación y registro de modelos

Carlos abrió los issues #22 a #29 con su alcance y sus criterios de aceptación antes de empezar.
Cada issue se trabajó en una rama propia y se integró a `master` con un pull request.

| Integrante | Rol | Issues | Pull requests | Entrega a la siguiente etapa |
|---|---|---|---|---|
| CarlosHeinze | ML Engineer | [#22](https://github.com/CarlosHeinze/MLOps_Team2/issues/22), [#23](https://github.com/CarlosHeinze/MLOps_Team2/issues/23), [#24](https://github.com/CarlosHeinze/MLOps_Team2/issues/24), [#25](https://github.com/CarlosHeinze/MLOps_Team2/issues/25) | [#30](https://github.com/CarlosHeinze/MLOps_Team2/pull/30), [#31](https://github.com/CarlosHeinze/MLOps_Team2/pull/31), [#32](https://github.com/CarlosHeinze/MLOps_Team2/pull/32), [#33](https://github.com/CarlosHeinze/MLOps_Team2/pull/33) | Métricas y folds, baselines, candidatos y modelos ajustados en los runs `tune-<modelo>-<objetivo>` |
| javieramirez90 | MLOps Engineer | [#26](https://github.com/CarlosHeinze/MLOps_Team2/issues/26), [#27](https://github.com/CarlosHeinze/MLOps_Team2/issues/27), [#28](https://github.com/CarlosHeinze/MLOps_Team2/issues/28) | [#34](https://github.com/CarlosHeinze/MLOps_Team2/pull/34), [#35](https://github.com/CarlosHeinze/MLOps_Team2/pull/35), [#36](https://github.com/CarlosHeinze/MLOps_Team2/pull/36) | Backtest y ablación de PCA, campeón en el Model Registry, pronóstico batch y `make pipeline` |
| JulianMaloRomeroTEC | DevOps y documentación | [#29](https://github.com/CarlosHeinze/MLOps_Team2/issues/29) | — | Reporte ejecutivo y video de la fase |
| chaverocarlos | QA y documentación | [#29](https://github.com/CarlosHeinze/MLOps_Team2/issues/29) | — | Reporte ejecutivo y video de la fase; revisión de las cifras contra MLflow y los notebooks |

Javier también reprodujo los notebooks 2.01 a 2.03 de Carlos en un segundo equipo antes de
continuar. Los baselines y `lgbm_tweedie` dieron números idénticos; `hgb_poisson` difiere en la
tercera decimal por el no determinismo del hardware.

### Entregas entre roles

Cada entrega es un artefacto que se puede comprobar en MLflow, no un acuerdo verbal.

1. **Data Engineer → ML Engineer:** las tablas de la Fase 1 (`mibici_station_hour_features` y
   `mibici_station_profiles`), cada una con su digest. Todo run de modelo registra esos digests
   como datasets de entrada.
2. **ML Engineer → MLOps Engineer:** los runs `tune-<modelo>-<objetivo>`, con los mejores
   parámetros y el modelo reentrenado.
   - El backtest toma los parámetros del último de esos runs y guarda de cuál los tomó
     (`params_from_<objetivo>`).
   - El registro comprueba que la ablación evaluó ese mismo modelo antes de usar sus métricas.
3. **MLOps Engineer → DevOps:** los pasos del modelado como comandos (`make train`, `tune`,
   `evaluate`, `register`, `predict` y `pipeline`), con sus tiempos en el README.
4. **DevOps → QA y documentación:** un entorno reproducible y los runs en el servidor de MLflow.
   QA compara cada cifra del reporte con su run o su celda de notebook.

### Cómo se revisa

issue con alcance y criterios de aceptación → rama `<número>-<título>` → pull request con
descripción y `Closes #<número>` → revisión (de un compañero o de Copilot) → merge a `master`.
La reproducción en un segundo equipo funciona como revisión cruzada de los resultados.

## Fase 1: actividades observables

Esta tabla describe trabajo verificable en PR, no asignaciones formales de personas.

| Actividad funcional | Persona según PR | Evidencia | Entrega a la siguiente etapa |
|---|---|---|---|
| Ingeniería de datos: adquisición y normalización | CarlosHeinze | [#2](https://github.com/CarlosHeinze/MLOps_Team2/pull/2), [#4](https://github.com/CarlosHeinze/MLOps_Team2/pull/4) | CSV y Parquet intermedio |
| Ciencia de datos: limpieza y análisis exploratorio | CarlosHeinze | [#6](https://github.com/CarlosHeinze/MLOps_Team2/pull/6) | Datos limpios, contrato y hallazgos |
| Análisis de producto y ML: Canvas, variables y PCA | javieramirez90 | [#17](https://github.com/CarlosHeinze/MLOps_Team2/pull/17), [#18](https://github.com/CarlosHeinze/MLOps_Team2/pull/18), [#19](https://github.com/CarlosHeinze/MLOps_Team2/pull/19) | Objetivos y entradas para futuros modelos |
| MLOps: trazabilidad entre etapas | javieramirez90 | [#20](https://github.com/CarlosHeinze/MLOps_Team2/pull/20) | Digests y runs de MLflow |
| Ingeniería de reproducibilidad y documentación | JulianMaloRomeroTEC | [#21](https://github.com/CarlosHeinze/MLOps_Team2/pull/21) | Comandos y entorno local coherentes; material para presentación |

**Interacción:** adquisición → normalización → limpieza → análisis y variables → versionado de
entradas y salidas en MLflow → entrenamiento (Fase 2).
