# Declaración de uso de IA

## Fase 2

Todo el código se revisó y ejecutó, y todas las cifras del reporte provienen de ejecutar ese
código sobre los datos reales. Ningún resultado fue generado por la IA sin esa ejecución.

### javieramirez90

**Referencia:** Anthropic (2026). *Claude Code* (basado en Claude Opus 5.5) [modelo de lenguaje
grande]. https://claude.com/claude-code

**Propósito y alcance:**
- generación y depuración de código y pruebas: backtest, ablación de PCA, Model Registry y
  pronóstico batch;
- ejecución y verificación de los notebooks 2.01 a 2.05 y análisis de sus resultados;
- capturas de MLflow;
- redacción de la documentación, de las descripciones de PR y del reporte ejecutivo.

Javier revisó y asume la responsabilidad de los commits y de las afirmaciones del reporte.

**Archivos y PR:** [#34](https://github.com/CarlosHeinze/MLOps_Team2/pull/34),
[#35](https://github.com/CarlosHeinze/MLOps_Team2/pull/35),
[#36](https://github.com/CarlosHeinze/MLOps_Team2/pull/36) y los documentos de
[#29](https://github.com/CarlosHeinze/MLOps_Team2/issues/29).

### Revisión automática de PR

GitHub Copilot revisó los pull requests
[#30](https://github.com/CarlosHeinze/MLOps_Team2/pull/30),
[#31](https://github.com/CarlosHeinze/MLOps_Team2/pull/31) y
[#33](https://github.com/CarlosHeinze/MLOps_Team2/pull/33) (`copilot-pull-request-reviewer`).

### Declaraciones de los demás integrantes

Cada integrante declara personalmente las herramientas que usó en la Fase 2, como en la Fase 1.

## Fase 1

Cada integrante debe declarar personalmente las herramientas y modelos de IA que utilizó,
su propósito y los archivos o PR afectados. Esta declaración cubre **solo** la
contribución de Julian; no atribuye uso de IA a otros compañeros.

### JulianMaloRomeroTEC

**Referencia:** OpenAI (2026). *Codex* (agente de programación basado en la familia GPT-6)
[modelo de lenguaje]. https://openai.com/codex/ . El identificador exacto del modelo de
esta sesión no se presenta en los materiales del repositorio y debe confirmarse en la
interfaz antes de entregar la cita definitiva.

**Propósito y alcance:** apoyo en la auditoría del repositorio y sus PR, identificación
de tareas pendientes, redacción y revisión de documentación, propuesta y edición de
cambios de reproducibilidad, y análisis de resultados de verificaciones locales.
Julian revisa y asume la responsabilidad de los commits y de cualquier afirmación
incluida en la presentación. No se empleó IA para fabricar resultados de datos reales,
métricas de modelos ni ejecuciones de MLflow.

**Archivos de esta contribución:** `Makefile`, `README.md`, `requirements.txt`,
`.env.example`, `team2_mlops2026/config.py`, `team2_mlops2026/tracking.py`,
`team2_mlops2026/data/interim_data_cleaning.py`, notebooks 1.02 y 1.03, y los documentos
`docs/fase1-presentacion-julian.md`, `docs/roles.md`, `docs/ai-usage.md` y
`docs/verificacion-fase1-julian.md`, además del borrador de descripción del PR en
`docs/pr-fase1-reproducibilidad-julian.md`. También se ajustó `tests/test_tracking.py`.

### Declaraciones de los demás integrantes (Fase 1)

Pendientes de aportación y aprobación individual. El [issue #16](https://github.com/CarlosHeinze/MLOps_Team2/issues/16)
requiere que cada integrante confirme herramienta, modelo, propósito y archivos o PR,
además de revisar el manifiesto completo. No se infiere su uso de IA a partir de los PR.


### Carlos

* **Fase 1 (revisión de configuración y documentación):**
  * **Herramienta y modelo:** No usé IA.
  * **Para qué se usó:** No aplica; revisión manual de la configuración de MLflow y de la documentación del proyecto.
  * **Archivos, PR o entregables:** Configuración de MLflow y documentación de la Fase 1.

* **Fase 2 (reporte y video, issue #29):**
  * **Herramienta y modelo:** Gemeni 4 Hydrogen (con nivel de pensamiento bajo) (Google).
  * **Para qué se usó:** Revisión de requisitos de la rúbrica, estructuración de la presentación, preparación de intervención en el video y revisión de documentación del equipo, con enfoque en comunicación de resultados, reproducibilidad y organización del proyecto.
  * **Archivos, PR o entregables:** Reporte, presentación y video del issue #29.

* **Fase 3 (verificación en WSL2 — Punto 2: *Otra máquina (integrante distinto del autor, issse #41)*):**
  * **Herramienta y modelo:** Gemeni 4 Hydrogen (con nivel de pensamiento bajo) (Google).
  * **Para qué se usó:** Configuración y depuración del entorno en Ubuntu/WSL2 (permisos de Docker en `/var/run/docker.sock` y Python 3.12 para `.venv`), ejecución guiada del pipeline (`make test`, generación local de datos/características/perfiles, `make docker-models`, contenedor `racj90120hplmnv09/mibici-ml-service:0.1.1`, `make train` y diagnóstico de digests con `team2_mlops2026.modeling.reproducibility compare --no-retrain --train-runs --url http://127.0.0.1:8000`), autenticación y resolución de conflicto en `git pull --rebase`, y redacción de la evidencia para el cierre del issue.
  * **Archivos, PR o entregables:** `docs/verificacion-fase3.md` (sección *2. Otra máquina (integrante distinto del autor)*), `docs/reproducibility/maquina-carlos-wsl.json`, commit `d8d2d2d937ff00c542be65f1196ff2dd61d6da7b` en `master` y evidencia para el cierre del issue de verificación de la Fase 3.
