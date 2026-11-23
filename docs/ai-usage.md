# Declaración de uso de IA

## Fase 3 y declaraciones pendientes de las fases 1 y 2

Declaraciones entregadas por cada integrante el 22 de noviembre de 2026 para el reporte final.

### JulianMaloRomeroTEC

**Referencias:**
- OpenAI (2026). *Codex* (GPT-6 Sol, razonamiento medio, en la Fase 1; GPT-6.2 Sol,
  razonamiento alto, en la Fase 3) [modelo de lenguaje grande]. https://openai.com/codex/
- OpenAI (2026). *ChatGPT* [modelo de lenguaje grande]. https://chatgpt.com/

**Propósito y alcance:**
- **Fase 1** ([#21](https://github.com/CarlosHeinze/MLOps_Team2/pull/21)), Codex y ChatGPT:
  revisar el repositorio, identificar problemas de reproducibilidad, corregir el `Makefile`,
  las dependencias y la configuración de MLflow, y preparar documentación. Los cambios se
  verificaron con pruebas locales antes de integrarse.
- **Fase 2** ([#29](https://github.com/CarlosHeinze/MLOps_Team2/issues/29)), ChatGPT: revisar
  los requisitos de la rúbrica, organizar la presentación, preparar su intervención en el video
  y revisar la documentación del equipo.
- **Fase 3** ([#55](https://github.com/CarlosHeinze/MLOps_Team2/pull/55),
  [#56](https://github.com/CarlosHeinze/MLOps_Team2/pull/56)), Codex y ChatGPT: desarrollar y
  revisar la simulación de drift (#42): monitoreo, alertas, pruebas y registro en MLflow; e
  integrar los comandos del `Makefile`, la documentación, las dependencias y las pruebas de
  trazabilidad (#43). Se validó con datos sintéticos e históricos y se documentaron las
  limitaciones de reproducibilidad encontradas.

En todas las fases la IA fue apoyo para programación, depuración, análisis y redacción.
Julián revisó los resultados, mantuvo las pruebas y evidencias correspondientes y no presentó
resultados sintéticos como datos reales.

### chaverocarlos

**Referencia:** Google (2026). *Gemini* (Gemini 4 Hydrogen, nivel de razonamiento bajo) [modelo
de lenguaje grande]. https://gemini.google.com/

**Propósito y alcance:** usado en las tres fases, en sus actividades: la validación de cifras
contra MLflow y las pruebas (Fase 1), el reporte ejecutivo y el video
([#29](https://github.com/CarlosHeinze/MLOps_Team2/issues/29), Fase 2) y la reproducción en una
segunda máquina con WSL2 (Fase 3, `docs/reproducibility/maquina-carlos-wsl.json`).

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
