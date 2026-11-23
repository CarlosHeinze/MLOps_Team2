# Declaración de uso de IA

## Fase 3

### JulianMaloRomeroTEC

**Referencia:** OpenAI (2026). *ChatGPT* y *Codex* (GPT-6.2 Sol High).

**Propósito y alcance:** usé IA para asistir en el diseño y la depuración del monitoreo
de deriva de datos, las pruebas de registro en MLflow y el análisis del replay histórico
de septiembre de 2026. También la usé para proponer y revisar cambios al `Makefile`,
a las dependencias y a las pruebas de linaje de los modelos exportados, y para redactar
la documentación.

**Verificación humana:** revisé los cambios y contrasté las métricas y alertas reportadas
con las ejecuciones de monitoreo registradas en MLflow. Verifiqué en macOS la instalación
de dependencias, el lint y las 307 pruebas, incluida la comprobación del linaje desde el alias
`champion` hasta el run y los digests en `model_card.json`. La comparación con el servicio local
validó las predicciones de referencia de los modelos exportados con `--no-retrain`.

**Límites de reproducibilidad:** el replay real de septiembre de 2026 usó los Parquet
compartidos por el equipo. Los digests de las tablas de características y perfiles difieren
de los del snapshot original de los modelos `champion`. Los puntajes PCA también difirieron
en los tres casos de referencia reconstruidos a partir de esos archivos. Las métricas del
replay corresponden a la exportación compartida y la prueba de predicciones de referencia
se verificó con entradas fijas. Reproducir el entrenamiento original requiere las tablas
del snapshot con sus digests correspondientes.

**PR e issues:** [PR #55](https://github.com/CarlosHeinze/MLOps_Team2/pull/55)
([issue #42](https://github.com/CarlosHeinze/MLOps_Team2/issues/42)) y
[PR #56](https://github.com/CarlosHeinze/MLOps_Team2/pull/56)
([issue #43](https://github.com/CarlosHeinze/MLOps_Team2/issues/43)).

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

### JulianMaloRomeroTEC

**Referencia:** OpenAI (2026). *ChatGPT*.

**Propósito y alcance:** usé ChatGPT para revisar la rúbrica y la documentación de entrega
y para preparar y revisar el guion y el material del video. Mi aporte se centró en la
documentación y la comunicación de los resultados generados por el equipo.

**Verificación humana:** revisé el guion y el material del video frente a la rúbrica
y la documentación del equipo.

**Issue:** [#29](https://github.com/CarlosHeinze/MLOps_Team2/issues/29).

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

**Referencia:** OpenAI (2026). *ChatGPT* y *Codex* (GPT-6 Sol Medium).

**Propósito y alcance:** usé IA para apoyar el diagnóstico del repositorio y de los PR,
proponer y revisar cambios de reproducibilidad en el `Makefile`, las dependencias y
la configuración local de MLflow, y redactar y revisar la documentación de la entrega.

**Verificación humana:** revisé los cambios, los resultados de `make test`, `make lint`
y `pip check`, y la validez de los notebooks. La verificación fue local. La ejecución
completa con el historial de MiBici quedó fuera de la verificación de este PR. Las cifras
históricas de la presentación se atribuyeron a los PR del equipo.

**PR:** [#21](https://github.com/CarlosHeinze/MLOps_Team2/pull/21).

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
