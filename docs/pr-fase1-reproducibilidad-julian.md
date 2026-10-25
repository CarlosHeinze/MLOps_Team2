# PR sugerido

**Título:** Fase 1: reproducibilidad local y material verificable para presentación

**Descripción para copiar al Pull Request:**

## Resumen

- Alinea el Makefile con Python 3.12 y con la canalización real de extracción,
  normalización y limpieza. `make data` ya no instala paquetes ni ejecuta `dataset.py`.
- Configura MLflow local en el puerto 5001, documenta `.env.example` y actualiza el
  mensaje de error y su prueba.
- Fija las versiones de las dependencias directas instaladas para esta suite y corrige
  los usos de `dt.dayofweek` de los notebooks 1.02 y 1.03.
- Añade material para la presentación, roles funcionales observables, declaración de
  uso de IA de Julian y registro de verificaciones. Los resultados anteriores se citan
  como reportados por sus PR y se distinguen de las verificaciones de esta rama.

## Verificación

- `make test`: **163 aprobadas**, una advertencia procedente de SQLAlchemy a través de
  MLflow; sin fallos.
- `make lint`: formato y análisis estático correctos.
- `.venv/bin/python -m pip check`: sin dependencias incompatibles.
- `make -n data`: muestra los tres módulos reales, sin ejecutarlos.
- Los cinco módulos de la canalización aceptan `--help` con código de salida 0.
- Los seis notebooks son JSON válido y los 16 módulos del paquete tienen sintaxis válida.

## Alcance y límites

No se descargó el historial MiBici, no se ejecutaron notebooks con datos reales y no se
generaron métricas nuevas de modelos o runs de MLflow de producción. El entorno local
fija dependencias directas; las transitivas aún no tienen un lock completo. La
presentación PDF final y las declaraciones de IA de los demás integrantes requieren
revisión del equipo. No se modifica la metodología de limpieza ni de variables.

Refs #12, #15, #16.
