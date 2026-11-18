# Entrega final: instrucciones y rúbrica

Transcripción de la actividad «Entrega Final» del curso TC5044.10 en Canvas (captura del 18 de
noviembre de 2026). Fecha de entrega: **domingo 22 de noviembre de 2026, 23:59**.

## Objetivo

Demostrar la capacidad de aplicar todo lo aprendido en las fases anteriores e integrar el trabajo
de extremo a extremo, desde la preparación de los datos hasta el servicio del modelo (un artefacto
que empaqueta el modelo seleccionado, accesible mediante FastAPI y ejecutado en un contenedor
Docker) y la evaluación de *data drift*. Incluye:

- Realizar tareas de manipulación y preparación de datos.
- Explorar y preprocesar datos.
- Aplicar técnicas de versionado de datos para asegurar reproducibilidad y trazabilidad.
- Construir, ajustar y evaluar modelos de Machine Learning.
- Estructurar proyectos de Machine Learning de manera organizada (Cookiecutter).
- Registrar métricas y aplicar control de versiones a los experimentos con herramientas de
  *logging* y *tracking* (MLflow/DVC).
- Visualizar y comparar resultados (métricas) y gestionar el registro de los modelos (Model
  Registry de MLflow/DVC).
- Aplicar mejores prácticas de codificación en cada etapa del pipeline (refactorización).
- Implementar pruebas unitarias y de integración.
- Realizar el serving y la portabilidad del modelo con FastAPI e integrar el modelo desplegable en
  un contenedor (Docker).
- Evaluar el *data drifting* con detección de pérdida de desempeño usando datos sintéticos.

> El entregable será un **reporte final** que incluya cada una de las fases del pipeline
> completo, que indique cómo resolvieron el problema planteado inicialmente (ML Canvas),
> destacando las actividades de **todos los roles involucrados** en el flujo de trabajo usando
> MLOps (integrando el trabajo realizado en cada una de las fases anteriores).

## Implementaciones de la Fase 3 (issues #38–#43)

1. **Pruebas unitarias y de integración.** Unitarias para módulos clave (preprocesamiento,
   cálculo de métricas, inferencia); de integración que validen el flujo extremo a extremo (carga
   de datos → preprocesamiento → predicción → métricas); `pytest`; documentar cómo ejecutarlas
   con un comando único (`pytest -q`).
2. **Serving con FastAPI.** Al menos un endpoint `POST /predict`; validación de entrada con
   Pydantic y manejo de errores; esquema de entrada y salida documentado (OpenAPI/Swagger); la
   ruta y versión del artefacto del modelo en el README (p. ej. `models:/model/version`).
3. **Reproducibilidad en otro entorno.** Dependencias fijadas (`requirements.txt`) y semillas
   aleatorias; ejecutar el pipeline y el servicio en un entorno limpio (otra máquina, VM o
   contenedor) y comparar métricas y salidas con las de referencia; versionar artefactos clave
   (modelo, *feature store* o *dataset snapshot*) en DVC/MLflow.
4. **Docker.** Un Dockerfile que instale dependencias, contenga el modelo y lo sirva con FastAPI;
   los comandos para construir y correr (`docker build -t ml-service:latest .`,
   `docker run -p 8000:8000 ml-service:latest`); publicar la imagen en un registro (Docker Hub) y
   documentar los tags versionados.
5. **Data drift.** Un conjunto de monitoreo con una distribución alterada (desplazamiento de
   medias, *features* faltantes, cambio estacional) respecto al de validación; evaluar las
   métricas de desempeño en ese conjunto y compararlas con la línea base; registrar resultados o
   gráficas, implementar umbrales y criterios de alerta y una acción propuesta (revisión del
   *feature pipeline* o *retrain*).

## Outline sugerido del reporte

**Introducción**
- Descripción de la problemática a resolver (para el dataset asignado), aplicando ML Canvas.
- Contextualización y justificación del uso de MLOps.
- Diagrama de la solución (incluir componentes y herramientas de software).

**Descripción de actividades realizadas durante cada una de las fases**

**Métodos usados y resultados por etapa**
- Descripción detallada de los métodos aplicados en cada fase y de los distintos roles
  involucrados por fase.
- Resultados obtenidos y análisis reflexivo de los resultados, y justificación técnica de los
  métodos utilizados en el pipeline.

**Conclusiones generales**
- Lecciones aprendidas durante el proyecto.
- Puntos de mejora identificados.
- Trabajo futuro y recomendaciones.

> **Importancia:** el reporte debe estar bien estructurado, proporcionando un análisis detallado
> del proyecto desde la perspectiva de MLOps. Debe incluir gráficos, tablas y otras herramientas
> visuales para ilustrar los puntos clave y mejorar la comprensión.

**Consideraciones adicionales: integración de las fases anteriores**
- Asegurarse de que las actividades y resultados de las fases anteriores estén integrados en el
  reporte final.
- Referenciar los repositorios de GitHub **mostrando la actividad de todos los integrantes del
  equipo**.

## Especificaciones de entrega

- Modalidad: en equipo (una entrega por grupo en Canvas, botón «Entregar tarea»).
- Formato: PDF.
- Nombre: `Entrega_Final_Equipo02.pdf`.
- **Contenido obligatorio del PDF:** el enlace al repositorio de GitHub del equipo y a la imagen
  publicada en Docker Hub.
- El video no aparece en las instrucciones de la entrega final (en las fases 1 y 2 sí); es
  opcional.

## Rúbrica (100 puntos)

Cada criterio califica el reporte. Se muestra el descriptor del nivel «Excelente» y dónde lo
atiende la plantilla.

| Criterio | Pts | Excelente | Sección |
|---|---|---|---|
| Acercamiento inicial al problema de la actividad | 15 | Recurren a una gran cantidad de sus conocimientos previos para llegar a una solución. Demuestran mucho interés por el caso. | Resumen, 1.1, 1.2 y el hilo entre fases (cada fase parte de la anterior) |
| Análisis del problema | 15 | Identifican claramente el problema a resolver, correspondiente a la actividad. Plantean las características del problema que guían el análisis hacia la solución. | 1.1 (ML Canvas y los datos que muestran el problema) |
| Actividades y tareas a realizar por rol | 20 | Especifican claramente las actividades de cada rol y describen cómo se alinean dentro de la metodología de MLOps. | 2 y 4 (roles por fase, entregas entre roles, actividad en GitHub) |
| Métodos y técnicas por utilizar | 20 | Argumentan de forma clara y detallada la selección de métodos y técnicas adecuados para resolver el problema. | 3 (cada método con su justificación frente a las alternativas) |
| Resultados | 15 | Obtienen resultados de validez demostrable y proporcionan fundamento metodológico para la obtención de los mismos. Presentan los resultados de forma clara y concisa. | 3 (tablas y figuras con su fuente) y el resumen |
| Conclusiones y reflexión final | 15 | Son capaces de reconocer las áreas en las que pueden mejorar en el análisis del caso. Identifican de forma adecuada las estrategias que emplearon en el análisis y solución del caso. | 5 |

Niveles inferiores, para saber qué evitar: «Regular» (ambiguo, argumentos poco claros, sin
fundamento metodológico de los resultados), «Básico» (parcial, superficial, omisiones),
«En desarrollo» (incompleto o inconsistente, errores conceptuales), «Limitado» (incorrecto o sin
resultados).
