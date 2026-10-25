# Fase 1 · MiBici — material para la presentación ejecutiva

**Uso previsto:** texto y estructura para incorporar al PDF `Fase 1_Equipo02` del equipo.
No es una entrega académica ni sustituye la revisión de los demás integrantes. Estado de
referencia: `master` en `2b74f5b` (25 de octubre de 2026). **Cifras marcadas como
«reportadas» proceden de notebooks, ML Canvas o PR del equipo y no se reprodujeron al
preparar este documento.**

## 1. Enfoque inicial, problema y valor

MiBici registra viajes que salen de una estación y terminan en otra. Ese flujo puede dejar
estaciones sin bicicletas o sin espacios libres. La propuesta de valor es anticipar las
salidas y llegadas por estación y hora para apoyar la redistribución antes de los periodos
de mayor demanda. El usuario previsto es el equipo de operación de MiBici; el proyecto
académico trabaja con datos públicos históricos y no dispone de inventario de bicicletas
en tiempo real ni de registros de rutas de redistribución.

El [Machine Learning Canvas](../references/ml-canvas.md) define una predicción de salidas
y llegadas para las próximas 24 horas, con una actualización nocturna y posible ajuste al
mediodía. Plantea comparar una referencia estacional y un promedio histórico contra modelos
como Gradient Boosting, usando MAE, WAPE y desviación de Poisson. También propone medir la
identificación de días con gran desequilibrio de flujo. **Esos modelos, métricas de
pronóstico y procesos de predicción son objetivos documentados; no están implementados en
la Fase 1 actual.**

## 2. Arquitectura y roles funcionales

La estructura separa módulos reutilizables, notebooks explicativos y pruebas. El flujo
implementado es:

`CSV públicos → data/raw → Parquet intermedio → Parquet limpio → tabla estación-hora → perfiles PCA`

Cada etapa de datos tiene una interfaz de línea de comandos y registra sus entradas y
salidas en MLflow cuando se ejecuta con seguimiento. Los notebooks 0.01 y 1.01–1.05
documentan las decisiones y muestran resultados históricos; el código de entrenamiento y
predicción aún es una plantilla. Las actividades observables por persona y la interacción
entre roles están en [roles.md](roles.md); deben ser validadas por el equipo antes de
presentarlas como asignaciones oficiales. La refactorización trasladó operaciones
reutilizables de los notebooks a módulos con pruebas; el uso adicional de POO cuando
reduzca duplicación sigue abierto en el [issue #14](https://github.com/CarlosHeinze/MLOps_Team2/issues/14).

## 3. Adquisición y normalización

La extracción descubre enlaces CSV en la página de datos abiertos de MiBici, descarga los
archivos que faltan y mantiene un `manifest.json` con URL, tamaño, SHA-256 y fecha. La
normalización unifica nombres de columnas, codificación, fechas y tipos, y genera un
Parquet por mes. Conserva los CSV originales. Este diseño permite rastrear el origen de
cada conjunto de datos y repetir una etapa sin rehacer necesariamente las anteriores.

**Resultados reportados:** [PR #20](https://github.com/CarlosHeinze/MLOps_Team2/pull/20)
describe 143 archivos normalizados, 100 UTF-8, 43 Latin-1 y tres meses con fechas sin
segundos. El [Canvas](../references/ml-canvas.md) habla de 142 CSV mensuales de viajes;
la diferencia corresponde al archivo de estaciones. El número de filas crudas difiere
ligeramente entre el Canvas y el PR #20; conviene cotejar la definición exacta antes de
poner ese total en una gráfica final.

## 4. Limpieza y análisis exploratorio

La limpieza elimina duplicados exactos y viajes incompatibles con el contrato de datos
(duraciones extremas y velocidades inverosímiles). Los valores demográficos inválidos se
marcan como faltantes; los faltantes que pueden recuperarse se completan con información
del mismo usuario. Cada mes se valida antes de escribirse y se produce un reporte de
reglas aplicadas. El análisis exploratorio estudia calidad, horarios, estaciones y flujos.

**Resultados reportados:** la revisión del [PR #6](https://github.com/CarlosHeinze/MLOps_Team2/pull/6)
afirma que se reprodujeron **37,786,886 viajes procesados** en otra Mac. El Canvas
describe picos de viajes laborales a las 08:00 y 18:00, baja actividad entre 01:00 y
05:00 y concentración del 39.5 % de las salidas en el 10 % de estaciones con más uso.
Estos resultados proceden de los notebooks 1.02 y 1.03; aquí no se ejecutaron con los
datos históricos.

## 5. Variables, tabla estación-hora y PCA

El [PR #18](https://github.com/CarlosHeinze/MLOps_Team2/pull/18) incorporó una fila por
estación y hora activa, incluso horas con cero viajes; dos objetivos (salidas y llegadas),
rezagos de demanda conocidos con antelación, calendario cíclico y atributos de estación.
El preprocesador incluye imputación, escalado, codificación de día de la semana y
codificación de estación. El notebook 1.04 separa entrenamiento y prueba por tiempo y
ajusta el preprocesador con entrenamiento.

El [PR #19](https://github.com/CarlosHeinze/MLOps_Team2/pull/19) construyó perfiles
semanales de uso y ajustó PCA solo en el periodo de entrenamiento. **Resultados
reportados:** #18 describe 12.2 millones de filas estación-hora y 486 estaciones; #19
describe 26 componentes que explican el 90.1 % de la varianza para 371 estaciones.
La tabla y PCA preparan posibles entradas de un modelo, pero aún no hay entrenamiento
ni backtest implementados.

## 6. Versionado y trazabilidad en MLflow

Extracción, normalización, limpieza, construcción de variables y PCA crean runs con
parámetros, métricas de cada etapa y datasets de entrada y salida. Los datasets grandes
se identifican mediante un digest calculado sobre los bytes de sus archivos; el digest
de salida de una etapa puede cotejarse con el de entrada de la siguiente. El [PR #20](https://github.com/CarlosHeinze/MLOps_Team2/pull/20)
reporta que esa cadena coincidió al reproducir el historial completo. El seguimiento
configurado por defecto es local: el equipo debe conservar o compartir explícitamente
la evidencia que use en la presentación.

La comparación de estrategias A, B y C de datos faltantes como versiones separadas en
MLflow sigue pendiente en el [issue #11](https://github.com/CarlosHeinze/MLOps_Team2/issues/11).

## 7. Pruebas, reproducibilidad y límites

El repositorio contiene pruebas unitarias sobre datos sintéticos para módulos de
extracción, normalización, limpieza, análisis, variables, PCA y continuidad de digests.
Los PR #18, #19 y #20 reportan respectivamente 152, 158 y 163 pruebas aprobadas en sus
ramas. **Son resultados declarados por sus autores, no ejecuciones de este documento.**
La evidencia de pruebas de esta rama debe copiarse desde
[verificacion-fase1-julian.md](verificacion-fase1-julian.md) una vez concluida.

Limitaciones que deben mencionarse: no hay CSV ni Parquet reales versionados en Git;
la reconstrucción completa ocupa aproximadamente 5 GB de disco y el notebook 1.02
puede necesitar unos 20 GB de RAM. Las dependencias no tenían bloqueo completo. MiBici
publica solo la nomenclatura actual de estaciones; faltan datos históricos de capacidad
y existencias de bicicletas. El código actual no genera aún pronósticos operativos.

## 8. Conclusiones y siguientes pasos

La Fase 1 deja una base de datos y análisis modular, con contratos, pruebas y trazabilidad.
Los siguientes pasos son cerrar el entorno reproducible, completar la comparación de
estrategias de limpieza en MLflow, definir cómo incluir estaciones activas sin viajes
recientes en horizontes futuros y desarrollar baselines con backtesting temporal antes
de probar modelos más complejos. La evaluación debe distinguir salidas y llegadas,
horarios pico y estaciones de demanda baja.

## 9. Contribuciones y declaración de IA

- **CarlosHeinze:** PR #2 (extracción), #4 (normalización) y #6 (limpieza y EDA).
- **javieramirez90:** PR #17 (Canvas), #18 (variables), #19 (PCA) y #20 (cadena MLflow).
- **JulianMaloRomeroTEC:** rama `fix/fase1-reproducibilidad-julian`: revisión de
  reproducibilidad, corrección de comandos y configuración local, actualización de
  documentación y preparación de este material. El alcance final debe cotejarse con
  los commits de la rama antes de publicarlo.

La declaración individual de herramientas de IA está en [ai-usage.md](ai-usage.md).
Cada integrante debe agregar sus propias herramientas, modelos y usos; esta rama solo
documenta el trabajo de Julian. Incluir en el PDF el enlace al repositorio y los commits
de cada integrante, y exportar el PDF final para Canvas conforme al issue #15.

## Fuentes internas

[README](../README.md) · [ML Canvas](../references/ml-canvas.md) ·
[PR #6](https://github.com/CarlosHeinze/MLOps_Team2/pull/6) ·
[PR #18](https://github.com/CarlosHeinze/MLOps_Team2/pull/18) ·
[PR #19](https://github.com/CarlosHeinze/MLOps_Team2/pull/19) ·
[PR #20](https://github.com/CarlosHeinze/MLOps_Team2/pull/20) ·
[issues #15](https://github.com/CarlosHeinze/MLOps_Team2/issues/15) y
[#16](https://github.com/CarlosHeinze/MLOps_Team2/issues/16).
