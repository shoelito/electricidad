# Monitor eléctrico: predicción de picos de consumo con IA

Proyecto universitario (Ingeniería en Software, UIA). Un medidor conectado a un microcontrolador envía la
**potencia en kilovatios (kW) cada segundo** a un backend local. Un modelo de Machine Learning predice el consumo de
la próxima hora y avisa cuando viene un **pico**. El dashboard está hecho con **Streamlit**. Todo corre en una
computadora local que se publica a Internet con **ngrok**.

## Arquitectura

```
Medidor → ESP32 (potencia en kW, 1 lectura/s) → WiFi → ngrok → Backend FastAPI :8000 (computadora local)
                                                                 ├─ SQLite (lecturas y predicciones)
                                                                 └─ Modelo de IA (HistGradientBoosting)
                                                                        ↑ peticiones HTTP
Dispositivos de los usuarios → ngrok → Frontend Streamlit :8501 ───────┘
                                        ├─ Dashboard
                                        └─ Explorador de la API
```

Son **dos servidores**: el backend (FastAPI) recibe las lecturas del ESP32 y expone la API; el frontend (Streamlit)
muestra el dashboard y consulta al backend por HTTP. El frontend nunca toca la base de datos directamente.

| Responsable | Parte |
|---|---|
| Equipo de hardware | Medidor, ESP32 y envío de lecturas en kW |
| Equipo de backend | Computadora local, ngrok, backend, base de datos, IA y frontend |

El punto de unión entre ambos equipos es el **contrato de la API**, descrito más abajo.

## Estructura

```
backend/                   # API (FastAPI)
├── app/
│   ├── main.py            # crea la app FastAPI
│   ├── config.py          # configuración por variables de entorno
│   ├── schemas.py         # contrato de la API (modelos Pydantic)
│   ├── database.py        # esquema SQLite
│   ├── routes/            # ingesta, salud y consultas del dashboard
│   ├── repositories/      # acceso a dispositivos, lecturas y predicciones
│   ├── services/          # predicción periódica y entrenamiento en segundo plano
│   └── ml/                # features, entrenamiento y datos sintéticos
├── scripts/               # entrenar el modelo desde la consola
└── tests/
frontend/                  # interfaz (Streamlit)
├── app.py                 # punto de entrada y navegación
├── views/                 # páginas: dashboard y explorador de la API
├── monitor_frontend/      # cliente HTTP del backend, formatos, gráfica y lectura de OpenAPI
├── .streamlit/config.toml # puerto 8501, sin estadísticas de uso
└── tests/
```

## Requisitos

- Python 3.13
- ngrok, para publicar los servidores

## Instalación

.venv\Scripts\Activate.ps1      

Backend, desde la carpeta `backend/`:

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

```powershell
Copy-Item .env.example .env
```

Frontend, desde la carpeta `frontend/` (tiene su propio entorno virtual):

```powershell
python -m venv .venv
```

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

### Configuración (`backend/.env`)

| Variable | Por defecto | Descripción |
|---|---|---|
| `ELECTRICIDAD_DATABASE_PATH` | `data/electricidad.db` | Archivo SQLite |
| `ELECTRICIDAD_MODEL_PATH` | `models/modelo_picos.joblib` | Modelo entrenado |
| `ELECTRICIDAD_LOCAL_TIMEZONE` | `America/Costa_Rica` | Zona horaria para las features de hora y día |
| `ELECTRICIDAD_MAX_POWER_KW` | `25` | Potencia máxima aceptada por lectura (kW) |
| `ELECTRICIDAD_MAX_READING_AGE_DAYS` | `7` | Antigüedad máxima de una lectura reenviada |
| `ELECTRICIDAD_ENABLE_PREDICTION_SCHEDULER` | `true` | Predicción automática cada 7 min |

El archivo `.env`, la base de datos y los modelos están en `.gitignore`; no se suben al repositorio.

El frontend se configura con la variable de entorno `MONITOR_API_URL` (por defecto `http://127.0.0.1:8000`), que
indica dónde está el backend.

## Puesta en marcha

Cada servidor va en su propia terminal.

1. **Backend**, desde `backend/`:

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn app.main:create_app --factory --host 127.0.0.1 --port 8000
   ```

2. **Frontend**, desde `frontend/`:

   ```powershell
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```

3. **Publicar con ngrok.** El ESP32 necesita llegar al backend (puerto 8000) y los usuarios al frontend (puerto 8501).
   El plan gratuito de ngrok da **un solo dominio fijo**, que conviene reservar para el backend, porque la URL va
   escrita en el ESP32:

   ```powershell
   ngrok http --url=<su-dominio>.ngrok-free.app 8000
   ```

   Publicar también el frontend en el mismo dominio requiere enrutar por ruta con la configuración de ngrok; queda
   pendiente para la fase de despliegue. Mientras tanto, el dashboard se usa en `http://localhost:8501`.

No hace falta registrar los dispositivos: un `device_id` nuevo se da de alta solo con su primer lote de lecturas.

Páginas disponibles:

- `http://localhost:8501`: dashboard (consumo, predicción, alertas de pico y entrenamiento).
- `http://localhost:8501/api_explorer`: explorador de la API.
- `http://localhost:8000/openapi.json`: especificación OpenAPI del backend.

### Entrenar el modelo

Desde el dashboard: elegir el medidor y pulsar **Entrenar modelo**. El entrenamiento corre en segundo plano, el
dashboard muestra el avance y, al terminar, el error del modelo y el recall de picos. El modelo nuevo se carga solo,
sin reiniciar el servidor. Se puede entrenar desde 100 intervalos de 7 min (~12 h de lecturas), pero el modelo solo es confiable con
varios días; lo recomendable son 2–4 semanas.

También se puede entrenar desde la consola, por ejemplo con datos sintéticos mientras no haya datos reales:

```powershell
.\.venv\Scripts\python.exe -m scripts.train_model --synthetic-days 28
```

Un modelo entrenado desde la consola se activa con el botón **Recargar modelo** del dashboard.

## Contrato de la API para el ESP32

```http
POST /api/lecturas
Content-Type: application/json

{
  "device_id": "medidor-01",
  "lecturas": [
    { "timestamp": "2026-11-25T18:00:00Z", "power_kw": 0.412 },
    { "timestamp": "2026-11-25T18:00:01Z", "power_kw": 0.408 }
  ]
}
```

- Medir cada **1 s** y enviar un **lote cada 5 min**: 300 lecturas por lote (máximo 500). Así se respeta el límite de
  peticiones del plan gratuito de ngrok.
- El `timestamp` lleva zona horaria y sale del reloj del ESP32 sincronizado por **NTP**.
- `power_kw` es la potencia en **kilovatios** medida en ese segundo, entre 0 y 25. Por ejemplo, 412 W se envían
  como `0.412`.
- `device_id`: de 3 a 40 caracteres en minúsculas, números o guiones (por ejemplo, `medidor-01`).
- Respuesta `201`: `{"aceptadas": n, "duplicadas": m}`. Reenviar un lote es seguro, porque los duplicados se ignoran.

| Respuesta | Qué hacer |
|---|---|
| `201` | Borrar el lote del búfer |
| `422` (datos inválidos) | **No reintentar**; revisar los datos enviados |
| `5xx` o error de red | Guardar el lote y **reintentar** más tarde |

Si una sola lectura del lote es inválida, se rechaza el lote completo.

## Endpoints

| Método | Ruta | Descripción |
|---|---|---|
| `POST` | `/api/lecturas` | Recibe un lote de lecturas |
| `GET` | `/api/salud` | Estado del servicio y si hay modelo cargado |
| `GET` | `/api/dispositivos` | Dispositivos y tiempo desde su última lectura |
| `GET` | `/api/consumo?device_id=&desde=&hasta=` | Potencia promedio y máxima cada 7 min (por defecto, últimas 24 h; rango máximo de 31 días) |
| `GET` | `/api/predicciones?device_id=` | Última predicción e historial de 24 h |
| `POST` | `/api/modelo/entrenar?device_id=` | Inicia el entrenamiento en segundo plano (`202`; `409` si ya hay uno en curso) |
| `GET` | `/api/modelo/entrenamiento` | Estado del último entrenamiento: `idle`, `running`, `succeeded` o `failed`, con sus métricas |
| `POST` | `/api/modelo/recargar?device_id=` | Recarga el modelo desde disco y predice al momento |

El **Explorador de la API** (página del frontend) lee `/openapi.json` del backend y muestra cada endpoint con sus
parámetros, esquemas y respuestas. También permite **enviar peticiones de prueba** y genera el comando `curl`
equivalente.

## Modelo de predicción

El modelo predice **cuánta potencia promedio (kW) habrá dentro de ~1 hora**. Si esa predicción supera un umbral, la
marca como **pico** y el dashboard muestra una alerta. Todo el sistema trabaja en **kW**: lecturas, gráficas,
predicciones, umbral de pico y métricas.

### 1. De lecturas a una serie de intervalos de 7 minutos

El medidor envía una lectura de kW por segundo. Para el modelo eso es demasiado detalle y demasiado ruido, así que
primero se resume ([features.py](backend/app/ml/features.py)):

- **Promedio cada 7 min:** las ~420 lecturas de cada intervalo se convierten en un solo valor. Por ejemplo, de
  12:03 a 12:10 hubo 0,82 kW en promedio.
- **Huecos cortos:** si faltan datos hasta ~1 h (9 intervalos), se rellenan con una línea recta entre el valor
  anterior y el siguiente.
- **Huecos largos:** se dejan vacíos, para no inventar datos.

**Cuadrícula de intervalos.** Todos los intervalos se cuentan desde 1970-01-01 UTC, tanto en SQLite como en pandas
y en la predicción en vivo, para que siempre coincidan. Como 7 min no divide exactamente el día (1 440 / 7 = 205,7),
los intervalos empiezan en minutos distintos cada día (por ejemplo 12:03, 12:10, 12:17…) y la cuadrícula se corre
5 minutos de un día al siguiente.

**Duraciones redondeadas.** 7 min tampoco divide la hora, así que las duraciones se redondean al número de intervalos
más cercano:

| Concepto | Intervalos | Duración real |
|---|---|---|
| "1 hora" (horizonte, ventana móvil, huecos) | 9 | 63 min |
| "Ayer a esta hora" | 206 | 24 h 2 min |

### 2. Qué "mira" el modelo (features)

Para cada intervalo se calculan 10 datos de entrada:

| Feature | Qué representa | Por qué ayuda |
|---|---|---|
| `hour` | Hora local de Costa Rica (18,25 = 18:15) | El consumo depende de la hora: duchas en la mañana, cocina en la noche |
| `day_of_week` | Día de la semana (0 = lunes) | Entre semana y fin de semana el patrón cambia |
| `is_weekend` | 1 si es sábado o domingo | Refuerza lo anterior |
| `power_kw` | Consumo del intervalo actual | Punto de partida |
| `power_lag_1` | Consumo hace 7 min | Tendencia inmediata |
| `power_lag_2` | Consumo hace 14 min | ¿Va subiendo o bajando? |
| `power_lag_9` | Consumo hace ~1 h (63 min) | Tendencia de la última hora |
| `power_lag_206` | Consumo **a esta misma hora ayer** (24 h 2 min) | La pista más fuerte para predecir picos que se repiten cada día |
| `power_rolling_mean_1h` | Promedio de los últimos 9 intervalos (~1 h) | Nivel general reciente |
| `power_rolling_max_1h` | Máximo de los últimos 9 intervalos (~1 h) | Si hubo un salto reciente |

Todas se calculan **solo con datos del pasado**. Un test lo verifica, para que el modelo nunca "haga trampa" viendo
el futuro. El mismo código calcula las features al entrenar y al predecir, así que ambos usan exactamente las mismas
entradas.

### 3. Qué aprende a predecir (objetivo)

Para cada intervalo, la respuesta correcta es **el consumo 9 intervalos después**, o sea ~1 h adelante (63 min). Así,
el modelo aprende relaciones como: "a las 17:56 de un martes, con 0,3 kW ahora y 1,4 kW ayer a esta hora, dentro de
una hora habrá ~1,3 kW".

### 4. El algoritmo: `HistGradientBoostingRegressor`

Es un **conjunto de 300 árboles de decisión que se corrigen entre sí** ([training.py](backend/app/ml/training.py)):

1. El primer árbol hace una predicción burda, con reglas del tipo "si la hora está entre 17 y 21 y el consumo de ayer
   fue mayor que 1 kW, entonces 1,2 kW".
2. El segundo árbol no predice el consumo: predice **el error del primero** y lo corrige.
3. Así sucesivamente hasta 300 árboles (`max_iter=300`). Cada uno aporta solo el 5 % de su corrección
   (`learning_rate=0.05`), lo que lo hace lento para aprender pero estable.
4. "Hist" significa que agrupa los valores en rangos (histogramas): es rápido y **acepta datos faltantes**. Por
   ejemplo, si no hay datos de ayer, igual predice.

Se eligió este modelo porque entrena en segundos, acepta datos faltantes sin preprocesamiento y es el mismo enfoque
que usa scikit-learn en su ejemplo de pronóstico con lags. La documentación de scikit-learn advierte que, con muy
pocos datos, la versión clásica `GradientBoostingRegressor` puede ser preferible; comparar ambas con los datos reales
es una mejora pendiente.

### 5. Entrenamiento y evaluación

El entrenamiento se lanza con el botón **Entrenar modelo** del dashboard
([training_service.py](backend/app/services/training_service.py)):

- Corre en un **hilo aparte**, así que el servidor sigue recibiendo lecturas mientras entrena. Solo se permite **un
  entrenamiento a la vez**.
- Usa **todo el historial** del medidor. Para no cargar millones de lecturas en memoria, SQLite entrega directamente
  los promedios de 7 min.
- Al terminar, guarda el modelo, lo carga y genera una predicción inmediata.

Reglas del entrenamiento:

- **Mínimo 100 intervalos útiles de 7 min** (~11,7 h con objetivo conocido); si hay menos, se niega a entrenar.
- **Sin datos de ayer:** con menos de un día de historial, la feature `power_lag_206` queda vacía. scikit-learn no
  puede entrenar con una columna sin datos, así que se llena con una constante. Una columna constante no permite
  cortes en los árboles, de modo que el modelo la ignora; al predecir, cuando ya hay datos de ayer, el modelo sigue
  funcionando.
- **División cronológica 80/20:** entrena con el primer 80 % del tiempo y se evalúa con el último 20 %. Nunca se
  mezcla al azar, porque sería como estudiar con las respuestas del examen.
- **Umbral de pico = percentil 90** del consumo en el entrenamiento: el 10 % de los intervalos más altos cuentan como
  pico.
- **Comparación contra un baseline de persistencia**, que supone que "dentro de una hora habrá lo mismo que ahora".
  Si el modelo no le gana a eso, no sirve, y el dashboard lo advierte en amarillo.

| Métrica | Qué mide |
|---|---|
| MAE (kW) | Error promedio de la predicción |
| Precisión de picos | De las alertas que dio, cuántas fueron picos reales |
| **Recall de picos** | De los picos reales, cuántos anticipó. **Es la métrica clave**: importa más no dejar pasar un pico que dar alguna falsa alarma |
| Picos en prueba | Cuántos picos reales hubo en el periodo de prueba. Si es 0, el recall **no se puede evaluar** y el dashboard lo indica |

**Cuántos datos hacen falta.** Experimento con datos sintéticos (mismo patrón diario, distintas cantidades de
historial):

| Historial | Intervalos de entrenamiento / prueba | MAE modelo | MAE persistencia | Recall de picos (modelo / persistencia) |
|---|---|---|---|---|
| ~13 h (mínimo) | 84 / 21 | 0,144 kW | 0,016 kW | no evaluable (0 picos en prueba) |
| 1 día | 158 / 40 | 0,453 kW | 0,257 kW | no evaluable (0 picos en prueba) |
| 3 días | 487 / 122 | 0,028 kW | 0,184 kW | no evaluable (0 picos en prueba) |
| 14 días | 2 297 / 575 | 0,019 kW | 0,222 kW | 98 % / 56 % |

Con menos de un día el modelo es **peor que la persistencia**: todavía no vio el patrón diario completo. Desde unos
3 días ya la supera, y con 2 semanas también detecta los picos. **Se puede entrenar con 100 intervalos, pero lo
recomendable siguen siendo 2–4 semanas.** Con datos reales los números serán peores, porque los sintéticos son muy
regulares.

El resultado se guarda en un `.joblib` con el modelo, el umbral, la lista de features y las métricas.

### 6. Predicción en vivo

El backend predice solo ([prediction_service.py](backend/app/services/prediction_service.py)), **6 minutos después de
que cierra cada intervalo de 7 min**. Esa espera da tiempo a que llegue el lote del ESP32, que se envía cada 5 min.

Por ejemplo, a las 12:09 (hora de Costa Rica):

1. Carga las lecturas de las últimas ~26 h, que es lo necesario para tener el dato de "ayer a esta hora".
2. Calcula las features del último intervalo completo (11:56–12:03).
3. El modelo predice el consumo del intervalo **12:59–13:06** (9 intervalos después).
4. Si la predicción supera el umbral, la marca como pico.
5. La guarda en la tabla `predictions` y el dashboard la muestra. Si es un pico futuro, aparece el aviso rojo.

No predice si el medidor lleva más de 30 min sin enviar datos, para no alertar con información vieja.

### 7. Limitaciones del modelo

- **Predice promedios de 7 min.** Un pico de 2 minutos, como un microondas, se diluye en el promedio. El dashboard sí
  muestra el máximo de cada intervalo, pero el modelo no lo predice.
- **Con poco historial el modelo no es confiable** (ver la tabla anterior). Hay que reentrenar a medida que se
  acumulan días.
- **El umbral queda fijo** en lo que se vio en el entrenamiento. Si cambian los hábitos de consumo, hay que volver a
  pulsar "Entrenar modelo".
- **No conoce feriados ni eventos especiales.** Un feriado entre semana lo tratará como día laboral.
- **Solo mira ~1 hora adelante.** No sirve para planificar el día siguiente.

## Tests

Backend, desde `backend/`:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

Frontend, desde `frontend/` (incluye pruebas de las páginas con `streamlit.testing`, contra un backend simulado):

```powershell
.\.venv\Scripts\python.exe -m pytest
```

## Seguridad

El proyecto **no tiene autenticación**: no hay login ni API key. Cualquiera que conozca la URL de ngrok puede enviar
lecturas, ver el consumo, entrenar y recargar el modelo. Conviene no compartir la URL públicamente.

Lo que sí se mantiene:

- El servidor valida tipos, rangos y fechas de cada lectura, y rechaza los campos desconocidos. Así se evita que
  datos corruptos lleguen al modelo.
- Las respuestas del backend llevan encabezados de seguridad (`nosniff`, `X-Frame-Options`, CSP).
- Streamlit está configurado para **no enviar estadísticas de uso** (`gatherUsageStats = false`).
- Los archivos `.joblib` pueden ejecutar código al cargarse: solo se cargan modelos generados por este proyecto.

## Limitaciones

- El backend confía en que el medidor envía potencia real en kW; no puede corregir errores de calibración del
  medidor.
- Con una lectura por segundo se guardan unas 86 400 filas al día por medidor (unos 2,6 millones al mes). SQLite
  lo maneja bien, pero conviene respaldar el archivo con regularidad.
- La computadora local debe estar encendida y sin suspensión para no perder datos.
- Las métricas con datos sintéticos son optimistas: el patrón es muy regular.
