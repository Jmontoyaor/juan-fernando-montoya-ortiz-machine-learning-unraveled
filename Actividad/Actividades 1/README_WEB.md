# MNIST: red densa vs. red convolucional — Actividad 1

## Cómo usarla

| Forma | Qué hace | Cómo se abre |
| --- | --- | --- |
| **Página web** | Dibujar, usar la cámara o subir una imagen; ver la corrección a fondo negro y trazo blanco, las predicciones de las tres redes y todas las métricas. Desde su menú se llega a Regresión y series de tiempo. No instala nada. | <https://jmontoyaor.github.io/juan-fernando-montoya-ortiz-machine-learning-unraveled/mnist/> |
| **App local (localhost)** | Red densa y CNN original con dibujo o imagen, y además **entrenar** desde la interfaz. La CNN con pooling se entrena por comando y se ve en la página web. | Doble clic en `iniciar_app.bat` (Windows) o `./iniciar_app.sh` (Linux/macOS). Se abre <http://localhost:8501>. |
| **Versión original OpenCV** | Cámara en tiempo real con osciloscopio (`app_cam.py`, `train.py`). Se conserva sin cambios. | Ver [README.md](README.md). |

> **¿Por qué el enlace de localhost no abre la app por sí solo?** `localhost` significa «este mismo equipo». El enlace solo funciona mientras el lanzador está corriendo en la PC de quien hace clic: GitHub no puede ejecutar programas en tu computador. La página web, en cambio, se ejecuta dentro de tu navegador: los pesos entrenados se descargan una vez y las operaciones de la red las hace JavaScript.

## Objetivo

Entrenar las dos arquitecturas de `scratchpad.py` con los mismos datos, compararlas sobre el mismo conjunto de prueba y usarlas para clasificar dígitos dibujados, fotografiados o cargados, convirtiendo antes cada entrada a la representación real de MNIST.

## Resultados reales (30 épocas, lote 32, semilla 42, CPU)

| | Red densa | Red convolucional | CNN con pooling |
| --- | ---: | ---: | ---: |
| Exactitud en prueba | **97.67 %** | **98.63 %** | **99.01 %** |
| Pérdida en prueba | 0.1646 | 0.1648 | 0.0695 |
| F1 macro | 0.9766 | 0.9862 | 0.9900 |
| Errores (de 10 000) | 233 | 137 | 99 |
| Parámetros | 55 050 | 387 466 | 50 186 |
| Tiempo de entrenamiento | 63 s | 627 s | 395 s |

Datos: 54 000 imágenes para entrenar, 6 000 para validar (tomadas del entrenamiento) y 10 000 de prueba reservadas. Python 3.12.10, TensorFlow 2.21.0, Windows 11, solo CPU.

**Cómo leerlos.** La CNN comete un 41 % menos de errores porque sus filtros aprovechan la vecindad entre píxeles. Las dos pérdidas de prueba son casi iguales aunque las exactitudes difieran: la pérdida castiga mucho los pocos errores cometidos con mucha confianza. En las curvas, la **pérdida de validación de la CNN sube desde la época 3** mientras la de entrenamiento llega casi a 0: es **sobreajuste**, esperable en una arquitectura sin pooling, sin dropout y con 368 650 parámetros en la última capa. La red densa muestra lo mismo, más suave: su pérdida de validación es mínima en la época 5. Treinta épocas es más de lo necesario; *early stopping*, pooling o dropout lo reducirían. Esas mejoras no se aplicaron a la CNN original para mantener la arquitectura de referencia.

**CNN con pooling (`scratchpad_3.ipynb`).** Es la versión actualizada de la clase y la mejor de las tres: 99.01 % con 50 186 parámetros, casi 8 veces menos que la CNN sin pooling. El MaxPooling reduce los mapas de 28×28 a 7×7 antes de la capa final, así que esa capa pasa de 368 650 a 31 370 parámetros y la red generaliza mejor: su pérdida de prueba es menos de la mitad (0.0695 frente a 0.1648). También sobreajusta, aunque mucho menos: su pérdida de validación es mínima en la época 3 (0.066) y termina en 0.101.

## Instalación y ejecución

Requiere **Python 3.12 de 64 bits**. TensorFlow 2.21 corre en CPU en Windows nativo.

**Windows (PowerShell), desde esta carpeta:**

```powershell
.\iniciar_app.bat
```

**Linux/macOS:**

```bash
chmod +x iniciar_app.sh
./iniciar_app.sh
```

El lanzador crea el entorno virtual en `~/.venvs/mnist-a1` si no existe, instala `requirements-app.txt`, inicia el servidor en `127.0.0.1:8501` y abre el navegador. El servidor se detiene con **Ctrl+C** o cerrando la ventana, y se reinicia ejecutando de nuevo el lanzador.

> El entorno vive en una ruta corta y no dentro de esta carpeta porque TensorFlow incluye archivos con rutas muy largas. En `...\Actividad\Actividades 1\.venv` superan el límite de 260 caracteres de Windows y la instalación queda rota.

Comando manual equivalente:

```powershell
py -3.12 -m venv "$HOME\.venvs\mnist-a1"
& "$HOME\.venvs\mnist-a1\Scripts\python.exe" -m pip install -r requirements-app.txt
& "$HOME\.venvs\mnist-a1\Scripts\python.exe" -m streamlit run web_app.py --server.address 127.0.0.1 --server.port 8501
```

El servidor solo escucha en `127.0.0.1`: no queda expuesto a la red ni a Internet.

## Entrenamiento, evaluación y publicación

```powershell
$py = "$HOME\.venvs\mnist-a1\Scripts\python.exe"
& $py train_mnist.py --kind dense --epochs 30 --batch-size 32 --seed 42   # ~1 min
& $py train_mnist.py --kind cnn   --epochs 30 --batch-size 32 --seed 42   # ~10 min
& $py train_mnist.py --kind cnn_pool --epochs 30 --batch-size 32 --seed 42  # ~7 min
& $py export_web.py                                                         # actualiza docs/mnist/
```

- `--samples 2048 --epochs 1` es el **modo rápido** para comprobar el flujo; su resultado no es comparable con el de 30 épocas.
- Desde la app local, cada pestaña tiene un formulario «Entrenar». El entrenamiento corre en **otro proceso**, así que la interfaz sigue usable; cambiar de pestaña o mover un control no reentrena.
- Cada entrenamiento guarda `artifacts/mnist_{dense,cnn}.keras` y `artifacts/{dense,cnn}_results.json`. Este último contiene el historial, la configuración, el tiempo, los parámetros, las métricas de prueba, la matriz de confusión, los errores y la descripción del preprocesamiento. La app los carga al reiniciar.
- `export_web.py` copia los pesos y resultados a `docs/mnist/model_data.js`, que es lo que usa la página de GitHub Pages.
- Se versionan los modelos y resultados de `artifacts/` (pocos MB), de modo que la app funciona al clonar sin reentrenar. No se versionan los registros, los datos descargados ni el entorno virtual.

## Estructura

```text
Actividades 1/
├── web_app.py              # interfaz Streamlit (localhost)
├── iniciar_app.bat / .sh   # lanzadores
├── train_mnist.py          # entrenamiento por línea de comandos
├── export_web.py           # exporta modelos y métricas a docs/mnist/
├── mnist_app/
│   ├── config.py           # rutas, semilla, nombres de modelos
│   ├── data.py             # carga de MNIST, normalización y particiones
│   ├── preprocessing.py    # dibujos e imágenes externas → formato MNIST
│   ├── models.py           # las dos arquitecturas de referencia
│   ├── training.py         # entrenamiento y persistencia
│   └── evaluation.py       # métricas sobre el conjunto de prueba
├── artifacts/              # modelos .keras y resultados .json
├── tests/                  # pruebas Python (unittest) y JavaScript (Node)
├── train.py, app_cam.py, test_cam.py, *.keras, scratchpad.ipynb   # versión original, sin cambios
└── README.md               # guía de la versión original
docs/mnist/                 # página web estática (GitHub Pages)
├── index.html
├── mnist_core.js           # preprocesamiento + inferencia en JavaScript
└── model_data.js           # pesos exportados (generado)
```

## Arquitecturas (fieles a `scratchpad.py` y `scratchpad_3.ipynb`)

| Red | Arquitectura | Parámetros |
| --- | --- | ---: |
| Densa | `Input(784) → Dense(64, relu) → Dense(64, relu) → Dense(10, softmax)` | 55 050 |
| CNN | `Input(28,28,1) → Conv2D(32, 3×3, relu) → Conv2D(64, 3×3, relu) → Flatten → Dense(10, softmax)` | 387 466 |
| CNN con pooling | `Input(28,28,1) → Conv2D(32, 3×3, same, relu) → MaxPool(2×2) → Conv2D(64, 3×3, same, relu) → MaxPool(2×2) → Flatten → Dense(10, softmax)` | 50 186 |

Las tres usan Adam y `sparse_categorical_crossentropy`, y se entrenaron con la misma partición y la misma normalización (el notebook original entrena con píxeles de 0 a 255 y sin validación; aquí se normalizan a [0, 1] para que las tres sean comparables). La versión OpenCV original (`train.py`) usa una CNN distinta, con MaxPooling y una Dense(64) extra. Se conserva como variante histórica, pero no es la de referencia.

**Red densa.** Aplana la imagen en 784 números; cada neurona combina todos los píxeles: $h = \max(0, Wx + b)$. Conteo: $784·64+64 + 64·64+64 + 64·10+10 = 55\,050$.

**CNN.** Cada filtro 3×3 se desliza por la imagen y produce un mapa de características: $y_{i,j} = \mathrm{ReLU}\left(\sum_{u,v} w_{u,v}\,x_{i+u,j+v} + b\right)$. Sin padding, cada convolución reduce 2 píxeles (28 → 26 → 24). Conteo: $(9·1·32+32) + (9·32·64+64) + (24·24·64·10+10) = 387\,466$; el 95 % está en la última capa.

**CNN con pooling.** Con `padding="same"` se rellena con ceros el borde para que la convolución conserve el tamaño, y `MaxPooling 2×2` toma el máximo de cada bloque de 2×2: $\text{pool}_{i,j} = \max(y_{2i,2j}, y_{2i,2j+1}, y_{2i+1,2j}, y_{2i+1,2j+1})$. Tamaños: 28 → 28 → 14 → 14 → 7. Conteo: $(9·1·32+32) + (9·32·64+64) + (7·7·64·10+10) = 50\,186$.

**Filtros y mapas en la página.** Las pestañas de las dos CNN muestran, por cada capa convolucional, una grilla de 32 subplots con los pesos de los filtros y otra con los 32 mapas de activación de tu dígito. En la segunda capa cada filtro tiene un 3×3 por cada mapa de entrada; se dibuja el del mapa 0 (`W[:, :, 0, k]`), igual que en el notebook.

**Salida.** Softmax convierte los 10 puntajes en probabilidades: $p_k = e^{z_k}/\sum_j e^{z_j}$. La pérdida es $-\log p_{y}$, el logaritmo de la probabilidad asignada a la clase correcta.

## Normalización y polaridad: fondo negro, trazo blanco

En MNIST cada píxel es un número: **0 = negro = fondo**, **1 = blanco = trazo** y grises intermedios en los bordes. Blanco y negro no significan «reconocido» o «no reconocido»; solo indican dónde hay tinta. Las redes aprendieron con esa convención, así que toda entrada se transforma así antes de predecir:

1. **Escala de grises.** La transparencia se compone sobre blanco, como si fuera papel.
2. **Polaridad.** Se mira el borde de la imagen, que casi siempre es fondo. Si su mediana es clara (> 127), es un dígito oscuro sobre papel claro y se invierte: $p' = 255 - p$. El control «Polaridad» permite forzarlo si la detección falla.
3. **Limpieza (solo cámara).** El umbral de Otsu separa automáticamente tinta y papel, y los grises del papel o de las sombras pasan a 0.
4. **Recorte** del trazo, ignorando 2 píxeles del marco del lienzo.
5. **Escalado sin deformar.** El lado mayor pasa a 20 píxeles, conservando la proporción, y se pega centrado en 28×28.
6. **Centrado por centro de masa**, como en MNIST: $\bar r = \sum r\,x_{r,c} / \sum x_{r,c}$, desplazando hasta $(\bar r,\bar c) = (14,14)$.
7. **Normalización** $x = p/255$, **una sola vez**, igual que en el entrenamiento.

Si la imagen no tiene contraste o tiene menos de 3 píxeles de tinta, se rechaza y **no** se muestra ninguna predicción. La app enseña cada paso y los 784 números finales.

## Entrenamiento, validación y prueba

- **Entrenamiento (54 000):** con estos datos se ajustan los pesos.
- **Validación (6 000, tomada del entrenamiento):** se mide en cada época para vigilar el sobreajuste. No ajusta pesos.
- **Prueba (10 000):** se usa una sola vez al final. Es la estimación honesta del desempeño con datos nuevos.

La partición es estratificada y usa `random_state=42`, la misma para ambas redes. Las semillas fijan Python, NumPy y TensorFlow, pero otro hardware o versión puede producir diferencias numéricas pequeñas por el orden de las operaciones en paralelo.

## Interpretación de las métricas

- **Exactitud:** fracción de aciertos.
- **Precisión de la clase k:** de las veces que la red dijo «k», cuántas eran k. $TP/(TP+FP)$.
- **Recall de la clase k:** de los k reales, cuántos encontró. $TP/(TP+FN)$.
- **F1:** media armónica de precisión y recall. El **F1 macro** promedia las 10 clases por igual.
- **Matriz de confusión:** la fila es la clase real y la columna la predicha; lo que queda fuera de la diagonal son errores. Por ejemplo, la confusión 7 → 2 aparece en ambas redes.

## Limitaciones con dibujos y fotos

- La exactitud reportada vale para MNIST. Un dibujo propio es **un solo caso** y puede diferir por el grosor del trazo, el estilo o un dígito incompleto.
- Softmax **siempre** reparte el 100 % entre 10 clases: una letra o un garabato también reciben un «dígito» con alta probabilidad. No es un detector de «esto no es un número».
- Con cámara influyen la luz, las sombras, el desenfoque y los números que tocan el borde del recuadro. Lo mejor es usar marcador grueso sobre papel blanco liso.
- La versión web hace exactamente los mismos cálculos que Keras: la diferencia máxima medida es de 4·10⁻⁷ sobre 32 imágenes. Su preprocesamiento redimensiona por promedio de área en lugar de Lanczos, así que puede diferir levemente de la app local en los bordes grises.

## Regresión y series de tiempo (`scratchpad_General.ipynb`)

Página: <https://jmontoyaor.github.io/juan-fernando-montoya-ortiz-machine-learning-unraveled/mnist/senales.html>. Se llega desde el menú de la página de MNIST.

### Regresión

La salida es un número continuo, no una clase. Se mide con $\text{MSE} = \frac{1}{n}\sum (y_i - \hat y_i)^2$ y $\text{MAE} = \frac{1}{n}\sum |y_i - \hat y_i|$. Partición aleatoria 80/20 (`random_state=42`), Adam y lote 32, como en el notebook.

| Ejemplo | Red | Parámetros | Épocas | MSE en prueba | Resultado |
| --- | --- | ---: | ---: | ---: | --- |
| Recta $y = 3x + 1$, $x \in [-10, 10)$ | `Dense(1)` | 2 | 100 | 1.0·10⁻⁷ | aprendió $y = 2.9999x + 0.9999$ |
| Coseno $y = \cos(2\pi \cdot 2x)$, $x \in [-2, 2)$ | `Dense(64, relu) ×3 → Dense(1)` | 8 513 | 300 | 0.0010 | reproduce las 8 oscilaciones |

La página dibuja también las predicciones **fuera del rango de entrenamiento**. La recta extrapola bien porque su modelo tiene la misma forma que la función real. El coseno no: fuera de $[-2, 2]$ la red con ReLU se vuelve una recta (en $x = 2.8$ predice 5.25 cuando el valor real es −0.81). Una red neuronal interpola dentro de sus datos, pero no descubre la fórmula.

### Series de tiempo: comparación justa de RNN, LSTM y GRU

Con 1000 muestras pasadas, cada red predice las 200 siguientes. Se evalúa en dos modos:
- **Directo:** una sola predicción de 200 muestras.
- **Autorregresivo:** 10 pasos de 200, en los que cada predicción entra en la ventana del paso siguiente, así que el error se acumula.

**Igualdad de condiciones.** Las tres redes recurrentes y la DNN de referencia comparten:
- la misma señal sintética (40 000 muestras: nivel, tendencia, dos ciclos y ruido) y las mismas ventanas (537 de entrenamiento, 97 de validación y 97 de prueba, una cada 50 muestras);
- una división **cronológica** 70/15/15, para no entrenar con el futuro;
- la normalización $z = (x - \mu)/\sigma$ con $\mu$ y $\sigma$ calculadas **solo con entrenamiento**;
- 64 unidades, Adam (lr = 0.001), MSE, lote 32, hasta 100 épocas sin barajar;
- EarlyStopping (paciencia 15, restaura los mejores pesos) y ReduceLROnPlateau (×0.5, paciencia 7);
- la semilla 42, reiniciada antes de construir cada red.

| | DNN (referencia) | RNN | LSTM | GRU |
| --- | ---: | ---: | ---: | ---: |
| Arquitectura | `Dense(64, relu) → Dense(200)` | `SimpleRNN(64) → Dense(200)` | `LSTM(64) → Dense(200)` | `GRU(64) → Dense(200)` |
| Parámetros | 77 064 | 17 224 | 29 896 | 25 864 |
| Épocas (early stopping) | 100 | 52 | 52 | 100 |
| Tiempo (CPU) | 20 s | 87 s | 294 s | 596 s |
| MSE en las 97 ventanas de prueba (normalizado) | **0.0043** | 0.1872 | 0.1915 | **0.1570** |
| MSE directo, 200 muestras (unidades reales) | 0.0009 | 0.0149 | 0.0142 | **0.0077** |
| MSE autorregresivo, 2000 muestras (unidades reales) | 0.0013 | 0.5630 | **0.4467** | 0.4850 |

**Cómo leerlo.** Entre las tres recurrentes, la **GRU** tiene el menor error en las ventanas de prueba y en la predicción directa. La **LSTM** acumula menos error en el modo autorregresivo. La RNN simple, sin compuertas, es la más débil en general.

La **DNN de referencia es unas 36 veces mejor** que la mejor recurrente, igual que en el notebook. No es un fallo del experimento. La señal es casi periódica y la DNN ve las 1000 muestras a la vez, así que una combinación casi lineal de ellas le basta para reproducir el patrón. Las recurrentes tienen que llevar esa información a través de 1000 pasos de memoria, donde el gradiente se debilita (desvanecimiento del gradiente). Las compuertas de la LSTM y la GRU lo mitigan, pero no lo eliminan con solo 64 unidades.

Que la comparación sea justa no significa que los resultados sean iguales: significa que las diferencias vienen de la arquitectura y no de los datos ni del entrenamiento.

En la página se puede elegir dónde empieza la predicción dentro del conjunto de prueba. Las cuatro redes predicen **en el navegador**: el motor JavaScript implementa Dense, SimpleRNN, LSTM y GRU con los pesos y el orden de compuertas de Keras, con una diferencia máxima medida de 10⁻⁶.

```powershell
$py = "$HOME\.venvs\mnist-a1\Scripts\python.exe"
& $py train_signals.py --task regression    # ~1 min
& $py train_signals.py --task timeseries    # ~17 min en CPU
& $py export_web.py --part signals          # actualiza docs/mnist/signals_data.js
```

Código: `signal_models/regression.py`, `signal_models/timeseries.py`, `train_signals.py`; página: `docs/mnist/senales.html` y `docs/mnist/signals_core.js`.

## Cómo reproducir y verificar

```powershell
$py = "$HOME\.venvs\mnist-a1\Scripts\python.exe"
& $py -m unittest discover -s tests      # preprocesamiento, modelos, guardado/recarga, series y regresión
& $py train_mnist.py --kind dense ; & $py train_mnist.py --kind cnn ; & $py train_mnist.py --kind cnn_pool
& $py export_web.py
node tests/test_web_core.js               # MNIST: JavaScript vs. Keras, polaridad, lienzo vacío, cámara
node tests/test_signals_core.js           # Regresión y series: JavaScript vs. Keras
```

## Ampliaciones futuras (hoja de ruta, fuera de esta actividad)

Procesos estocásticos, procesamiento de lenguaje natural, otras tareas de imágenes, despliegue en la nube y sistemas embebidos son temas para actividades posteriores. Regresión, series de tiempo y predicción autorregresiva ya están implementadas (sección anterior).
