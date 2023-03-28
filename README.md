# MLOps para Dummies

Antes de iniciar con el entendimiento de lo que es MLOps, primero recordemos el proceso de desarrollo de un modelo de Machine Learning.

![image](https://docs.microsoft.com/en-us/azure/machine-learning/media/concept-ml-pipelines/pipeline-flow.png)

Partiendo de que los datos ya se encuentran disponibles para nosotros, los pasos que tenemos son los siguientes:

1. Preparación de los datos: En este paso se realiza la limpieza de los datos, se eliminan los datos faltantes, se normalizan los datos, se realiza la selección de características, etc.
2. Entrenamiento del modelo: En este paso se entrena el modelo con los datos de entrenamiento.
3. Empaquetado del modelo: En este paso se empaqueta el modelo para su despliegue.
4. Evaluación del modelo: En este paso se evalúa el modelo con los datos de prueba.
5. Despliegue del modelo: En este paso se despliega el modelo para su uso.
6. Monitoreo del modelo: En este paso se monitorea el modelo para detectar cambios en el comportamiento del modelo.

Ahora pensemos que pasa si nosotros queremos que este modelo sea repetible, escalable, que cuando se tenga un cambio en el modelo se pueda volver a entrenar, que se pueda desplegar en diferentes entornos, que se pueda monitorear, etc.  Todos estos pasos hacen parte del ciclo de vida de un modelo de Machine Learning.

Por lo cual para poder automatizar este proceso se creó MLOps.  Proceso en el cual vamos a poder hacer lo siguiente:

1. Versionar los datos: En este paso vamos a poder versionar los datos para que cuando se tenga un cambio en los datos se pueda volver a entrenar el modelo.
2. Versionar el código: En este paso vamos a poder versionar el código para que cuando se tenga un cambio en el código se pueda volver a entrenar el modelo.
3. Versionar el modelo: En este paso vamos a poder versionar el modelo para que cuando se tenga un cambio en el modelo se pueda volver a entrenar el modelo.
4. Automatizar el despliegue: En este paso vamos a poder automatizar el despliegue del modelo para que cuando se tenga un cambio en el modelo se pueda volver a desplegar el modelo.
5. Automatizar el monitoreo: En este paso vamos a poder automatizar el monitoreo del modelo para que cuando se tenga un cambio en el modelo se pueda volver a monitorear el modelo.

# MLOps

MLOps es un conjunto de prácticas y herramientas que permiten automatizar el ciclo de vida de un modelo de Machine Learning.  En la siguiente imagen vemos un ejemplo de un flujo de trabajo de MLOps:

![image](https://github.com/QuentinAmbard/databricks-demo/raw/main/product_demos/mlops-end2end-flow-0.png)

Como podemos notar hay diferentes roles que participan en el proceso de MLOps:

1. Data Scientist: Es el rol que se encarga de realizar el análisis de los datos, la preparación de los datos, el entrenamiento del modelo, la evaluación del modelo, etc.
    - **_Preparacion de los datos e ingenieria de caracteristicas_** 
    - **_Construir una linea base del modelo_**
    - **_Promover el mejor modelo para versionar_**

2. ML Engineer: Es el rol que se encarga de realizar la automatización del proceso de MLOps.
    - **_Configurar webhooks o notificaciones_** 
    - **_Automatizar la evalucion del modelo_**
    - **_Programar el reentrenamiento del modelo_**

3. Data Engineer: Es el rol que se encarga de realizar la automatización del proceso de ETL.

    - **_Se encarga de programar la ejecucion de inferencias en batch del modelo_**

> Tengamos en cuenta que le Data Engineer, antes de este proceso fue quien se encargo de realizar el proceso de ETL para que los datos esten disponibles para el Data Scientist.

Ahora que sabemos que es MLOps, vamos a ver como podemos implementar MLOps en nuestro proyecto de Machine Learning.

# Herramientas para MLOps

## MLFlow
MLflow es una plataforma para agilizar el desarrollo de aprendizaje automático, incluidos los experimentos de seguimiento, el empaquetado de código en ejecuciones reproducibles y el uso compartido e implementación de modelos. MLflow ofrece un conjunto de API ligeras que se pueden usar con cualquier aplicación o biblioteca de aprendizaje automático existente (TensorFlow, PyTorch, XGBoost, etc.), dondequiera que ejecute actualmente código ML (por ejemplo, en portátiles, aplicaciones independientes o en la nube). Los componentes actuales de MLflow son:

- MLflow Tracking: una API para registrar parámetros, código y resultados en experimentos de aprendizaje automático y compararlos mediante una interfaz de usuario interactiva.
- Proyectos MLflow: un formato de empaquetado de código para ejecuciones reproducibles usando Conda y Docker, para que pueda compartir su código ML con otros.
- Modelos de MLflow: un formato de empaquetado de modelos y herramientas que le permiten implementar fácilmente el mismo modelo (desde cualquier biblioteca de ML) para obtener puntuación por lotes y en tiempo real en plataformas como Docker, Apache Spark, Azure ML y AWS SageMaker.
- Registro de modelos de MLflow: un almacén de modelos centralizado, un conjunto de API y una interfaz de usuario para administrar de forma colaborativa el ciclo de vida completo de los modelos de MLflow.

Para mas informacion sobre MLFlow, pueden visitar el siguiente link: [https://mlflow.org/](https://mlflow.org/)

## DVC

Data Version Control (DVC) es un sistema de control de versiones de código abierto utilizado en proyectos de aprendizaje automático. También se conoce como Git para ML. Se trata de versiones de datos en lugar de versiones de código. DVC le ayuda a manejar modelos grandes y archivos de datos que no se pueden manejar con Git.  Le permite almacenar información sobre diferentes versiones de sus datos para realizar un seguimiento adecuado de los datos de ML y acceder al rendimiento de su modelo más adelante. Puede definir un repositorio remoto para enviar sus datos y modelos, lo que garantiza una colaboración sencilla entre los miembros del equipo.

El seguimiento de datos es algo necesario para cualquier flujo de trabajo de ciencia de datos. Aún así, se vuelve difícil para los científicos de datos administrar y rastrear los conjuntos de datos. Por lo tanto, existe la necesidad de versionar datos, que se puede lograr utilizando DVC. DVC es una de las herramientas convenientes que se pueden utilizar para sus proyectos de ciencia de datos. Estas son algunas de las razones para usar DVC:

- Permite que los modelos de ML sean reproducibles y compartan los resultados entre el equipo.
- Ayuda a administrar la complejidad de las canalizaciones de ML para que pueda entrenar el mismo modelo repetidamente.
- Permite a los equipos mantener archivos de versión para hacer referencia a modelos de ML y sus resultados rápidamente.
- Tiene todo el poder de las ramas Git.
- A veces, los miembros del equipo se confunden si los conjuntos de datos están etiquetados incorrectamente de acuerdo con la convención; DVC ayuda a etiquetar los conjuntos de datos correctamente.
- Los usuarios pueden trabajar en computadoras de escritorio, portátiles con GPU y recursos en la nube si necesitan más memoria.
- Su objetivo es excluir la necesidad de hojas de cálculo, herramientas y scripts ad hoc para compartir documentos para la comunicación.
- Los comandos push/pull se usan para mover paquetes coherentes de modelos, datos y código de aprendizaje automático a producción, máquinas remotas o al equipo de un compañero.

Para mas informacion sobre DVC, pueden visitar el siguiente link: [https://dvc.org/](https://dvc.org/)

Articulo que habla sobre MLFlow vs DVC: [https://censius.ai/blogs/dvc-vs-mlflow](https://censius.ai/blogs/dvc-vs-mlflow)

## Dvclive

DVCLive es una biblioteca de Python para registrar métricas de aprendizaje automático y otros metadatos en formatos de archivo simples, que es totalmente compatible con DVC.

Para mas informacion sobre Dvclive, pueden visitar el siguiente link: [https://dvc.org/doc/dvclive](https://dvc.org/doc/dvclive)


# Prerequisitos

### Crear un ambiente virtual
Para poder ejecutar el codigo de este tutorial, es necesario crear un ambiente virtual con python 3.10. Para crear el ambiente virtual, ejecutamos el siguiente comando:

```bash
conda create --name env-mlops python=3.10
```
Luego de crear el ambiente virtual, activamos el ambiente virtual con el siguiente comando:

```bash
conda activate env-mlops
```

### Instalar DVC

Para instalar DVC, ejecutamos el siguiente comando:

```bash
conda install -c conda-forge mamba # installs much faster than conda
mamba install -c conda-forge dvc
```

Teniendo instalado DVC, podemos inicializar nuestro proyecto ejecutamos:

```bash
dvc init
```

Esto nos creara una carpeta .dvc. En esta carpeta se guardaran los archivos de configuracion de DVC.




