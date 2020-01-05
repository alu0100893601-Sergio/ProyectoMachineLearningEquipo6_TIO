########################################
# Instalacion de paquetes
if (!requireNamespace("caret"))
  install.packages("caret")
if (!requireNamespace("tidyverse"))
  install.packages("tidyverse")
if (!requireNamespace("ggpubr"))
  install.packages("ggpubr")
if (!requireNamespace("randomForest"))
  install.packages("randomForest")
if (!requireNamespace("doParallel"))
  install.packages("doParallel")
if (!requireNamespace("dplyr"))
  install.packages("dplyr")
if (!requireNamespace("kernlab"))
  install.packages("kernlab")
if (!requireNamespace("gbm"))
  install.packages("gbm")

##################################
# Include
library(caret)
library(tidyverse)
library(randomForest)
library(ggpubr)
library(doParallel)
library(dplyr)
library(kernlab)
library(gbm)

###################################
# Paralelismo
parallel::detectCores()

cl <- makePSOCKcluster(4)

registerDoParallel(cl)

####################################
# Lectura de los datos
datos_castillo <- read_csv("C:\\Users\\Sergio\\Desktop\\final_dataset_castillodelromeral.csv", col_types = cols(.default = col_character()))
datos_castillo %>% mutate_if(is.character, as.factor)

#####################################
# Resumen de los datos
glimpse(datos_castillo)

####################################
# PREPROCESADO DE LOS DATOS
####################################
# Establecer NA de R
datos_castillo$NO[datos_castillo$NO == "N"] <- NA
datos_castillo$NO2[datos_castillo$NO2 == "N"] <- NA
datos_castillo$PM1.0[datos_castillo$PM1.0 == "N"] <- NA
datos_castillo$PM2.5[datos_castillo$PM2.5 == "N"] <- NA
datos_castillo$O3[datos_castillo$O3 == "N"] <- NA
datos_castillo$CO[datos_castillo$CO == "N"] <- NA
datos_castillo$NO2[datos_castillo$NO2 == "N"] <- NA

# Establecer todos los datos a numerico
datos_castillo$Hora <- as.numeric(datos_castillo$Hora)
datos_castillo$SO2 <- as.numeric(datos_castillo$SO2)
datos_castillo$NO <- as.numeric(datos_castillo$NO)
datos_castillo$PM1.0 <- as.numeric(datos_castillo$PM1.0)
datos_castillo$PM2.5 <- as.numeric(datos_castillo$PM2.5)
datos_castillo$O3 <- as.numeric(datos_castillo$O3)
datos_castillo$CO <- as.numeric(datos_castillo$CO)
datos_castillo$NO2 <- as.numeric(datos_castillo$NO2)

####################################
# Resumen de los datos
glimpse(datos_castillo)

# Obtener el numero de entradas
nrow(datos_castillo)

####################################
# Comprobar si hay alguna fila incompleta
any(!complete.cases(datos_castillo))

# Numero de datos ausentes por variable
map_dbl(datos_castillo, .f = function(x){sum(is.na(x))})

####################################
#Para eliminar las filas con valor nulo en la clase
datos_castillo <- datos_castillo[!is.na(datos_castillo$NO2),]
datos_castillo <- datos_castillo[!is.na(datos_castillo$NO),]
datos_castillo <- datos_castillo[!is.na(datos_castillo$PM1.0),]
datos_castillo <- datos_castillo[!is.na(datos_castillo$PM2.5),]
datos_castillo <- datos_castillo[!is.na(datos_castillo$O3),]
datos_castillo <- datos_castillo[!is.na(datos_castillo$CO),]
datos_castillo <- datos_castillo[!is.na(datos_castillo$SO2),]

####################################
# Numero de datos ausentes por variable
map_dbl(datos_castillo, .f = function(x){sum(is.na(x))})

# Obtener el numero de entradas
nrow(datos_castillo)

####################################
# Identificar que variables contienen valores ""
datos_castillo %>% map_lgl(.f = function(x){any(!is.na(x) & x == "")})

####################################
# Variables con varianza proxima a cero
datos_castillo %>% select(SO2, NO, PM1.0, PM2.5, O3, CO, NO2) %>%
          nearZeroVar(saveMetrics = TRUE)

# Mirar la varianza de Class
datos_castillo %>% select(NO2) %>% nearZeroVar(saveMetrics = TRUE)

####################################
# División de los datos en entrenamiento y testeo

set.seed(123)

# Se crean los indices de las observaciones de entrenamiento
train <- createDataPartition(y = datos_castillo$NO2, p = 0.8, list = FALSE, times = 1)
   
datos_train <- datos_castillo[train, ]
datos_test  <- datos_castillo[-train, ]

#####################################
#         RANDOM FOREST             #
#####################################

##################################
# 1.
# Metodo de entrenamiento repeatedcv sobre randomForest
#
# repeatedcv: permite hacer repeticiones
# trainControl: Controla los matices de cálculo de la función train
# tuneLength: El parámetro le dice al algoritmo que pruebe diferentes valores predeterminados
#   para el parámetro principal
# number: número de iteraciones de remuestreo
# metric: RMSE (Raiz del error cuadratico medio)
#
set.seed(123)
tc <- trainControl(method="repeatedcv", 
                   number=6, 
                   repeats=2, 
                   verboseIter = FALSE,
                   allowParallel = TRUE)
modelo <- train(NO2~.,
                data=datos_train ,
                method='rf',
                trControl=tc,
                tuneLength = 3,
                importance=TRUE,
                metric='RMSE')
modelo

# Vista de la importancia
#
# finalModel: muestra el tipo de modelo creado, el valor de los hiperparámetros 
#    e información adicional, entre ella, el error de entrenamiento
#
varImp(modelo)
modelo$finalModel

# Valores metricas
#
# getTrainPerf: proporciona los resultados de rendimiento promedio de 
#   los mejores parámetros ajustados promediados en los pliegues de 
#   validaciones cruzadas repetidas
#
getTrainPerf(modelo)

# Evaluar modelo
#
# predict: Hacer un objeto con predicciones de un objeto modelo ajustado
#     a un entrenamiento previo
# postResample: Dados dos vectores numéricos de datos, se calculan el error 
#     cuadrático medio y el R cuadrado
#
set.seed(123)
modelo_pred <- predict(modelo, 
                       newdata = datos_test ) 
postResample(modelo_pred, 
             datos_test$NO2)

####################################
# 2.
# Metodo de entrenamiento oob sobre randomForest
# 
# oob: 
#
set.seed(123)
tc <- trainControl("oob",
                   number=6, 
                   repeats=2, 
                   verboseIter = FALSE,
                   allowParallel = TRUE)
modelo <- train(NO2~.,
                data=datos_train,
                method='rf',
                trControl=tc,
                tuneLength=3,
                importance=TRUE,
                metric='RMSE')
modelo

# Vista de la importancia
#
varImp(modelo)
modelo$finalModel

# Valores metricas
#
getTrainPerf(modelo)

# Evaluar modelo
#
set.seed(123)
modelo_pred <- predict(modelo, 
                       newdata = datos_test ) 
postResample(modelo_pred, 
             datos_test$NO2)

####################################
# 3.
# Metodo de entrenamiento boot sobre randomForest
# 
# boot: 
#
set.seed(123)
tc <- trainControl("boot",
                   number=6, 
                   repeats=2, 
                   verboseIter = FALSE,
                   allowParallel = TRUE)
modelo <- train(NO2~.,
                data=datos_train,
                method='rf',
                trControl=tc,
                tuneLength=3,
                importance=TRUE,
                metric='RMSE')
modelo

# Vista de la importancia
#
varImp(modelo)
modelo$finalModel

# Valores metricas
#
getTrainPerf(modelo)

# Evaluar modelo
#
set.seed(123)
modelo_pred <- predict(modelo, 
                       newdata = datos_test ) 
postResample(modelo_pred, 
             datos_test$NO2)

#####################################
#         SVM_Linear                #
#####################################

##################################
# 1.
# Metodo de entrenamiento boot sobre SVM-Linear
#
set.seed(123)
tc <- trainControl("boot",
                   number=5,
                   verboseIter = FALSE,
                   allowParallel = TRUE)
modelo <- train(NO2~.,
                data=datos_train,
                method='svmLinear',
                trControl=tc,
                tuneLength=3,
                importance=TRUE,
                metric='RMSE')
modelo

# Vista de la importancia
#
varImp(modelo)
modelo$finalModel

# Valores metricas
#
getTrainPerf(modelo)

# Evaluar modelo
#
set.seed(123)
modelo_pred <- predict(modelo, 
                       newdata = datos_test ) 
postResample(modelo_pred, 
             datos_test$NO2)

#####################################
#         LM                        #
#####################################

##################################
# 1.
# Metodo de entrenamiento boot sobre LM
#
# lm: 
#

set.seed(123)
tc <- trainControl("boot",
                   number=6, 
                   allowParallel = TRUE)
modelo <- train(NO2~.,
                data=datos_train,
                method='lm',
                trControl=tc,
                tuneLength=3,
                metric='RMSE')
modelo

# Vista de la importancia
#
varImp(modelo)
modelo$finalModel

# Valores metricas
#
getTrainPerf(modelo)

# Evaluar modelo
#
set.seed(123)
modelo_pred <- predict(modelo, 
                       newdata = datos_test ) 
postResample(modelo_pred, 
             datos_test$NO2)
