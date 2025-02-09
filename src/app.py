import streamlit as st
import numpy as np
import pandas as pd
import matplotlib as plt
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def main():

    #Pasos iniciales de diseño
    st.title("Solución del project Pagina Web de ML con Streamlit Leonardo Peña")
    st.header("Esto es un encabezado")
    st.subheader("Esto es un sub-encabezado")
    st.text("Esto es un texto")


    nombre = "Leo Herrera"

    st.text(f"Mi nombre es {nombre} y soy alumno")


    st.success("Este mi mensaje de aprobado con exito")
    st.warning("No se visualiza el project corregir y enviar")
    st.info("Tarea rechazada, vuelva a enviar")

    ##Inicial con el modelo
    #Datos dummies

    np.random.seed(42)

    X = np.random.rand(100,1)*10
    y = 3* X + 8 + np.random.rand(100,1)*2

    #Separar conjunto de datos entre train y test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size= 0.3, random_state=42)

    #generar el modelo, vamos a utilizar regresion lineal
    model = LinearRegression()
    model.fit(X_train, y_train)

    #prediccion

    y_pred = model.predict(X_test)
    nse = mean_squared_error(y_test,y_pred)

    ##La interfaz
    st.title("Mi primer regresion lineal en web")
    st.write(" este es un modelo de ejemplo para entregar el project")

    #Usar un SelectBox
    opcion = st.selectbox("Seleccione el tipo de visualizacion", ["Dispersion", "Linea de regresión"])

    #Checkbox para mostrar coeficientes

    if st.checkbox("Mostrar coeficientes de la regresion lineal"):
        st.write(f"coeficiente:{model.coef_[0][0]:.2f}")
        st.write(f"coeficiente interseccion:{model.intercept_[0][0]:.2f}")
        st.write(f"Error medio cuadratico: {nse:.2f}")
                 

    #Slider
    data_range = st.slider("Seleccion el rango que quiere evaluar",0,100,(10,90) )
    x_display = X_test[data_range[0]:data_range[1]]
    y_display = y_test[data_range[0]:data_range[1]]
    y_pred_display = y_pred[data_range[0]:data_range[1]]            



    




if __name__ == '__main__':
    main()