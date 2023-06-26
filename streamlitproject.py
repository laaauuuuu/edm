# -*- coding: utf-8 -*-

# pip install streamlit
# pip install folium
# pip install folium_static

import streamlit as st
import pandas as pd
import numpy as np
import base64
import plotly.express as px
import altair as alt
from PIL import Image
from pmdarima.arima import auto_arima

contenedores = pd.read_csv('Contenedores_varios.csv', sep = ';', encoding='iso-8859-1')
residuos2021 = pd.read_csv('Modelo_Residuos_2021.csv', sep = ';', encoding='iso-8859-1')
residuos2022 = pd.read_csv('Modelo_Residuos_2022.csv', sep = ';', encoding='iso-8859-1')
residuos2023 = pd.read_csv('Modelo_Residuos_2023.csv', sep = ';', encoding='iso-8859-1')

# Cambios en los Dataframes ...........................................................................................................................................
cambio_col_res2023 = {'ï»¿AÃ±o': 'Año', 'NOMBRE': 'Nombre', 'Fracción': 'Residuo', 'Total':'Toneladas'}
residuos2023 = residuos2023.rename(columns=cambio_col_res2023)


cambio_col_res2021 = {'ï»¿AÃ±o': 'Año', 'Nombre Distrito': 'Nombre'}
residuos2021 = residuos2021.rename(columns=cambio_col_res2021)


cambio_col_res2022 = {'Fracción': 'Residuo', 'NOMBRE': 'Nombre', 'Total':'Toneladas'}
residuos2022 = residuos2022.rename(columns=cambio_col_res2022)



def change_value(dataframe, col, valor_original, valor_nuevo):
    dataframe.loc[dataframe[col] == valor_original, col] = valor_nuevo
    
change_value(contenedores, 'Distrito', 'VILLAÂ DEÂ VALLECAS', 'VILLA DE VALLECAS') 
change_value(contenedores, 'Distrito', 'PUENTEÂ\xa0DEÂ\xa0VALLECAS', 'PUENTE DE VALLECAS') 
change_value(contenedores, 'Distrito', 'FUENCARRAL-ELÂ\xa0PARDO', 'FUENCARRAL - EL PARDO') 
change_value(contenedores, 'Distrito', 'MONCLOA-ARAVACA', 'MONCLOA - ARAVACA') 

   
change_value(residuos2021, 'Nombre', 'TetuÃ¡n', 'Tetuan')
change_value(residuos2021, 'Nombre', 'ChamberÃ\xad', 'Chamberi')
change_value(residuos2021, 'Nombre', 'ChamartÃ\xadn', 'Chamartin')
change_value(residuos2021, 'Nombre', 'VicÃ¡lvaro', 'Vicalvaro')


change_value(residuos2022, 'Nombre', 'TetuÃ¡n', 'Tetuan')
change_value(residuos2022, 'Nombre', 'ChamberÃ\xad', 'Chamberi')
change_value(residuos2022, 'Nombre', 'ChamartÃ\xadn', 'Chamartin')
change_value(residuos2022, 'Nombre', 'VicÃ¡lvaro', 'Vicalvaro')

change_value(residuos2023, 'Nombre', 'TetuÃ¡n', 'Tetuan')
change_value(residuos2023, 'Nombre', 'ChamberÃ\xad', 'Chamberi')
change_value(residuos2023, 'Nombre', 'ChamartÃ\xadn', 'Chamartin')
change_value(residuos2023, 'Nombre', 'VicÃ¡lvaro', 'Vicalvaro')

# Creamos un único Dataframe...........................................................................................................................................

residuos_juntos = pd.concat([residuos2021, residuos2022, residuos2023], axis = 0)
residuos_juntos['Toneladas'] = residuos_juntos['Toneladas'].str.replace('.', '').str.replace(',', '.').astype(float)
residuos_juntos.Nombre = residuos_juntos.Nombre.str.upper()
residuos_juntos = residuos_juntos[residuos_juntos['Nombre'] != 'SIN DISTRITO']
change_value(residuos_juntos, 'Nombre', 'CHAMARTÍN', 'CHAMARTIN')
change_value(residuos_juntos, 'Nombre', 'CHAMBERÍ', 'CHAMBERI')
change_value(residuos_juntos, 'Nombre', 'TETUÁN', 'TETUAN')
change_value(residuos_juntos, 'Nombre', 'VICÁLVARO', 'VICALVARO')
change_value(residuos_juntos, 'Residuo', 'CARTON COMERCIAL', 'PAPEL-CARTON')
change_value(residuos_juntos, 'Residuo', 'CAMA DE CABALLO', 'OTROS')
change_value(residuos_juntos, 'Residuo', 'CLINICOS', 'OTROS')
change_value(residuos_juntos, 'Residuo', 'CONTENEDORES DE ROPA', 'OTROS')
change_value(residuos_juntos, 'Residuo', 'PUNTOS LIMPIOS', 'OTROS')
change_value(residuos_juntos, 'Residuo', 'RCD', 'OTROS')
change_value(residuos_juntos, 'Residuo', 'ROPA', 'OTROS')
change_value(residuos_juntos, 'Residuo', 'VIDRIO COMERCIAL', 'VIDRIO')


def main():
    st.set_page_config(layout="wide", page_title="EcoMadrid", 
                       page_icon="♻️")

    st.title("Madrid Clean and Sustainable: A Look at Its Waste :recycle:")    

    image = Image.open('residuos_madrid.jpg')
    st.image(image, caption='Waste collection in the eastern area of Madrid')
    st.write('---')
# Descargar csv de residuos---------------------------------------------------------------------------------------------------------------------------------------------    
    st.subheader(":blue[Download selected Data]")

    col1, col2, col3 = st.columns(3)    
    
    with col1:
        sorted_unique_nombre = sorted(residuos_juntos.Nombre.unique())
        selected_nombre = st.multiselect('District', sorted_unique_nombre, sorted_unique_nombre)
    with col2:
       sorted_unique_year = sorted(residuos_juntos.Año.unique())
       selected_year = st.multiselect('Year', sorted_unique_year, sorted_unique_year)
    with col3:
        sorted_unique_tipo = sorted(residuos_juntos.Residuo.unique())
        selected_tipo = st.multiselect('Waste type', sorted_unique_tipo, sorted_unique_tipo)
        
    # Filtramos los datos por la selección del usuario
    df_selected = residuos_juntos[(residuos_juntos.Nombre.isin(selected_nombre)) & (residuos_juntos.Año.isin(selected_year)) & (residuos_juntos.Residuo.isin(selected_tipo))]
    
    if df_selected.shape[0] < 1 or df_selected.shape[0] < 1:
        st.error('Select at least one option for each attribute')
    else:
        st.write('Data size: ' + str(df_selected.shape[0]) + ' rows and ' + str(df_selected.shape[1]) + ' columns.')
        st.dataframe(df_selected)
        
        def filedownload(df):
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
            href = f'<a href="data:file/csv;base64,{b64}" download="residuos.csv">Download CSV File</a>'
            return href
    
        st.markdown(filedownload(df_selected), unsafe_allow_html=True)
    st.write('---')
    
# Mapa de contenedores -------------------------------------------------------
    st.subheader(":blue[Containers map]")

    col1, col2, col3 = st.columns(3)

    tipos_contenedores = contenedores['Tipo Contenedor'].unique()
    tipos_contenedores = np.append('TODOS', sorted(tipos_contenedores))
    
    distritos = contenedores['Distrito'].unique()
    distritos = np.append('TODOS', sorted(distritos))
    
    with col1:
        tipo_seleccionado = st.selectbox('Select container type:', tipos_contenedores)
    with col2:
        distrito_seleccionado = st.selectbox('Select district:', distritos)
    
    colors = {'VIDRIO': '#0EC100', 'PAPEL-CARTON': '#0007C3', 'ENVASES': '#FFCF00', 'ORGANICA': '#B55F00', 'RESTO': '#7D7D7D'}
    
    if tipo_seleccionado == 'TODOS' and distrito_seleccionado == 'TODOS':
        contenedores_filtrados = contenedores
        
    elif tipo_seleccionado == 'TODOS':
        contenedores_filtrados = contenedores[contenedores['Distrito'] == distrito_seleccionado]
        
    elif distrito_seleccionado == 'TODOS':
        contenedores_filtrados = contenedores[contenedores['Tipo Contenedor'] == tipo_seleccionado]
        
    else:
        contenedores_filtrados = contenedores[(contenedores['Tipo Contenedor'] == tipo_seleccionado) & (contenedores['Distrito'] == distrito_seleccionado)]
        
    if len(contenedores_filtrados)  == 0:
        st.error("There is not enough data to display the map")
        
    else:
        fig = px.scatter_mapbox(contenedores_filtrados, lat="LATITUD", lon="LONGITUD", hover_name="Tipo Contenedor",
                                color="Tipo Contenedor", color_discrete_map=colors, zoom=10)
        fig.update_traces(marker=dict(opacity=0.9, size = 5.75))
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        
        with col3:
            tipo_mapa = st.radio('Select map type:', ['Ligth', 'Dark', 'Geographic'])
        if tipo_mapa == 'Dark':
            fig.update_layout(mapbox_style="carto-darkmatter", mapbox=dict(center=dict(lat=40.4488, lon=-3.7038)), height=500, width=1380)
            st.plotly_chart(fig)
            
        elif tipo_mapa == 'Geographic':
            fig.update_layout(mapbox_style="stamen-terrain", mapbox=dict(center=dict(lat=40.4488, lon=-3.7038)), height=500, width=1380)
            st.plotly_chart(fig)
            
        else:
            fig.update_layout(mapbox_style="open-street-map", mapbox=dict(center=dict(lat=40.4488, lon=-3.7038)), height=500, width=1380)
            st.plotly_chart(fig)
    st.write('---') 
# Info varia -------------------------------------------------------
    st.subheader(':blue[Varied information by district]')
    
    col1, col2 = st.columns(2)
    
    with col1:
        sorted_unique_nombre = sorted(residuos_juntos.Nombre.unique())
        distritos_seleccionado = st.multiselect('Select district:', sorted_unique_nombre, sorted_unique_nombre)
    with col2:
        sorted_unique_tipos = sorted(residuos_juntos.Residuo.unique())
        tipo_sel = st.multiselect('Select waste type', sorted_unique_tipos, sorted_unique_tipos)
     
    if  len(distritos_seleccionado) < 1 or len(tipo_sel) < 1:
        st.error('Choose at least one district and one type of waste')
    else:
        df = residuos_juntos[(residuos_juntos.Nombre.isin(distritos_seleccionado)) & (residuos_juntos.Residuo.isin(tipo_sel))]
        
        meses = {'enero': '01',
            'febrero': '02',
            'marzo': '03',
            'abril': '04',
            'mayo': '05',
            'junio': '06',
            'julio': '07',
            'agosto': '08',
            'septiembre': '09',
            'octubre': '10',
            'noviembre': '11',
            'diciembre': '12'}
        
        colors = {'VIDRIO': '#0EC100', 'PAPEL-CARTON': '#0007C3', 'ENVASES': '#FFCF00', 'ORGANICA': '#B55F00', 'RESTO': '#7D7D7D', 'OTROS': '#FFFFFF', 'TOTAL': '#FF0000'}
        
    
        df['Mes'] = df['Mes'].str.lower().map(meses)
    
        df['Fecha'] = pd.to_datetime(df['Año'].astype(str) + '-' + df['Mes'] + '-' + '01')
        
        df =  df.groupby(['Fecha', 'Residuo'])['Toneladas'].sum().reset_index()
        
        df_total = df.groupby('Fecha')['Toneladas'].sum().reset_index()
        df_total['Residuo'] = 'TOTAL'
        
        df = pd.concat([df, df_total])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Tons/month')
            
            line_chart = alt.Chart(df).mark_line().encode(
                x=alt.X('Fecha:T', title='Date'),
                y=alt.Y('Toneladas:Q', title='Tons'),
                color=alt.Color('Residuo:N', legend=alt.Legend(title='Residuo'), scale=alt.Scale(domain=list(colors.keys()), range=list(colors.values())))
            ).properties(
                width=600,
                height=400
            )
                
            line_chart
                
        with col2:
            st.subheader('Container count')
            
            conteo_madrid = contenedores['Tipo Contenedor'].value_counts()
            total = sum(conteo_madrid)
            
            contenedores_seleccionados = contenedores[(contenedores['Distrito'].isin(distritos_seleccionado) & contenedores['Tipo Contenedor'].isin(tipo_sel))]
            
            
            
            conteo_residuos = contenedores_seleccionados['Tipo Contenedor'].value_counts()
            conteo_residuos = pd.DataFrame(conteo_residuos).rename(columns={'Tipo Contenedor': 'Tipo de contenedor', 'count': 'Nº Contenedores'})
            conteo_residuos['% Selección'] = round((conteo_residuos['Nº Contenedores'] / len(contenedores_seleccionados)) * 100, 3)
            conteo_residuos['% Madrid'] = round((conteo_residuos['Nº Contenedores'] / total) * 100, 3)
            
            df_media = pd.DataFrame(df.groupby('Residuo')['Toneladas'].sum()).rename(columns={'Toneladas':'Toneladas med./mes'})
            df_media['Toneladas med./mes'] = round(df_media['Toneladas med./mes'] / len(df.Fecha.unique()), 3)

            df_final = conteo_residuos.join(df_media)
            df_final['Toneladas/contenedor'] = round(df_final['Toneladas med./mes'] / df_final['Nº Contenedores'], 3)
            
            if 'OTROS' in tipo_sel:
                df_final.loc['OTROS'] = ['-', '-', '-', df_media['Toneladas med./mes']['OTROS'], '-']
                
            totales = df_final[df_final.index != 'OTROS'].sum()
            totales['% Selección'] = 100.0
            totales['% Madrid'] = round(totales['% Madrid'], 2)
            totales['Toneladas med./mes'] = df_final['Toneladas med./mes'].sum()
            totales['Toneladas/contenedor'] = round(totales['Toneladas med./mes']/totales['Nº Contenedores'], 3)
            
            df_final.loc['TOTAL'] = totales
            
            df_final
    st.write('---')
# Modelo de regresión------------------------------------------------------------------------------
    st.subheader(":blue[Regression Model Prediction]")
    
    def user_input_features():
        col1, col2, col3, col4 = st.columns(4)    
        
        with col1:
            sorted_unique_nombre = sorted(residuos_juntos.Nombre.unique())
            distrito_seleccionado = st.selectbox('Select district:', sorted_unique_nombre)
        with col2:
            sorted_unique_year = sorted(residuos_juntos.Año.unique())
            sorted_unique_year += [x for x in range(2024, 2026)]
            year_seleccionado = st.selectbox('Select year:', sorted_unique_year)
        with col3:
            sorted_unique_mes = (residuos_juntos.Mes.unique())
            mes_seleccionado = st.selectbox('Select month:', sorted_unique_mes)
        with col4:
            sorted_unique_residuo = sorted(residuos_juntos.Residuo.unique())
            residuo_seleccionado = st.selectbox('Select waste:', sorted_unique_residuo)

    
        data_input = {'Distrito': distrito_seleccionado,
                  'Año': year_seleccionado,
                  'Mes': mes_seleccionado,
                  'Residuo': residuo_seleccionado}

        features = pd.DataFrame(data_input, index=[0])
        return features

    def train_data(features):
        meses = {
            'enero': '01',
            'febrero': '02',
            'marzo': '03',
            'abril': '04',
            'mayo': '05',
            'junio': '06',
            'julio': '07',
            'agosto': '08',
            'septiembre': '09',
            'octubre': '10',
            'noviembre': '11',
            'diciembre': '12'
        }
    
        residuos_juntos['Mes'] = residuos_juntos['Mes'].str.lower().map(meses)
        features['Mes'] = features['Mes'].str.lower().map(meses)

        train = residuos_juntos[(residuos_juntos['Nombre'].isin([features['Distrito'].values[0]])) & (residuos_juntos['Año'] <= features['Año'].values[0]) & (residuos_juntos['Residuo'].isin([features['Residuo'].values[0]])) & (residuos_juntos['Mes'] < features['Mes'].values[0])]

        train['Fecha'] = pd.to_datetime(train['Año'].astype(str) + '-' + train['Mes'] + '-' + '01')
        
        if len(train) == 0:
            st.error('It is not possible to make the prediction for the selected values because we do not have the necessary data. We apologize for the inconvenience.')
        return train.loc[:,['Fecha', 'Año', 'Mes', 'Lote', 'Residuo', 'Toneladas']]
  
        
    def calcula_periodo(features):
        dif_year = (int(features.Año) - 2023) * 12
    
        if features.Mes.eq('enero').any():
            num_mes = 1
        elif features.Mes.eq('febrero').any():
            num_mes = 2
        elif features.Mes.eq('marzo').any():
            num_mes = 3
        elif features.Mes.eq('abril').any():
            num_mes = 4
        elif features.Mes.eq('mayo').any():
            num_mes = 5
        elif features.Mes.eq('junio').any():
            num_mes = 6
        elif features.Mes.eq('julio').any():
            num_mes = 7
        elif features.Mes.eq('agosto').any():
            num_mes = 8
        elif features.Mes.eq('septiembre').any():
            num_mes = 9
        elif features.Mes.eq('octubre').any():
            num_mes = 10
        elif features.Mes.eq('noviembre').any():
            num_mes = 11
        else:
            num_mes = 12
        
        dif_mes = 3 - num_mes
        
        periodo = dif_year + dif_mes
        return periodo
    
            
    def Prediccion(train, periodos):
        if len(train) != 0:
            train = train.set_index('Fecha')
            variables_exogenas = ['Año', 'Mes', 'Lote']
    
            model = auto_arima(
                train.Toneladas,
                exogenous=train[variables_exogenas],
                error_action='ignore',
                suppress_warnings=True,
                trace=False
                )
    
            model = model.fit(train.Toneladas, exogenous=train[variables_exogenas])
        
            a = train.tail(1)
            start_valor = calcula_periodo(a)
    
            return model.predict(start = start_valor, end=periodos)
    
    user_input = user_input_features()
    periodos = calcula_periodo(user_input)
    train = train_data(user_input)
    if len(train) != 0:
        pred = Prediccion(train, periodos)
        st.header('Prediction of Waste')
        st.write(f'The prediction for the year {user_input.Año.values[0]}, month {user_input.Mes.values[0]}, district {user_input.Distrito.values[0]}, and type of waste {user_input.Residuo.values[0]} is **:blue[{round(pd.DataFrame(pred).iloc[-1, 0], 3)}]** tons.')
        
                    
            
if __name__ == "__main__":
    main()
