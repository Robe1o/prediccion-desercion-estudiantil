import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configurar página
st.set_page_config(page_title="Predicción Deserción", layout="wide", page_icon="🎓")
st.title("🎓 Sistema de Predicción de Deserción Estudiantil")

# Cargar modelo
@st.cache_resource
def cargar_modelo():
    try:
        with open('models/modelo_desercion.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error al cargar modelo: {e}")
        return None

modelo_data = cargar_modelo()

# Sidebar
st.sidebar.title("📋 Navegación")
opcion = st.sidebar.radio(
    "Seleccione opción:",
    ["📊 Análisis Exploratorio", "📈 Métricas del Modelo", "🔍 Predecir Deserción (ML)", "👥 Ver Todos Estudiantes"]
)

# Función para calcular categoría (igual que en training.py)
def calcular_categoria(promedio, asistencia, tasa_reprobacion, max_intentos):
    """Calcula categoría de riesgo usando las mismas reglas que training.py"""
    categoria = 'BAJO'
    puntos = 0
    
    # Puntos para categoría DESERTOR
    if promedio < 4.0:
        puntos += 1
    if asistencia < 40:
        puntos += 1
    if tasa_reprobacion > 0.8:
        puntos += 1
    
    if puntos >= 2:
        categoria = 'DESERTOR'
    elif (promedio < 6.0) or (asistencia < 60) or (tasa_reprobacion > 0.5) or (max_intentos >= 2):
        categoria = 'ALTO'
    elif (promedio < 7.5) or (asistencia < 80) or (tasa_reprobacion > 0.2):
        categoria = 'MEDIO'
    
    return categoria, puntos

# Función para cargar datos procesados
@st.cache_data
def cargar_datos():
    try:
        # Intentar cargar estudiantes únicos
        if os.path.exists('models/estudiantes_unicos.csv'):
            return pd.read_csv('models/estudiantes_unicos.csv')
        elif os.path.exists('models/estudiantes_procesados.csv'):
            df = pd.read_csv('models/estudiantes_procesados.csv')
            return df
        else:
            return None
    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None

# Función para formatear DataFrame para display
def formatear_dataframe(df):
    """Formatea números en DataFrame para display"""
    if df is None or len(df) == 0:
        return df
    
    df_display = df.copy()
    
    # Formatear columnas numéricas (evitar errores si ya están formateadas)
    num_cols = {
        'PROMEDIO_GENERAL': ':.2f',
        'PEOR_PROMEDIO': ':.2f',
        'ASISTENCIA_PROMEDIO': ':.1f',
        'TASA_REPROBACION': ':.1%'
    }
    
    for col, fmt in num_cols.items():
        if col in df_display.columns:
            try:
                # Verificar si ya tiene formato de porcentaje
                if col == 'ASISTENCIA_PROMEDIO' and df_display[col].dtype == 'object':
                    if df_display[col].astype(str).str.contains('%').any():
                        continue
                
                # Convertir a numérico si es posible
                df_display[col] = pd.to_numeric(df_display[col], errors='coerce')
                
                if col == 'ASISTENCIA_PROMEDIO':
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.1f}%" if not pd.isna(x) else "N/A")
                elif col == 'TASA_REPROBACION':
                    df_display[col] = df_display[col].apply(lambda x: f"{x:.1%}" if not pd.isna(x) else "N/A")
                else:
                    df_display[col] = df_display[col].apply(lambda x: format(x, fmt) if not pd.isna(x) else "N/A")
            except:
                pass
    
    # Convertir columnas enteras
    int_cols = ['MATERIAS_REPROBADAS', 'MAX_INTENTOS', 'PUNTOS_DESERTOR', 'TOTAL_MATERIAS']
    for col in int_cols:
        if col in df_display.columns:
            try:
                df_display[col] = df_display[col].astype(int)
            except:
                pass
    
    return df_display

if opcion == "📊 Análisis Exploratorio":
    st.header("Análisis Exploratorio de Datos")
    
    estudiantes = cargar_datos()
    
    if estudiantes is not None and len(estudiantes) > 0:
        # Mostrar estadísticas descriptivas
        st.subheader("📌 Resumen General")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Estudiantes", len(estudiantes))
        with col2:
            st.metric("Promedio General", f"{estudiantes['PROMEDIO_GENERAL'].mean():.2f}")
        with col3:
            st.metric("Asistencia Promedio", f"{estudiantes['ASISTENCIA_PROMEDIO'].mean():.1f}%")
        with col4:
            st.metric("Tasa Reprobación", f"{estudiantes['TASA_REPROBACION'].mean()*100:.1f}%")
        
        # Mostrar distribución real
        st.subheader("📋 Distribución por Categoría")
        if 'CATEGORIA' in estudiantes.columns:
            # Calcular distribución
            dist_cat = estudiantes['CATEGORIA'].value_counts()
            
            # Asegurar todas las categorías
            for cat in ['DESERTOR', 'ALTO', 'MEDIO', 'BAJO']:
                if cat not in dist_cat:
                    dist_cat[cat] = 0
            
            dist_cat = dist_cat.reindex(['DESERTOR', 'ALTO', 'MEDIO', 'BAJO'])
            
            dist_df = pd.DataFrame({
                'Categoría': dist_cat.index,
                'Cantidad': dist_cat.values,
                'Porcentaje': (dist_cat.values / len(estudiantes) * 100).round(1)
            })
            st.dataframe(dist_df, width='stretch', hide_index=True)
        
        # Mostrar gráficos guardados
        st.subheader("📊 Visualizaciones")
        
        if os.path.exists('models/analisis_exploratorio.png'):
            st.image('models/analisis_exploratorio.png', 
                    caption="Análisis Exploratorio - Distribuciones por Categoría",
                    width='stretch')
        
        # Estadísticas por categoría
        st.subheader("📊 Estadísticas por Categoría")
        if 'CATEGORIA' in estudiantes.columns:
            for categoria in ['DESERTOR', 'ALTO', 'MEDIO', 'BAJO']:
                if categoria in estudiantes['CATEGORIA'].values:
                    cat_data = estudiantes[estudiantes['CATEGORIA'] == categoria]
                    
                    with st.expander(f"**{categoria}** - {len(cat_data)} estudiantes"):
                        # Mostrar métricas principales
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Promedio", f"{cat_data['PROMEDIO_GENERAL'].mean():.2f}")
                        with col2:
                            st.metric("Asistencia", f"{cat_data['ASISTENCIA_PROMEDIO'].mean():.1f}%")
                        with col3:
                            st.metric("Tasa Reprobación", f"{cat_data['TASA_REPROBACION'].mean()*100:.1f}%")
                        
                        # Mostrar ejemplos
                        st.write(f"**Ejemplos de estudiantes ({min(5, len(cat_data))} de {len(cat_data)}):**")
                        
                        # Seleccionar columnas a mostrar
                        columnas_mostrar = ['ESTUDIANTE', 'PERIODO', 'PROMEDIO_GENERAL', 
                                          'ASISTENCIA_PROMEDIO', 'TASA_REPROBACION', 
                                          'MAX_INTENTOS', 'PUNTOS_DESERTOR']
                        
                        # Filtrar columnas existentes
                        columnas_existentes = [c for c in columnas_mostrar if c in cat_data.columns]
                        
                        muestra = cat_data[columnas_existentes].head(5)
                        muestra_display = formatear_dataframe(muestra)
                        
                        # Mostrar tabla con nombres
                        st.dataframe(muestra_display, width='stretch')
    else:
        st.warning("⚠️ Primero ejecuta `python training.py` para generar los datos")

elif opcion == "📈 Métricas del Modelo":
    st.header("Evaluación del Modelo de Machine Learning")
    
    if modelo_data is not None:
        # Mostrar métricas
        st.subheader("🎯 Métricas de Rendimiento")
        
        metrics = modelo_data['metrics']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{metrics['accuracy']:.4f}", 
                     help="Proporción de predicciones correctas")
        with col2:
            st.metric("Precisión", f"{metrics['precision']:.4f}",
                     help="De los que predice desertor, cuántos realmente lo son")
        with col3:
            st.metric("Recall", f"{metrics['recall']:.4f}",
                     help="De los desertores reales, cuántos detecta")
        with col4:
            st.metric("F1-Score", f"{metrics['f1_score']:.4f}",
                     help="Media armónica de precisión y recall")
        
        # Mostrar matriz de confusión
        st.subheader("🔲 Matriz de Confusión")
        col1, col2 = st.columns([2, 1])
        with col1:
            if os.path.exists('models/matriz_confusion.png'):
                st.image('models/matriz_confusion.png', 
                        caption="Matriz de Confusión del Modelo",
                        width='stretch')
        with col2:
            st.info("""
            **Interpretación:**
            - **Verdaderos Positivos:** Desertores correctamente identificados
            - **Falsos Positivos:** No desertores identificados como desertores
            - **Falsos Negativos:** Desertores no detectados
            - **Verdaderos Negativos:** No desertores correctamente identificados
            """)
        
        # Mostrar importancia de características
        st.subheader("📊 Importancia de Características")
        col1, col2 = st.columns([2, 1])
        with col1:
            if os.path.exists('models/importancia_caracteristicas.png'):
                st.image('models/importancia_caracteristicas.png', 
                        caption="Importancia de Variables para la Predicción",
                        width='stretch')
        with col2:
            if 'features' in modelo_data and 'model' in modelo_data:
                # Obtener las importancias del modelo
                model = modelo_data['model']
                features = modelo_data['features']
                importancias = model.feature_importances_
                
                # Crear DataFrame para ordenar
                importancia_df = pd.DataFrame({
                    'Característica': features,
                    'Importancia': importancias
                }).sort_values('Importancia', ascending=False)
                
                st.info("**Características ordenadas por importancia:**")
                for i, (idx, row) in enumerate(importancia_df.iterrows(), 1):
                    st.write(f"{i}. **{row['Característica']}**: {row['Importancia']:.4f}")
        
        # Detalles del modelo
        st.subheader("⚙️ Detalles del Modelo")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Algoritmo:** Random Forest Classifier")
            st.write(f"**Número de árboles:** 100")
            st.write(f"**Ponderación de clases:** Balanceada")
            st.write(f"**Target:** Deserción Real (0=Continuó, 1=Desertó)")
        with col2:
            st.write(f"**Método de validación:** Train/Test Split (80%/20%)")
            if 'features' in modelo_data:
                st.write(f"**Características utilizadas:** {len(modelo_data['features'])}")
            st.write(f"**Periodos analizados:** Solo CI y CII (ordinarios)")
        
    else:
        st.warning("⚠️ Primero ejecuta `python training.py` para entrenar el modelo")

elif opcion == "🔍 Predecir Deserción (ML)":
    st.header("Predicción de Deserción usando Modelo de ML")
    
    if modelo_data is None:
        st.error("❌ No hay modelo disponible. Ejecuta primero `python training.py`")
        st.stop()
    
    st.write("Ingresa los datos del estudiante para predecir si desertará usando el modelo de Machine Learning:")
    
    with st.form("form_prediccion_ml"):
        col1, col2 = st.columns(2)
        
        with col1:
            promedio = st.slider("Promedio General (0-10)", 0.0, 10.0, 7.0, 0.1,
                               help="Promedio general del estudiante en todas sus materias")
            peor_promedio = st.slider("Peor Promedio de Materia (0-10)", 0.0, 10.0, 5.0, 0.1, 
                                     help="El promedio más bajo que obtuvo en alguna materia")
            asistencia = st.slider("Asistencia Promedio (%)", 0.0, 100.0, 80.0, 1.0,
                                 help="Porcentaje promedio de asistencia a clases")
        
        with col2:
            max_intentos = st.selectbox("Máximo de Intentos por Materia", 
                                      options=[1, 2, 3],
                                      index=0,
                                      help="1=Primera vez, 2=Segunda vez, 3=Tercera vez (máx.)")
            materias_reprobadas = st.number_input("Materias Reprobadas", 0, 50, 2,
                                                help="Número total de materias reprobadas")
            total_materias = st.number_input("Total de Materias Cursadas", 1, 100, 10,
                                           help="Número total de materias tomadas por el estudiante")
        
        submit = st.form_submit_button("🔮 Predecir Deserción con ML", width='content')
        
        if submit:
            # Calcular tasa de reprobación
            tasa_reprobacion = materias_reprobadas / total_materias if total_materias > 0 else 0
            
            # Calcular categoría de riesgo (para usar como feature)
            categoria, puntos = calcular_categoria(promedio, asistencia, tasa_reprobacion, max_intentos)
            
            # Crear array con las features en el orden correcto
            features_order = modelo_data['features']
            
            # Crear diccionario con todos los valores
            features_dict = {
                'PROMEDIO_GENERAL': promedio,
                'PEOR_PROMEDIO': peor_promedio,
                'ASISTENCIA_PROMEDIO': asistencia,
                'MAX_INTENTOS': max_intentos,
                'MATERIAS_REPROBADAS': materias_reprobadas,
                'TASA_REPROBACION': tasa_reprobacion
            }
            
            # Crear DataFrame usando dinámicamente el orden de features del modelo
            # Esto evita errores si la lista de características cambia en el entrenamiento
            row_values = [features_dict[col] for col in features_order]
            features_df = pd.DataFrame([row_values], columns=features_order)
            
            # Hacer predicción
            model = modelo_data['model']
            prediccion = model.predict(features_df)[0]
            probabilidad = model.predict_proba(features_df)[0]
            
            # Mostrar resultado
            st.subheader("🎯 Resultado de la Predicción (ML)")
            
            if prediccion == 1:
                st.error(f"🚨 **PREDICCIÓN: DESERTOR**")
                st.write(f"**Probabilidad de deserción:** {probabilidad[1]*100:.1f}%")
                st.write(f"**Probabilidad de continuar:** {probabilidad[0]*100:.1f}%")
                
                st.warning("""
                **Recomendaciones:**
                - Intervención inmediata requerida
                - Evaluación integral del estudiante
                - Asesoramiento académico y psicológico
                - Contacto con familia/tutores
                """)
            else:
                st.success(f"✅ **PREDICCIÓN: NO DESERTOR**")
                st.write(f"**Probabilidad de continuar:** {probabilidad[0]*100:.1f}%")
                st.write(f"**Probabilidad de deserción:** {probabilidad[1]*100:.1f}%")
                
                # Mostrar nivel de riesgo basado en categoría
                if categoria == 'ALTO':
                    st.warning(f"⚠️ **Categoría de riesgo: {categoria}**")
                    st.info("""
                    **Recomendaciones:**
                    - Seguimiento cercano semanal
                    - Apoyo académico urgente
                    - Tutorías personalizadas
                    """)
                elif categoria == 'MEDIO':
                    st.info(f"📊 **Categoría de riesgo: {categoria}**")
                    st.success("""
                    **Recomendaciones:**
                    - Monitoreo periódico mensual
                    - Apoyo preventivo
                    - Refuerzo en áreas débiles
                    """)
                else:
                    st.success(f"✅ **Categoría de riesgo: {categoria}**")
                    st.success("""
                    **Recomendaciones:**
                    - Seguimiento normal
                    - Motivación para mantener desempeño
                    """)
            
            # Mostrar datos ingresados
            st.subheader("📋 Datos Analizados por el Modelo")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Promedio General", f"{promedio:.2f}/10")
                st.metric("Peor Promedio", f"{peor_promedio:.2f}/10")
                st.metric("Asistencia", f"{asistencia:.1f}%")
            
            with col2:
                st.metric("Máx. Intentos", max_intentos)
                st.metric("Materias Reprobadas", materias_reprobadas)
                st.metric("Tasa Reprobación", f"{tasa_reprobacion*100:.1f}%")
                st.metric("Categoría Riesgo", categoria)
            
            # Mostrar cómo el modelo tomó la decisión
            st.subheader("🧠 Cómo el modelo tomó la decisión")
            
            # Obtener importancia de características
            if hasattr(model, 'feature_importances_'):
                importancia_df = pd.DataFrame({
                    'Característica': features_order,
                    'Importancia': model.feature_importances_
                }).sort_values('Importancia', ascending=False)
                
                st.write("**Características más influyentes:**")
                for idx, row in importancia_df.head(3).iterrows():
                    valor = features_dict[row['Característica']]
                    st.write(f"- **{row['Característica']}**: {valor} (importancia: {row['Importancia']:.3f})")
            
            # Explicar factores de riesgo
            st.subheader("⚠️ Factores de Riesgo Identificados")
            
            factores = []
            if promedio < 6.0:
                factores.append(f"🔴 Promedio general bajo ({promedio:.2f}/10)")
            if peor_promedio < 4.0:
                factores.append(f"🔴 Peor promedio crítico ({peor_promedio:.2f}/10)")
            if asistencia < 70:
                factores.append(f"🔴 Asistencia insuficiente ({asistencia:.1f}%)")
            if max_intentos >= 3:
                factores.append(f"🔴 Máximo de intentos alcanzado ({max_intentos})")
            elif max_intentos >= 2:
                factores.append(f"🟡 Múltiples intentos ({max_intentos})")
            if tasa_reprobacion > 0.3:
                factores.append(f"🔴 Alta tasa de reprobación ({tasa_reprobacion*100:.1f}%)")
            
            if factores:
                for factor in factores:
                    st.write(factor)
            else:
                st.success("✅ No se identificaron factores de riesgo significativos")

else:  # Ver Todos Estudiantes
    st.header("Listado de Estudiantes (Último Periodo)")
    
    estudiantes = cargar_datos()
    
    if estudiantes is not None and len(estudiantes) > 0:
        # Información sobre el dataset
        st.info(f"**Mostrando el último periodo de cada estudiante:** {len(estudiantes)} estudiantes únicos")
        
        # Filtros
        st.subheader("🔍 Filtros de Búsqueda")
        
        col1, col2, col3 = st.columns([2, 2, 2])
        with col1:
            # Inicializar con lista vacía por defecto
            categoria_filtro = []
            if 'CATEGORIA' in estudiantes.columns:
                categoria_filtro = st.multiselect(
                    "Filtrar por categoría:",
                    options=["DESERTOR", "ALTO", "MEDIO", "BAJO"],
                    default=["DESERTOR", "ALTO", "MEDIO", "BAJO"]
                )
        
        with col2:
            orden = st.selectbox("Ordenar por:", 
                               ["Categoría", "Promedio ↓", "Promedio ↑", 
                                "Asistencia ↓", "Asistencia ↑", 
                                "Tasa Reprobación ↓", "Tasa Reprobación ↑",
                                "Estudiante A-Z", "Estudiante Z-A"])
        
        with col3:
            st.write("")
            st.write("")
            mostrar_detalles = st.checkbox("Mostrar detalles completos", value=False)
        
        # Aplicar filtros
        datos_filtrados = estudiantes.copy()
        
        if 'CATEGORIA' in estudiantes.columns and len(categoria_filtro) > 0:
            datos_filtrados = datos_filtrados[datos_filtrados['CATEGORIA'].isin(categoria_filtro)]
        
        # Ordenar
        if orden == "Promedio ↓":
            datos_filtrados = datos_filtrados.sort_values('PROMEDIO_GENERAL', ascending=False)
        elif orden == "Promedio ↑":
            datos_filtrados = datos_filtrados.sort_values('PROMEDIO_GENERAL', ascending=True)
        elif orden == "Asistencia ↓":
            datos_filtrados = datos_filtrados.sort_values('ASISTENCIA_PROMEDIO', ascending=False)
        elif orden == "Asistencia ↑":
            datos_filtrados = datos_filtrados.sort_values('ASISTENCIA_PROMEDIO', ascending=True)
        elif orden == "Tasa Reprobación ↓":
            datos_filtrados = datos_filtrados.sort_values('TASA_REPROBACION', ascending=False)
        elif orden == "Tasa Reprobación ↑":
            datos_filtrados = datos_filtrados.sort_values('TASA_REPROBACION', ascending=True)
        elif orden == "Estudiante A-Z":
            datos_filtrados = datos_filtrados.sort_values('ESTUDIANTE', ascending=True)
        elif orden == "Estudiante Z-A":
            datos_filtrados = datos_filtrados.sort_values('ESTUDIANTE', ascending=False)
        else:
            # Ordenar por categoría (DESERTOR, ALTO, MEDIO, BAJO)
            cat_order = {'DESERTOR': 0, 'ALTO': 1, 'MEDIO': 2, 'BAJO': 3}
            datos_filtrados['_orden'] = datos_filtrados['CATEGORIA'].map(cat_order)
            datos_filtrados = datos_filtrados.sort_values(['_orden', 'ESTUDIANTE'])
            datos_filtrados = datos_filtrados.drop('_orden', axis=1)
        
        # Seleccionar columnas a mostrar
        columnas_base = ['ESTUDIANTE', 'PERIODO', 'CATEGORIA']
        
        if mostrar_detalles:
            columnas_detalle = ['PROMEDIO_GENERAL', 'PEOR_PROMEDIO', 'ASISTENCIA_PROMEDIO', 
                              'TASA_REPROBACION', 'MAX_INTENTOS', 'MATERIAS_REPROBADAS', 
                              'TOTAL_MATERIAS', 'PUNTOS_DESERTOR']
        else:
            columnas_detalle = ['PROMEDIO_GENERAL', 'ASISTENCIA_PROMEDIO', 
                              'TASA_REPROBACION', 'MAX_INTENTOS']
        
        columnas_mostrar = [c for c in columnas_base + columnas_detalle if c in datos_filtrados.columns]
        
        # Mostrar tabla
        st.subheader(f"📋 Estudiantes ({len(datos_filtrados)} de {len(estudiantes)} total)")
        
        datos_display = formatear_dataframe(datos_filtrados[columnas_mostrar])
        
        # MOSTRAR TODOS LOS DATOS - SIN LIMITACIÓN
        st.dataframe(
            datos_display,
            height=600,
            width='stretch'
        )
        
        # Contador de registros mostrados
        st.caption(f"📊 Mostrando {len(datos_display)} estudiantes")
        
        # Estadísticas rápidas
        if len(datos_filtrados) > 0:
            st.subheader("📊 Resumen de Estudiantes Filtrados")
            
            total = len(datos_filtrados)
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Estudiantes", total)
            with col2:
                prom = datos_filtrados['PROMEDIO_GENERAL'].mean()
                st.metric("Promedio", f"{prom:.2f}")
            with col3:
                asist = datos_filtrados['ASISTENCIA_PROMEDIO'].mean()
                st.metric("Asistencia", f"{asist:.1f}%")
            with col4:
                tasa = datos_filtrados['TASA_REPROBACION'].mean()
                st.metric("Tasa Reprobación", f"{tasa*100:.1f}%")
            
            # Distribución por categoría
            if 'CATEGORIA' in datos_filtrados.columns:
                st.write("**Distribución por categoría:**")
                cat_dist = datos_filtrados['CATEGORIA'].value_counts()
                for cat in ['DESERTOR', 'ALTO', 'MEDIO', 'BAJO']:
                    if cat in cat_dist.index:
                        count = cat_dist[cat]
                        st.write(f"- {cat}: {count} ({count/total*100:.1f}%)")
            
            # Exportar datos
            st.subheader("📥 Exportar Datos")
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV
                csv = datos_filtrados[columnas_mostrar].to_csv(index=False, float_format='%.2f')
                st.download_button(
                    label="📄 Descargar CSV",
                    data=csv,
                    file_name="estudiantes_filtrados.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel
                import io
                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    datos_filtrados[columnas_mostrar].to_excel(writer, index=False, sheet_name='Estudiantes')
                
                st.download_button(
                    label="📊 Descargar Excel",
                    data=buffer.getvalue(),
                    file_name="estudiantes_filtrados.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        else:
            st.info("No hay estudiantes que coincidan con los filtros aplicados")
    else:
        st.warning("⚠️ Primero ejecuta `python training.py` para generar los datos")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Acerca del Proyecto")
st.sidebar.info("""
**Proyecto Final:** Almacenes y Minería de Datos  
**Realizado por:** Roberto Alvarez  
**Nivel:** 5to Semestre  
**Carrera:** Ciencia de Datos e IA
**Estudiante:** 201  
**Fecha:** Febrero 2026
""")

st.sidebar.markdown("### ℹ️ Información Técnica")
st.sidebar.success("""
**Modelo:** Random Forest Classifier  
**Variables:** 6 características  
**Predicción:** Deserción (Sí/No)  
**Categorías:** 4 niveles de riesgo  
**Periodos:** Solo CI y CII (ordinarios)  
**Herramientas:** Python, Scikit-learn, Streamlit
""")