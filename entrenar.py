import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Crear directorios
os.makedirs('models', exist_ok=True)
os.makedirs('data', exist_ok=True)

def es_periodo_ordinario(periodo):
    """Determina si un periodo es ordinario (CI o CII)"""
    if pd.isna(periodo):
        return False
    periodo_str = str(periodo).upper()
    return ('CI' in periodo_str or 'CII' in periodo_str) and 'ING' not in periodo_str

def cargar_datos():
    """Carga el archivo Excel de data/"""
    try:
        archivos = [f for f in os.listdir('data') if f.endswith('.xlsx')]
        if not archivos:
            print("ERROR: Coloca tu archivo Excel en la carpeta 'data/'")
            return None
        
        df = pd.read_excel(f'data/{archivos[0]}')
        print(f"Datos cargados: {len(df)} registros, {len(df.columns)} columnas")
        return df
    except Exception as e:
        print(f"Error: {e}")
        return None

def detectar_desercion_real(df):
    """
    Detecta deserción REAL solo para períodos ordinarios (CI/CII).
    """
    print("\n=== DETECCIÓN DE DESERCIÓN REAL (SOLO PERÍODOS ORDINARIOS) ===")
    
    # Filtrar solo períodos ordinarios
    df_ordinarios = df[df['PERIODO'].apply(es_periodo_ordinario)].copy()
    if len(df_ordinarios) == 0:
        print("⚠️ No hay períodos ordinarios (CI/CII) en los datos")
        return None
    
    # Función para parsear y ordenar periodos correctamente
    def parsear_periodo(periodo):
        if pd.isna(periodo):
            return (0, 0, 99)
        partes = str(periodo).split()
        if len(partes) < 4:
            return (0, 0, 99)
        año_inicio = int(partes[0])
        año_fin = int(partes[2])
        ciclo = partes[3] if len(partes) > 3 else ""
        orden_map = {'CI': 1, 'CII': 2}
        orden = orden_map.get(ciclo, 99)
        return (año_inicio, año_fin, orden)
    
    # Ordenar periodos cronológicamente (solo ordinarios)
    periodos_unicos = sorted(df_ordinarios['PERIODO'].unique(), key=parsear_periodo)
    
    print(f"📅 Periodos ordinarios encontrados: {len(periodos_unicos)}")
    for i, p in enumerate(periodos_unicos, 1):
        print(f"   {i}. {p}")
    
    # Para almacenar resultados
    desercion_data = []
    
    # Para cada estudiante, determinar si desertó
    estudiantes_unicos = df_ordinarios['ESTUDIANTE'].unique()
    print(f"\n📊 Analizando {len(estudiantes_unicos)} estudiantes únicos...")
    
    # Para cada estudiante, encontrar todos sus periodos
    for estudiante in estudiantes_unicos:
        periodos_estudiante = df_ordinarios[df_ordinarios['ESTUDIANTE'] == estudiante]['PERIODO'].unique()
        periodos_estudiante_ordenados = sorted(periodos_estudiante, key=parsear_periodo)
        
        # Determinar deserción para cada periodo del estudiante
        for i, periodo_actual in enumerate(periodos_estudiante_ordenados):
            # Si este NO es el último periodo del estudiante
            if i < len(periodos_estudiante_ordenados) - 1:
                # No es desertor en este periodo (aparece en un periodo posterior)
                desertor = 0
            else:
                # Es el último periodo del estudiante
                # Verificar si es el último periodo general
                if periodo_actual == periodos_unicos[-1]:
                    # Es el último periodo general, no se puede determinar deserción
                    desertor = 0
                else:
                    # Es desertor porque no aparece en periodos posteriores
                    desertor = 1
            
            desercion_data.append({
                'ESTUDIANTE': estudiante,
                'PERIODO': periodo_actual,
                'DESERTOR_REAL': desertor
            })
    
    # Crear DataFrame
    df_desercion = pd.DataFrame(desercion_data)
    
    # Calcular estadísticas por estudiante único
    estudiantes_desercion = df_desercion.groupby('ESTUDIANTE')['DESERTOR_REAL'].max()
    total_estudiantes = len(estudiantes_desercion)
    desertores_unicos = estudiantes_desercion.sum()
    continuaron_unicos = total_estudiantes - desertores_unicos
    
    print(f"\n" + "="*60)
    print(f"📊 RESULTADOS POR ESTUDIANTE ÚNICO")
    print(f"="*60)
    print(f"Total estudiantes únicos: {total_estudiantes}")
    print(f"Estudiantes que continuaron: {continuaron_unicos} ({continuaron_unicos/total_estudiantes*100:.1f}%)")
    print(f"Estudiantes que desertaron: {desertores_unicos} ({desertores_unicos/total_estudiantes*100:.1f}%)")
    
    return df_desercion

def crear_variables(df, periodo=None, ultimo_periodo=False):
    """
    Crea variables por estudiante excluyendo materias especiales.
    
    Args:
        df: DataFrame con datos
        periodo: Si se especifica, filtra solo ese periodo
        ultimo_periodo: Si es True, toma solo el último periodo de cada estudiante
    """
    # Si se pide el último periodo, procesar cada estudiante por separado
    if ultimo_periodo and periodo is None:
        # Función para parsear periodo
        def parsear_periodo(periodo_str):
            if pd.isna(periodo_str):
                return (0, 0, 99)
            partes = str(periodo_str).split()
            if len(partes) < 4:
                return (0, 0, 99)
            año_inicio = int(partes[0])
            año_fin = int(partes[2])
            ciclo = partes[3] if len(partes) > 3 else ""
            orden_map = {'CI': 1, 'CII': 2}
            orden = orden_map.get(ciclo, 99)
            return (año_inicio, año_fin, orden)
        
        estudiantes_lista = []
        
        # Para cada estudiante, obtener su último periodo
        for estudiante in df['ESTUDIANTE'].unique():
            df_estudiante = df[df['ESTUDIANTE'] == estudiante].copy()
            periodos_estudiante = df_estudiante['PERIODO'].unique()
            
            if len(periodos_estudiante) > 0:
                # Encontrar el último periodo
                ultimo_periodo_est = max(periodos_estudiante, key=parsear_periodo)
                df_ultimo = df_estudiante[df_estudiante['PERIODO'] == ultimo_periodo_est]
                
                # Crear variables para este periodo
                vars_est = crear_variables_single(df_ultimo, estudiante, ultimo_periodo_est)
                if vars_est is not None:
                    estudiantes_lista.append(vars_est)
        
        if estudiantes_lista:
            return pd.concat(estudiantes_lista, ignore_index=True)
        else:
            return pd.DataFrame()
    
    # Filtrar por periodo si se especifica
    if periodo is not None:
        df = df[df['PERIODO'] == periodo].copy()
        if len(df) == 0:
            print(f"⚠️ No hay datos para el periodo {periodo}")
            return pd.DataFrame()
    
    # Si no es último periodo y no hay periodo específico, procesar todos los datos juntos
    if periodo is None and not ultimo_periodo:
        # Agrupar por estudiante y periodo
        estudiantes_lista = []
        for (estudiante, periodo_est), grupo in df.groupby(['ESTUDIANTE', 'PERIODO']):
            vars_est = crear_variables_single(grupo, estudiante, periodo_est)
            if vars_est is not None:
                estudiantes_lista.append(vars_est)
        
        if estudiantes_lista:
            return pd.concat(estudiantes_lista, ignore_index=True)
        else:
            return pd.DataFrame()
    
    # Para un solo periodo específico
    return crear_variables_batch(df)

def crear_variables_single(df_estudiante, estudiante, periodo):
    """Crea variables para un solo estudiante en un periodo específico"""
    if len(df_estudiante) == 0:
        return None
    
    # Convertir tipos
    df_estudiante['PROMEDIO'] = pd.to_numeric(df_estudiante['PROMEDIO'].astype(str).str.replace(',', '.'), errors='coerce')
    df_estudiante['ASISTENCIA'] = pd.to_numeric(df_estudiante['ASISTENCIA'], errors='coerce')
    
    # Identificar materias especiales para EXCLUIR
    grupos_especiales = ['MOVILIDAD', 'CONVALIDACION DE CERTIFICADOS DE IDIOMAS', 'HOMOLOGACION']
    palabras_especiales = ['TRABAJO DE TITULACION', 'PRACTICA', 'PROYECTO DE TITULACION']
    
    # Crear máscara para materias VÁLIDAS
    mascara_valida = ~df_estudiante['GRUPO/PARALELO'].isin(grupos_especiales)
    
    if 'MATERIA' in df_estudiante.columns:
        for palabra in palabras_especiales:
            mascara_valida &= ~df_estudiante['MATERIA'].str.upper().str.contains(palabra, na=False)
    
    mascara_valida &= ~((df_estudiante['ESTADO'] == 'APROBADA') & (df_estudiante['ASISTENCIA'] == 0))
    
    # Filtrar DataFrame
    df_valido = df_estudiante[mascara_valida].copy()
    
    if len(df_valido) == 0:
        return None
    
    # Calcular métricas
    promedio_general = df_valido['PROMEDIO'].mean()
    peor_promedio = df_valido['PROMEDIO'].min()
    asistencia_promedio = df_valido['ASISTENCIA'].mean()
    max_intentos = df_valido['NO. VEZ'].max()
    materias_reprobadas = (df_valido['ESTADO'] == 'REPROBADA').sum()
    total_materias = len(df_valido)
    
    # Calcular tasa de reprobación
    tasa_reprobacion = materias_reprobadas / total_materias if total_materias > 0 else 0
    
    # LIMITAR MAX_INTENTOS a 3
    max_intentos = min(max_intentos, 3)
    
    # Limitar asistencia
    asistencia_promedio = max(0, min(asistencia_promedio, 100))
    
    # CATEGORÍAS DE RIESGO
    categoria = 'BAJO'
    puntos = 0
    
    # Puntos para categoría DESERTOR
    if promedio_general < 4.0:
        puntos += 1
    if asistencia_promedio < 40:
        puntos += 1
    if tasa_reprobacion > 0.8:
        puntos += 1
    
    if puntos >= 2:
        categoria = 'DESERTOR'
    elif (promedio_general < 6.0) or (asistencia_promedio < 60) or (tasa_reprobacion > 0.5) or (max_intentos >= 2):
        categoria = 'ALTO'
    elif (promedio_general < 7.5) or (asistencia_promedio < 80) or (tasa_reprobacion > 0.2):
        categoria = 'MEDIO'
    
    # Crear diccionario con los datos
    estudiante_data = {
        'ESTUDIANTE': estudiante,
        'PERIODO': periodo,
        'PROMEDIO_GENERAL': promedio_general,
        'PEOR_PROMEDIO': peor_promedio,
        'ASISTENCIA_PROMEDIO': asistencia_promedio,
        'MAX_INTENTOS': max_intentos,
        'MATERIAS_REPROBADAS': materias_reprobadas,
        'TOTAL_MATERIAS': total_materias,
        'TASA_REPROBACION': tasa_reprobacion,
        'PUNTOS_DESERTOR': puntos,
        'CATEGORIA': categoria
    }
    
    return pd.DataFrame([estudiante_data])

def crear_variables_batch(df):
    """Crea variables para un batch de estudiantes (para entrenamiento)"""
    # Convertir tipos
    df['PROMEDIO'] = pd.to_numeric(df['PROMEDIO'].astype(str).str.replace(',', '.'), errors='coerce')
    df['ASISTENCIA'] = pd.to_numeric(df['ASISTENCIA'], errors='coerce')
    
    # Identificar materias especiales para EXCLUIR
    grupos_especiales = ['MOVILIDAD', 'CONVALIDACION DE CERTIFICADOS DE IDIOMAS', 'HOMOLOGACION']
    palabras_especiales = ['TRABAJO DE TITULACION', 'PRACTICA', 'PROYECTO DE TITULACION']
    
    # Crear máscara para materias VÁLIDAS
    mascara_valida = ~df['GRUPO/PARALELO'].isin(grupos_especiales)
    
    if 'MATERIA' in df.columns:
        for palabra in palabras_especiales:
            mascara_valida &= ~df['MATERIA'].str.upper().str.contains(palabra, na=False)
    
    mascara_valida &= ~((df['ESTADO'] == 'APROBADA') & (df['ASISTENCIA'] == 0))
    
    # Filtrar DataFrame
    df_valido = df[mascara_valida].copy()
    
    if len(df_valido) == 0:
        return pd.DataFrame()
    
    # Calcular métricas por estudiante y periodo
    estudiantes_agg = df_valido.groupby(['ESTUDIANTE', 'PERIODO']).agg({
        'PROMEDIO': ['mean', 'min'],
        'ASISTENCIA': 'mean',
        'NO. VEZ': 'max',
        'ESTADO': lambda x: (x == 'REPROBADA').sum(),
        'COD_MATERIA': 'count'
    })
    
    # Aplanar MultiIndex
    estudiantes = pd.DataFrame()
    estudiantes['PROMEDIO_GENERAL'] = estudiantes_agg[('PROMEDIO', 'mean')]
    estudiantes['PEOR_PROMEDIO'] = estudiantes_agg[('PROMEDIO', 'min')]
    estudiantes['ASISTENCIA_PROMEDIO'] = estudiantes_agg[('ASISTENCIA', 'mean')]
    estudiantes['MAX_INTENTOS'] = estudiantes_agg[('NO. VEZ', 'max')]
    estudiantes['MATERIAS_REPROBADAS'] = estudiantes_agg[('ESTADO', '<lambda>')]
    estudiantes['TOTAL_MATERIAS'] = estudiantes_agg[('COD_MATERIA', 'count')]
    
    # Resetear índice para tener ESTUDIANTE y PERIODO como columnas
    estudiantes = estudiantes.reset_index()
    
    # Rellenar NaN
    estudiantes = estudiantes.fillna(0)
    
    # LIMITAR MAX_INTENTOS a 3
    estudiantes['MAX_INTENTOS'] = estudiantes['MAX_INTENTOS'].clip(upper=3)
    
    # Limitar asistencia
    estudiantes['ASISTENCIA_PROMEDIO'] = estudiantes['ASISTENCIA_PROMEDIO'].clip(0, 100)
    
    # Calcular tasa de reprobación
    estudiantes['TASA_REPROBACION'] = estudiantes['MATERIAS_REPROBADAS'] / estudiantes['TOTAL_MATERIAS']
    estudiantes['TASA_REPROBACION'] = estudiantes['TASA_REPROBACION'].fillna(0)
    
    # CATEGORÍAS DE RIESGO
    estudiantes['CATEGORIA'] = 'BAJO'
    
    # SISTEMA DE PUNTOS SOLO PARA DESERTOR
    estudiantes['PUNTOS_DESERTOR'] = 0
    estudiantes.loc[estudiantes['PROMEDIO_GENERAL'] < 4.0, 'PUNTOS_DESERTOR'] += 1
    estudiantes.loc[estudiantes['ASISTENCIA_PROMEDIO'] < 40, 'PUNTOS_DESERTOR'] += 1
    estudiantes.loc[estudiantes['TASA_REPROBACION'] > 0.8, 'PUNTOS_DESERTOR'] += 1
    
    # DESERTOR: Necesita al menos 2 de 3 condiciones graves
    estudiantes.loc[estudiantes['PUNTOS_DESERTOR'] >= 2, 'CATEGORIA'] = 'DESERTOR'
    
    # ALTO RIESGO
    cond_alto = (
        (estudiantes['CATEGORIA'] == 'BAJO') &
        ((estudiantes['PROMEDIO_GENERAL'] < 6.0) |
         (estudiantes['ASISTENCIA_PROMEDIO'] < 60) |
         (estudiantes['TASA_REPROBACION'] > 0.5) |
         (estudiantes['MAX_INTENTOS'] >= 2))
    )
    estudiantes.loc[cond_alto, 'CATEGORIA'] = 'ALTO'
    
    # MEDIO RIESGO
    cond_medio = (
        (estudiantes['CATEGORIA'] == 'BAJO') &
        ((estudiantes['PROMEDIO_GENERAL'] < 7.5) |
         (estudiantes['ASISTENCIA_PROMEDIO'] < 80) |
         (estudiantes['TASA_REPROBACION'] > 0.2))
    )
    estudiantes.loc[cond_medio, 'CATEGORIA'] = 'MEDIO'
    
    return estudiantes

def analisis_exploratorio(estudiantes):
    """Genera gráficos de análisis exploratorio"""
    print("\n=== ANÁLISIS EXPLORATORIO ===")
    
    # Crear figura
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Distribución de categorías (usando datos reales)
    cat_counts = estudiantes['CATEGORIA'].value_counts()
    
    # Asegurar que todas las categorías existan (incluso si tienen 0)
    for cat in ['DESERTOR', 'ALTO', 'MEDIO', 'BAJO']:
        if cat not in cat_counts:
            cat_counts[cat] = 0
    
    # Ordenar las categorías
    cat_counts = cat_counts.reindex(['DESERTOR', 'ALTO', 'MEDIO', 'BAJO'])
    
    print(f"Total estudiantes en datos: {len(estudiantes)}")
    
    colors_dict = {'DESERTOR': 'red', 'ALTO': 'orange', 'MEDIO': 'yellow', 'BAJO': 'green'}
    colors = [colors_dict.get(cat, 'gray') for cat in cat_counts.index]
    axes[0, 0].bar(cat_counts.index, cat_counts.values, color=colors)
    axes[0, 0].set_title('Distribución de Categorías', fontweight='bold')
    axes[0, 0].set_ylabel('Cantidad')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Añadir etiquetas con valores
    for i, v in enumerate(cat_counts.values):
        axes[0, 0].text(i, v + 5, str(v), ha='center', fontweight='bold')
    
    # 2. Promedio por categoría
    promedio_cat = estudiantes.groupby('CATEGORIA')['PROMEDIO_GENERAL'].mean()
    # Asegurar todas las categorías
    for cat in ['DESERTOR', 'ALTO', 'MEDIO', 'BAJO']:
        if cat not in promedio_cat:
            promedio_cat[cat] = 0
    promedio_cat = promedio_cat.reindex(['DESERTOR', 'ALTO', 'MEDIO', 'BAJO'])
    
    axes[0, 1].bar(promedio_cat.index, promedio_cat.values, color='skyblue')
    axes[0, 1].set_title('Promedio General por Categoría', fontweight='bold')
    axes[0, 1].set_ylabel('Promedio')
    axes[0, 1].axhline(y=6, color='red', linestyle='--', alpha=0.5)
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # 3. Asistencia por categoría
    asistencia_cat = estudiantes.groupby('CATEGORIA')['ASISTENCIA_PROMEDIO'].mean()
    for cat in ['DESERTOR', 'ALTO', 'MEDIO', 'BAJO']:
        if cat not in asistencia_cat:
            asistencia_cat[cat] = 0
    asistencia_cat = asistencia_cat.reindex(['DESERTOR', 'ALTO', 'MEDIO', 'BAJO'])
    
    axes[0, 2].bar(asistencia_cat.index, asistencia_cat.values, color='lightcoral')
    axes[0, 2].set_title('Asistencia Promedio por Categoría', fontweight='bold')
    axes[0, 2].set_ylabel('Asistencia (%)')
    axes[0, 2].axhline(y=70, color='red', linestyle='--', alpha=0.5)
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # 4. Tasa de reprobación por categoría
    reprobacion_cat = estudiantes.groupby('CATEGORIA')['TASA_REPROBACION'].mean()
    for cat in ['DESERTOR', 'ALTO', 'MEDIO', 'BAJO']:
        if cat not in reprobacion_cat:
            reprobacion_cat[cat] = 0
    reprobacion_cat = reprobacion_cat.reindex(['DESERTOR', 'ALTO', 'MEDIO', 'BAJO'])
    
    axes[1, 0].bar(reprobacion_cat.index, reprobacion_cat.values * 100, color='lightgreen')
    axes[1, 0].set_title('Tasa de Reprobación por Categoría', fontweight='bold')
    axes[1, 0].set_ylabel('Tasa (%)')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # 5. Intentos máximos por categoría
    intentos_cat = estudiantes.groupby('CATEGORIA')['MAX_INTENTOS'].mean()
    for cat in ['DESERTOR', 'ALTO', 'MEDIO', 'BAJO']:
        if cat not in intentos_cat:
            intentos_cat[cat] = 0
    intentos_cat = intentos_cat.reindex(['DESERTOR', 'ALTO', 'MEDIO', 'BAJO'])
    
    axes[1, 1].bar(intentos_cat.index, intentos_cat.values, color='gold')
    axes[1, 1].set_title('Intentos Máximos por Categoría', fontweight='bold')
    axes[1, 1].set_ylabel('Intentos')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # 6. Correlación promedio vs asistencia
    cat_to_num = {'DESERTOR': 0, 'ALTO': 1, 'MEDIO': 2, 'BAJO': 3}
    estudiantes['CATEGORIA_NUM'] = estudiantes['CATEGORIA'].map(cat_to_num)
    scatter = axes[1, 2].scatter(estudiantes['PROMEDIO_GENERAL'], 
                                estudiantes['ASISTENCIA_PROMEDIO'],
                                c=estudiantes['CATEGORIA_NUM'],
                                cmap='RdYlGn', alpha=0.6, s=50)
    axes[1, 2].set_title('Promedio vs Asistencia', fontweight='bold')
    axes[1, 2].set_xlabel('Promedio General')
    axes[1, 2].set_ylabel('Asistencia (%)')
    axes[1, 2].grid(alpha=0.3)
    cbar = plt.colorbar(scatter, ax=axes[1, 2])
    cbar.set_label('Riesgo (0=Desertor, 3=Bajo)')
    
    plt.tight_layout()
    plt.savefig('models/analisis_exploratorio.png', dpi=300, bbox_inches='tight')
    print("✓ Gráficos guardados en 'models/analisis_exploratorio.png'")
    
    # Estadísticas descriptivas
    print("\n--- Estadísticas Descriptivas ---")
    cols_stats = ['PROMEDIO_GENERAL', 'PEOR_PROMEDIO', 'ASISTENCIA_PROMEDIO', 
                  'MATERIAS_REPROBADAS', 'TASA_REPROBACION', 'MAX_INTENTOS']
    stats = estudiantes[cols_stats].describe()
    print(stats.round(2))
    
    # Distribución de categorías (real)
    print("\n" + "="*60)
    print("📊 DISTRIBUCIÓN DE CATEGORÍAS (ESTUDIANTES ÚNICOS)")
    print("="*60)
    
    total_estudiantes = len(estudiantes)
    
    for cat in ['DESERTOR', 'ALTO', 'MEDIO', 'BAJO']:
        count = cat_counts[cat]
        print(f"{cat}: {count} ({count/total_estudiantes*100:.1f}%)")
    
    print(f"\nTotal estudiantes únicos: {total_estudiantes}")

def entrenar_modelo(df, df_desercion):
    """
    Entrena modelo usando DESERCIÓN REAL como target.
    """
    print("\n=== ENTRENAMIENTO DEL MODELO CON DESERCIÓN REAL ===")
    
    # Filtrar solo períodos ordinarios
    df_ordinarios = df[df['PERIODO'].apply(es_periodo_ordinario)].copy()
    
    # Crear variables para entrenamiento (todos los periodos excepto el último)
    periodos_ordinarios = sorted(df_ordinarios['PERIODO'].unique(), 
                                 key=lambda x: (int(x.split()[0]), 1 if 'CI' in x else 2))
    
    # Excluir el último periodo (no se puede verificar deserción)
    if len(periodos_ordinarios) > 1:
        periodos_entrenamiento = periodos_ordinarios[:-1]
    else:
        periodos_entrenamiento = periodos_ordinarios
    
    print(f"\n📊 Procesando {len(periodos_entrenamiento)} periodos ordinarios para entrenamiento...")
    
    # Crear variables usando batch para entrenamiento
    df_completo = crear_variables_batch(df_ordinarios[df_ordinarios['PERIODO'].isin(periodos_entrenamiento)])
    
    if len(df_completo) == 0:
        print("❌ No hay datos para entrenamiento")
        return None
    
    print(f"✓ Variables creadas: {len(df_completo)} estudiantes-periodo")
    
    # Merge con deserción real
    df_modelo = df_completo.merge(
        df_desercion[['ESTUDIANTE', 'PERIODO', 'DESERTOR_REAL']],
        on=['ESTUDIANTE', 'PERIODO'],
        how='inner'
    )
    
    print(f"✓ Datos para entrenamiento: {len(df_modelo)} registros")
    print(f"   Desertores: {df_modelo['DESERTOR_REAL'].sum()} ({df_modelo['DESERTOR_REAL'].mean()*100:.1f}%)")
    print(f"   Continuaron: {(1-df_modelo['DESERTOR_REAL']).sum()} ({(1-df_modelo['DESERTOR_REAL']).mean()*100:.1f}%)")
    
    if len(df_modelo) == 0:
        print("\n❌ ERROR: No hay datos para entrenar.")
        return None
    
    # Features
    cat_map = {'BAJO': 0, 'MEDIO': 1, 'ALTO': 2, 'DESERTOR': 3}
    df_modelo['CATEGORIA_NUM'] = df_modelo['CATEGORIA'].map(cat_map)
    
    features = [
        'PROMEDIO_GENERAL', 'PEOR_PROMEDIO',
        'ASISTENCIA_PROMEDIO', 'MAX_INTENTOS', 
        'MATERIAS_REPROBADAS', 'TASA_REPROBACION',
        'CATEGORIA_NUM'
    ]
    
    X = df_modelo[features]
    y = df_modelo['DESERTOR_REAL']
    
    # Verificar distribución de clases
    print(f"\n📊 Distribución de clases:")
    print(f"   Clase 0 (Continuó): {(y==0).sum()}")
    print(f"   Clase 1 (Desertó): {(y==1).sum()}")
    
    # Dividir datos
    min_samples = y.value_counts().min()
    
    if min_samples >= 2:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"✓ Split con stratify")
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"⚠️ Split SIN stratify (clase pequeña: {min_samples} muestras)")
    
    print(f"\nTrain: {len(X_train)} | Test: {len(X_test)}")
    
    # Entrenar modelo
    print(f"\n🌳 Entrenando Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        class_weight='balanced',
        max_depth=10,
        min_samples_split=5
    )
    model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    print(f"\n📊 MÉTRICAS DEL MODELO:")
    print(f"✓ Accuracy: {accuracy:.4f}")
    print(f"✓ Precisión: {precision:.4f}")
    print(f"✓ Recall: {recall:.4f}")
    print(f"✓ F1-Score: {f1:.4f}")
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Continuó', 'Desertó'],
                yticklabels=['Continuó', 'Desertó'])
    plt.title('Matriz de Confusión - Deserción Real', fontweight='bold', fontsize=14)
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    plt.savefig('models/matriz_confusion.png', dpi=300, bbox_inches='tight')
    print("✓ Matriz de confusión guardada")
    
    # Reporte de clasificación
    print("\n--- Reporte de Clasificación ---")
    print(classification_report(y_test, y_pred, target_names=['Continuó', 'Desertó'], zero_division=0))
    
    # Importancia de características
    importancia = pd.DataFrame({
        'Característica': features,
        'Importancia': model.feature_importances_
    }).sort_values('Importancia', ascending=False)
    
    print("\n--- Importancia de Características ---")
    print(importancia.to_string(index=False))
    
    plt.figure(figsize=(10, 6))
    plt.barh(importancia['Característica'], importancia['Importancia'], color='steelblue')
    plt.title('Importancia de Características - Predicción de Deserción Real', fontweight='bold', fontsize=14)
    plt.xlabel('Importancia')
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('models/importancia_caracteristicas.png', dpi=300, bbox_inches='tight')
    print("✓ Importancia de características guardada")
    
    # Guardar modelo
    modelo_data = {
        'model': model,
        'features': features,
        'cat_map': cat_map,
        'feature_names': features,
        'metrics': {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'df_completo': df_modelo,
        'target': 'DESERTOR_REAL',
        'note': 'Modelo entrenado solo con períodos ordinarios (CI/CII)'
    }
    
    with open('models/modelo_desercion.pkl', 'wb') as f:
        pickle.dump(modelo_data, f)
    
    print("✓ Modelo guardado en 'models/modelo_desercion.pkl'")
    
    # Guardar datos procesados (sin DESERTOR_REAL para la app)
    df_para_app = df_modelo.drop('DESERTOR_REAL', axis=1, errors='ignore')
    df_para_app.to_csv('models/estudiantes_procesados.csv', float_format='%.2f', index=False)
    print("✓ Datos procesados guardados")
    
    return modelo_data

def main():
    """Función principal"""
    print("=" * 70)
    print("ENTRENAMIENTO DEL MODELO DE PREDICCIÓN DE DESERCIÓN")
    print("=" * 70)
    
    # Cargar datos
    df = cargar_datos()
    if df is None:
        return
    
    # Detectar deserción REAL (solo períodos ordinarios)
    df_desercion = detectar_desercion_real(df)
    
    if df_desercion is None or len(df_desercion) == 0:
        print("\n❌ No se pudo detectar deserción.")
        return
    
    # Crear variables agregadas (último periodo de cada estudiante para análisis)
    print("\n--- Creando variables para análisis exploratorio (último periodo) ---")
    
    # Filtrar solo períodos ordinarios
    df_ordinarios = df[df['PERIODO'].apply(es_periodo_ordinario)].copy()
    
    # Tomar solo el último periodo de cada estudiante
    estudiantes_global = crear_variables(df_ordinarios, periodo=None, ultimo_periodo=True)
    
    if len(estudiantes_global) == 0:
        print("❌ Error al crear variables")
        return
    
    print(f"✓ Estudiantes únicos (último periodo): {len(estudiantes_global)}")
    
    # Guardar estudiantes únicos para la app
    estudiantes_global.to_csv('models/estudiantes_unicos.csv', float_format='%.2f', index=False)
    print("✓ Datos de estudiantes únicos guardados en 'models/estudiantes_unicos.csv'")
    
    # Análisis exploratorio
    analisis_exploratorio(estudiantes_global)
    
    # Entrenar modelo CON DESERCIÓN REAL
    modelo_data = entrenar_modelo(df, df_desercion)
    
    if modelo_data is None:
        print("\n❌ Error al entrenar modelo")
        return
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("-" * 70)
    
    print("\n📊 Distribución de categorías de riesgo (estudiantes únicos):")
    cat_dist = estudiantes_global['CATEGORIA'].value_counts()
    total_estudiantes = len(estudiantes_global)
    
    for cat in ['DESERTOR', 'ALTO', 'MEDIO', 'BAJO']:
        if cat in cat_dist.index:
            count = cat_dist[cat]
            print(f"{cat}: {count} ({count/total_estudiantes*100:.1f}%)")
        else:
            print(f"{cat}: 0 (0.0%)")
    
    print(f"\nTotal estudiantes únicos: {total_estudiantes}")

if __name__ == "__main__":
    main()