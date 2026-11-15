import streamlit as st
import pandas as pd
import numpy as np
import requests
import urllib.parse
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw
import plotly.express as px

# ==============================================================================
# CONFIGURACIÓN DE PÁGINA Y ESTILOS UI/UX
# ==============================================================================
st.set_page_config(
    page_title="QSPR Explorer | ESOL Delaney & RDKit",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        padding: 12px 16px;
        border-radius: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .iupac-box {
        background-color: #f0fdf4;
        border-left: 4px solid #16a34a;
        padding: 12px;
        border-radius: 6px;
        font-family: monospace;
        color: #166534;
        margin-bottom: 10px;
    }
    .smiles-box {
        background-color: #eff6ff;
        border-left: 4px solid #2563eb;
        padding: 12px;
        border-radius: 6px;
        font-family: monospace;
        color: #1e40af;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# FUNCIONES DE CARGA DE DATOS, API IUPAC Y CÁLCULO RDKIT (PUNTOS 6 AL 9B)
# ==============================================================================
URL_DELANEY = "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/delaney-processed.csv"

@st.cache_data(show_spinner=False)
def cargar_dataset_delaney():
    """Sección 6: Carga del dataset real ESOL/Delaney desde DeepChem."""
    df = pd.read_csv(URL_DELANEY)
    return df

@st.cache_data(show_spinner=False)
def obtener_iupac_pubchem(smiles: str) -> str:
    """Consulta el nombre IUPAC oficial vía PubChem PUG-REST a partir del SMILES."""
    try:
        smi_encoded = urllib.parse.quote(smiles.strip())
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smi_encoded}/property/IUPACName/JSON"
        response = requests.get(url, timeout=4)
        if response.status_code == 200:
            data = response.json()
            return data["PropertyTable"]["Properties"][0].get("IUPACName", "No registrado en PubChem")
        return "No encontrado en PubChem"
    except Exception:
        return "Servicio PubChem no disponible temporalmente"

def calcular_9_descriptores_curso(mol):
    """Sección 9B: Calcula los 9 descriptores interpretables del cuaderno."""
    if mol is None:
        return None
    return {
        "MolWt (g/mol)": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA (Å²)": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
        "NumRotatableBonds": Descriptors.NumRotatableBonds(mol),
        "HeavyAtomCount": Descriptors.HeavyAtomCount(mol),
        "RingCount": Descriptors.RingCount(mol),
        "FractionCSP3": Descriptors.FractionCSP3(mol)
    }

def calcular_217_descriptores_mol(mol):
    """Calcula la lista completa de ~217 descriptores de RDKit para una molécula."""
    if mol is None:
        return {}
    resultados = {}
    for nombre, funcion in Descriptors.descList:
        try:
            resultados[nombre] = funcion(mol)
        except Exception:
            resultados[nombre] = np.nan
    return resultados

@st.cache_data(show_spinner=False)
def calcular_descriptores_dataset(smiles_series, modo="9_curso"):
    """Calcula los descriptores para los 1,128 compuestos del dataset."""
    lista = []
    for smi in smiles_series:
        mol = Chem.MolFromSmiles(str(smi))
        if modo == "9_curso":
            lista.append(calcular_9_descriptores_curso(mol))
        else:
            lista.append(calcular_217_descriptores_mol(mol))
    return pd.DataFrame(lista)

# ==============================================================================
# CARGA INICIAL Y BARRA LATERAL (SIDEBAR UX)
# ==============================================================================
df = cargar_dataset_delaney()
num_total_desc_rdkit = len(Descriptors.descList)

with st.sidebar:
    st.title("🧭 Control QSPR")
    st.caption("Sesión 2: Solubilidad Acuosa (ESOL/Delaney) • Puntos 1 al 9B")
    
    st.markdown("### 🔍 Selección de Molécula")
    compuesto_seleccionado = st.selectbox(
        "Buscar por Compound ID:",
        options=df["Compound ID"].tolist(),
        index=0,
        help="Selecciona uno de los 1,128 compuestos del dataset de Delaney."
    )
    
    idx_mol = df[df["Compound ID"] == compuesto_seleccionado].index[0]
    fila_mol = df.loc[idx_mol]
    
    st.divider()
    st.markdown("### 🎚️ Filtros del Espacio Químico")
    rango_logS = st.slider(
        "Solubilidad Exp. (log S [mol/L]):",
        min_value=float(df["measured log solubility in mols per litre"].min()),
        max_value=float(df["measured log solubility in mols per litre"].max()),
        value=(float(df["measured log solubility in mols per litre"].min()), float(df["measured log solubility in mols per litre"].max()))
    )
    
    rango_mw = st.slider(
        "Masa Molecular (g/mol):",
        min_value=float(df["Molecular Weight"].min()),
        max_value=float(df["Molecular Weight"].max()),
        value=(float(df["Molecular Weight"].min()), float(df["Molecular Weight"].max()))
    )
    
    decimales = st.slider("Redondeo de decimales:", 1, 6, 3)
    
    df_filtrado = df[
        (df["measured log solubility in mols per litre"].between(rango_logS[0], rango_logS[1])) &
        (df["Molecular Weight"].between(rango_mw[0], rango_mw[1]))
    ]
    st.info(f"Compuestos activos tras filtro: **{len(df_filtrado)} / {len(df)}**")

# ==============================================================================
# ENCABEZADO PRINCIPAL
# ==============================================================================
st.title("🧪 Predicción de Solubilidad Acuosa: SMILES y Descriptores RDKit")
st.markdown(
    "**Flujo QSPR (Puntos 1–9B):** `Estructura Molecular` ➔ `SMILES / IUPAC` ➔ `Grafo RDKit` ➔ `217 Descriptores Moleculares`"
)

# Pestañas organizadas según los puntos del cuaderno de Colab
tab_teoria, tab_dataset, tab_inspector, tab_malla, tab_matriz = st.tabs([
    "📘 1-4. Conceptos QSPR y SMILES",
    "🗂️ 6-7. Dataset ESOL/Delaney",
    "🔬 9. Inspector IUPAC, SMILES y 217 Descriptores",
    "🖼️ 8. Malla Molecular 2D",
    "📊 9B. Matriz Completa del Dataset"
])

# ==============================================================================
# PESTAÑA 1: CONTEXTO Y CONCEPTOS (PUNTOS 1 AL 4A)
# ==============================================================================
with tab_teoria:
    col_t1, col_t2 = st.columns([1.2, 1])
    with col_t1:
        st.subheader("1. Contexto del Caso y Pregunta Detonadora")
        st.info(
            "**¿Podemos predecir la solubilidad acuosa de un compuesto químico a partir de su estructura "
            "molecular para reducir tiempo, costo y experimentación innecesaria?**"
        )
        st.markdown("""
        En ingeniería química, ambiental y farmacéutica, medir experimentalmente la solubilidad acuosa ($\log S$, con $S$ en $\text{mol/L}$) de cientos de candidatos implica alto consumo de reactivos y tiempo. 
        El enfoque **QSPR** (*Quantitative Structure–Property Relationship*) transforma la estructura química en descriptores numéricos mediante **RDKit** para alimentar modelos predictivos.
        """)
        
        st.subheader("4.2 Los 9 Descriptores Base de la Sesión")
        tabla_desc = pd.DataFrame([
            ["MolWt", "Masa molecular promedio", "Mayor tamaño suele reducir la solubilidad"],
            ["MolLogP", "Coeficiente de partición octanol/agua (Crippen)", "Mayor valor indica mayor hidrofobicidad (menor solubilidad)"],
            ["TPSA", "Área polar superficial topológica", "Mayor área polar favorece la solvatación en agua"],
            ["NumHDonors", "Donadores de puente de hidrógeno (-OH, -NH)", "Aumenta la interacción soluto-agua"],
            ["NumHAcceptors", "Aceptores de puente de hidrógeno", "Favorece enlaces de hidrógeno con el solvente"],
            ["NumRotatableBonds", "Enlaces simples rotables no terminales", "Mide flexibilidad conformacional"],
            ["HeavyAtomCount", "Número de átomos distintos de Hidrógeno", "Indicador directo del tamaño estructural"],
            ["RingCount", "Número de anillos en el grafo molecular", "Aumenta rigidez y planaridad"],
            ["FractionCSP3", "Fracción de carbonos hibridados sp³", "Mide saturación y carácter tridimensional"]
        ], columns=["Descriptor", "¿Qué mide?", "Relación con Solubilidad Acuosa"])
        st.dataframe(tabla_desc, use_container_width=True, hide_index=True)

    with col_t2:
        st.subheader("🧪 Laboratorio Rápido de SMILES (Sección 4.5)")
        st.markdown("Prueba los ejemplos clásicos del cuaderno o escribe cualquier cadena SMILES:")
        ejemplos_curso = {
            "Etanol (CH₃–CH₂–OH)": "CCO",
            "Acetona (CH₃–C(=O)–CH₃)": "CC(=O)C",
            "Ácido Acético (CH₃–C(=O)–OH)": "CC(=O)O",
            "Benceno (Anillo aromático)": "c1ccccc1",
            "Dióxido de carbono": "O=C=O"
        }
        ej_sel = st.radio("Selecciona un ejemplo paso a paso:", list(ejemplos_curso.keys()))
        smi_ej = ejemplos_curso[ej_sel]
        mol_ej = Chem.MolFromSmiles(smi_ej)
        
        st.code(f"SMILES: {smi_ej}", language="text")
        st.image(Draw.MolToImage(mol_ej, size=(280, 220)), caption=f"Grafo molecular: {ej_sel}")
        st.caption(f"**Átomos pesados:** `{mol_ej.GetNumAtoms()}` | **Enlaces:** `{mol_ej.GetNumBonds()}` | **MolWt:** `{Descriptors.MolWt(mol_ej):.2f}`")

# ==============================================================================
# PESTAÑA 2: EXPLORACIÓN DEL DATASET DELANEY (PUNTOS 6 Y 7)
# ==============================================================================
with tab_dataset:
    st.subheader("6 & 7. Revisión Inicial del Dataset ESOL / Delaney")
    
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total de Compuestos", f"{df.shape[0]}")
    k2.metric("Columnas Originales", f"{df.shape[1]}")
    k3.metric("Valores Nulos Totales", f"{df.isnull().sum().sum()}")
    k4.metric("log S Promedio", f"{df['measured log solubility in mols per litre'].mean():.2f} mol/L")
    
    sub_vista = st.radio(
        "Selecciona la vista del conjunto de datos:",
        ["Columnas Clave (Compound ID, SMILES, log S)", "Dataset Completo Filtrado", "Estadísticas Descriptivas (df.describe)"],
        horizontal=True
    )
    
    if sub_vista == "Columnas Clave (Compound ID, SMILES, log S)":
        st.dataframe(
            df_filtrado[["Compound ID", "smiles", "measured log solubility in mols per litre", "ESOL predicted log solubility in mols per litre"]].round(decimales),
            use_container_width=True, height=380
        )
    elif sub_vista == "Dataset Completo Filtrado":
        st.dataframe(df_filtrado.round(decimales), use_container_width=True, height=380)
    else:
        st.dataframe(df_filtrado.describe().round(decimales), use_container_width=True)

    # Gráfico interactivo del espacio químico en Delaney
    st.markdown("#### Distribución del Espacio Químico en ESOL/Delaney")
    fig_esol = px.scatter(
        df_filtrado,
        x="Molecular Weight",
        y="measured log solubility in mols per litre",
        color="Polar Surface Area",
        hover_name="Compound ID",
        hover_data=["smiles", "Number of Rings", "Number of H-Bond Donors"],
        color_continuous_scale="Tealrose",
        labels={
            "Molecular Weight": "Masa Molecular (g/mol)",
            "measured log solubility in mols per litre": "Solubilidad Experimental (log S)"
        },
        template="plotly_white",
        height=420
    )
    st.plotly_chart(fig_esol, use_container_width=True)

# ==============================================================================
# PESTAÑA 3: INSPECTOR INDIVIDUAL: IUPAC, SMILES Y 217 DESCRIPTORES (PUNTO 9)
# ==============================================================================
with tab_inspector:
    st.subheader("9. Inspector Molecular: Nombre IUPAC, SMILES y los 217 Descriptores de RDKit")
    
    modo_entrada = st.toggle("✍️ Ingresar un SMILES personalizado manualmente", value=False)
    
    if modo_entrada:
        smi_activo = st.text_input("Escribe el código SMILES:", value="CCO")
        nombre_comun = "Compuesto Personalizado"
        log_s_exp = None
        log_s_esol = None
    else:
        smi_activo = str(fila_mol["smiles"]).strip()
        nombre_comun = str(fila_mol["Compound ID"])
        log_s_exp = fila_mol["measured log solubility in mols per litre"]
        log_s_esol = fila_mol["ESOL predicted log solubility in mols per litre"]

    mol_activa = Chem.MolFromSmiles(smi_activo)

    if mol_activa is None:
        st.error("❌ El código SMILES ingresado no es químicamente válido según RDKit.")
    else:
        col_mol_img, col_mol_id = st.columns([1.1, 1.9])
        
        with col_mol_img:
            st.markdown(f"##### Renderizado 2D: **{nombre_comun}**")
            img_2d = Draw.MolToImage(mol_activa, size=(360, 320))
            st.image(img_2d, use_container_width=True)
            
            c_at, c_en = st.columns(2)
            c_at.metric("Átomos explícitos/pesados", mol_activa.GetNumAtoms())
            c_en.metric("Enlaces en grafo", mol_activa.GetNumBonds())

        with col_mol_id:
            st.markdown("##### Identificadores Químicos Oficiales")
            
            with st.spinner("Consultando nomenclatura IUPAC en PubChem..."):
                nombre_iupac = obtener_iupac_pubchem(smi_activo)
            
            st.markdown(f"**Identificador Dataset (`Compound ID`):** `{nombre_comun}`")
            st.markdown("**Nombre Sistemático IUPAC (vía PubChem):**")
            st.markdown(f'<div class="iupac-box">🏷️ {nombre_iupac}</div>', unsafe_allow_html=True)
            
            st.markdown("**Código SMILES (Original y Canónico RDKit):**")
            smi_canonico = Chem.MolToSmiles(mol_activa, canonical=True)
            st.markdown(f'<div class="smiles-box">🧬 <b>Dataset:</b> {smi_activo}<br>✨ <b>Canónico:</b> {smi_canonico}</div>', unsafe_allow_html=True)
            
            if log_s_exp is not None:
                s1, s2, s3 = st.columns(3)
                s1.metric("log S Experimental", f"{log_s_exp:.2f} mol/L")
                s2.metric("log S Predicho (ESOL)", f"{log_s_esol:.2f} mol/L")
                s3.metric("Error Absoluto ESOL", f"{abs(log_s_exp - log_s_esol):.2f}")

            st.markdown("##### Los 9 Descriptores del Curso (Sección 9)")
            desc_9 = calcular_9_descriptores_curso(mol_activa)
            d_cols = st.columns(3)
            for idx_d, (k_desc, v_desc) in enumerate(desc_9.items()):
                val_fmt = f"{v_desc:.{decimales}f}" if isinstance(v_desc, float) else f"{v_desc}"
                d_cols[idx_d % 3].metric(k_desc, val_fmt)

        st.divider()
        st.markdown(f"#### 🧮 Explorador de los `{num_total_desc_rdkit}` Descriptores Moleculares de RDKit para **{nombre_comun}**")
        
        dict_217 = calcular_217_descriptores_mol(mol_activa)
        df_217_single = pd.DataFrame(
            [(k, v) for k, v in dict_217.items()],
            columns=["Nombre del Descriptor RDKit", "Valor Calculado"]
        )
        
        busqueda_desc = st.text_input(
            "🔎 Buscar descriptor por nombre (ej. Chi, Kappa, VSA, EState, MolWt, Ring, Bertz):",
            placeholder="Escribe para filtrar entre los 217 descriptores..."
        )
        
        if busqueda_desc:
            df_217_mostrar = df_217_single[df_217_single["Nombre del Descriptor RDKit"].str.contains(busqueda_desc, case=False, na=False)]
        else:
            df_217_mostrar = df_217_single
            
        col_tab_217, col_graf_217 = st.columns([1.4, 1.6])
        with col_tab_217:
            st.dataframe(
                df_217_mostrar.round(decimales),
                use_container_width=True,
                height=360,
                hide_index=True
            )
            csv_single = df_217_single.to_csv(index=False).encode("utf-8")
            st.download_button(
                f"📥 Descargar los {num_total_desc_rdkit} descriptores de {nombre_comun} (.CSV)",
                data=csv_single,
                file_name=f"descriptores_rdkit_{nombre_comun}.csv",
                mime="text/csv"
            )
        with col_graf_217:
            # Mostrar perfil de conteos estructurales del compuesto
            conteos_clave = {k: dict_217[k] for k in [
                "HeavyAtomCount", "NumHeteroatoms", "NumValenceElectrons", 
                "NumHDonors", "NumHAcceptors", "RingCount", "NumAromaticRings", "NumRotatableBonds"
            ] if k in dict_217}
            fig_bar = px.bar(
                x=list(conteos_clave.keys()),
                y=list(conteos_clave.values()),
                labels={"x": "Descriptor Estructural", "y": "Conteo"},
                title=f"Perfil de Conteos Constitucionales ({nombre_comun})",
                template="plotly_white",
                color=list(conteos_clave.values()),
                color_continuous_scale="Blues"
            )
            fig_bar.update_layout(height=360, showlegend=False)
            st.plotly_chart(fig_bar, use_container_width=True)

# ==============================================================================
# PESTAÑA 4: VISUALIZACIÓN EN MALLA 2D (PUNTO 8)
# ==============================================================================
with tab_malla:
    st.subheader("8. Visualización de Moléculas en Cuadrícula (Draw.MolsToGridImage)")
    
    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        n_mols = st.slider("Cantidad de moléculas a visualizar:", min_value=4, max_value=25, value=12, step=1)
    with c_m2:
        mols_por_fila = st.slider("Moléculas por fila (molsPerRow):", min_value=2, max_value=6, value=4)
    with c_m3:
        inicio_idx = st.number_input("Índice inicial en el dataset filtrado:", min_value=0, max_value=max(0, len(df_filtrado) - n_mols), value=0)
        
    sub_df = df_filtrado.iloc[inicio_idx : inicio_idx + n_mols]
    mols_grid = [Chem.MolFromSmiles(s) for s in sub_df["smiles"]]
    leyendas_grid = [
        f"{cid} (logS: {logs:.2f})" 
        for cid, logs in zip(sub_df["Compound ID"], sub_df["measured log solubility in mols per litre"])
    ]
    
    if mols_grid:
        img_grid = Draw.MolsToGridImage(
            mols_grid,
            molsPerRow=mols_por_fila,
            subImgSize=(260, 210),
            legends=leyendas_grid
        )
        st.image(img_grid, use_container_width=True)

# ==============================================================================
# PESTAÑA 5: CÁLCULO MASIVO PARA TODAS LAS MOLÉCULAS (PUNTO 9B)
# ==============================================================================
with tab_matriz:
    st.subheader("9B. Cálculo de Descriptores para Todos los Compuestos del Dataset")
    st.markdown("""
    Elige si deseas generar la tabla con los **9 descriptores seleccionados en el cuaderno** o extraer la matriz completa con los **~217 descriptores moleculares de RDKit** para los compuestos filtrados.
    """)
    
    col_modo, col_accion = st.columns([2, 1])
    with col_modo:
        tipo_calculo = st.radio(
            "Selecciona el conjunto de descriptores a construir:",
            ["🎯 Los 9 descriptores del curso (ejecución instantánea)", f"🚀 Todos los {num_total_desc_rdkit} descriptores de RDKit (Matriz QSPR completa)"],
            horizontal=True
        )
    with col_accion:
        ejecutar_masivo = st.button("⚡ Ejecutar Cálculo en lote", type="primary", use_container_width=True)

    if tipo_calculo.startswith("🎯"):
        df_desc_9 = calcular_descriptores_dataset(df_filtrado["smiles"], modo="9_curso")
        df_resultado_9 = pd.concat(
            [df_filtrado[["Compound ID", "smiles", "measured log solubility in mols per litre"]].reset_index(drop=True), df_desc_9],
            axis=1
        )
        st.dataframe(df_resultado_9.round(decimales), use_container_width=True, height=420)
        st.download_button(
            "📥 Descargar Dataset + 9 Descriptores (.CSV)",
            data=df_resultado_9.to_csv(index=False).encode("utf-8"),
            file_name="delaney_9_descriptores_rdkit.csv",
            mime="text/csv"
        )
    else:
        if ejecutar_masivo or "matriz_217_cache" in st.session_state:
            with st.spinner(f"Calculando {num_total_desc_rdkit} descriptores para {len(df_filtrado)} moléculas..."):
                df_desc_217 = calcular_descriptores_dataset(df_filtrado["smiles"], modo="217_todos")
                df_resultado_217 = pd.concat(
                    [df_filtrado[["Compound ID", "smiles", "measured log solubility in mols per litre"]].reset_index(drop=True), df_desc_217],
                    axis=1
                )
                st.session_state["matriz_217_cache"] = df_resultado_217
                
            df_mostrar_217 = st.session_state["matriz_217_cache"]
            st.success(f"✅ Matriz generada exitosamente: **{df_mostrar_217.shape[0]} filas × {df_mostrar_217.shape[1]} columnas**.")
            st.dataframe(df_mostrar_217.round(decimales), use_container_width=True, height=420)
            
            st.download_button(
                f"📥 Descargar Matriz QSPR Completa ({num_total_desc_rdkit} descriptores) (.CSV)",
                data=df_mostrar_217.to_csv(index=False).encode("utf-8"),
                file_name="delaney_217_descriptores_rdkit.csv",
                mime="text/csv"
            )
        else:
            st.info(f"Haz clic en **'⚡ Ejecutar Cálculo en lote'** para calcular los {num_total_desc_rdkit} descriptores en los {len(df_filtrado)} compuestos.")
