import streamlit as st
import folium
from streamlit_folium import st_folium
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Monitoramento Trigona spinipes", layout="wide")

# --- 1. INTRODUÇÃO À ESPÉCIE ---
st.title("Monitoramento e Previsão: Trigona spinipes")

col_intro_texto, col_intro_box = st.columns([2, 1])

with col_intro_texto:
    st.markdown("""
    ### Sobre a Espécie: Irapuá (*Trigona spinipes*)
    A *Trigona spinipes*, popularmente conhecida como **Irapuá**, **Arapuá**, **Abelha-cachorro** ou **Abelha-irapuá**, é uma abelha social sem ferrão (tribo Meliponini) nativa da América do Sul e extremamente comum no Brasil.

    #### 🐝 Características e Comportamento
    * **Defesa Agressiva:** Embora não possua ferrão funcional, é uma espécie territorial e defensiva. Ao se sentir ameaçada, ataca em enxame, enrolando-se nos cabelos e pelos do intruso e mordiscando a pele com suas fortes mandíbulas.
    * **Polinizadora Generalista:** Desempenha um papel crucial na polinização de diversas plantas nativas e cultivadas (como cenoura, girassol, laranja e manga).
    * **Comportamento de Coleta:** É conhecida por cortar botões florais e fibras vegetais para coletar resina e material de construção, o que ocasionalmente pode causar danos a certas culturas agrícolas (ex: citros e bananeiras).

    #### 🏠 Nidificação (Ninhos Aéreos)
    * Diferente de muitas abelhas que nidificam em ocos de árvores, a Irapuá constrói **ninhos externos, globosos e grandes**, visíveis em forquilhas de árvores ou estruturas humanas.
    * O ninho é feito de uma mistura resistente de barro, cerume, resinas e fibras vegetais.
    """)

with col_intro_box:
    st.info("""
    **Classificação Científica**
    * **Reino:** Animalia
    * **Ordem:** Hymenoptera
    * **Família:** Apidae
    * **Tribo:** Meliponini
    * **Gênero:** *Trigona*
    * **Espécie:** *T. spinipes*
    """)
    
    st.warning("""
    **Curiosidade:**
    O nome "Irapuá" tem origem tupi (*eíra* = mel, *apu'a* = redondo), significando **"Mel Redondo"**, uma referência direta ao formato característico de seu ninho.
    """)

st.divider()

# --- FUNÇÕES DE CARREGAMENTO E CÁLCULOS ---
@st.cache_data
def carregar_dados_ocorrencias():
    try:
        try:
            df = pd.read_csv('data/ocorrencias.csv', sep='\t')
        except FileNotFoundError:
            df = pd.read_csv('ocorrencias.csv', sep='\t')
            
        df_valid = df.dropna(subset=['year', 'decimalLatitude', 'decimalLongitude'])
        df_valid = df_valid[(df_valid['year'] >= 1900) & (df_valid['year'] <= 2024)]
        df_valid['year'] = df_valid['year'].astype(int)
        return df_valid
    except Exception:
        return None

def obter_top_estados(df):
    if df is None or 'stateProvince' not in df.columns: return None
    return df['stateProvince'].value_counts().head(8)

def calcular_centroide_historico(df):
    """Calcula a latitude média das ocorrências históricas."""
    if df is None: return None
    lat_media = df['decimalLatitude'].mean()
    return lat_media

# Carregamento dos dados iniciais
df_ocorrencias = carregar_dados_ocorrencias()
stats_estados = obter_top_estados(df_ocorrencias)
hist_lat = calcular_centroide_historico(df_ocorrencias)

# --- 2. HISTÓRICO DE OCORRÊNCIAS ---
st.header("Histórico de Ocorrências Registradas")
st.markdown("Visualização dos registros históricos temporais e geográficos.")

if df_ocorrencias is not None:
    col_temporal, col_geografico = st.columns(2)
    
    with col_temporal:
        st.subheader("Evolução Temporal (1900-2024)")
        contagem_anos = df_ocorrencias['year'].value_counts().sort_index()
        
        fig_line, ax_line = plt.subplots(figsize=(6, 4))
        ax_line.plot(contagem_anos.index, contagem_anos.values, color="#FF4B4B", linewidth=2)
        ax_line.set_ylabel("Registros")
        ax_line.set_xlabel("Ano")
        ax_line.grid(True, which='both', axis='y', linestyle='--', alpha=0.3)
        ax_line.spines['top'].set_visible(False)
        ax_line.spines['right'].set_visible(False)
        st.pyplot(fig_line, use_container_width=True)

    with col_geografico:
        st.subheader("Principais Estados de Ocorrência")
        if stats_estados is not None:
            fig_bar, ax_bar = plt.subplots(figsize=(6, 4))
            cores = ['#FF4B4B' if i == 0 else '#FF9999' for i in range(len(stats_estados))]
            stats_estados.plot(kind='bar', color=cores, ax=ax_bar)
            ax_bar.set_ylabel("Registros Totais")
            ax_bar.set_xticklabels(stats_estados.index, rotation=45, ha='right')
            ax_bar.grid(axis='y', alpha=0.3)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)
            st.pyplot(fig_bar, use_container_width=True)
        else:
            st.warning("Dados de estados indisponíveis.")
else:
    st.warning("Arquivo 'ocorrencias.csv' não encontrado.")

st.divider()

# --- 3. ANÁLISE DE TENDÊNCIA LATITUDINAL (NOVA SEÇÃO REFORMULADA) ---
st.header("Análise de Tendência Latitudinal e Habitat")
st.markdown("Análise de como a adequabilidade climática se distribui ao longo da latitude (Norte-Sul) e a disponibilidade de áreas ideais.")

arquivos_predicao = {
    "2021-2040": "data/previsoes_futuras/previsao_trigona_wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2021-2040.tif",
    "2041-2060": "data/previsoes_futuras/previsao_trigona_wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2041-2060.tif",
    "2061-2080": "data/previsoes_futuras/previsao_trigona_wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2061-2080.tif",
    "2081-2100": "data/previsoes_futuras/previsao_trigona_wc2.1_10m_bioc_BCC-CSM2-MR_ssp245_2081-2100.tif"
}

def analisar_perfil_latitudinal(arquivo_tif):
    """
    Calcula o perfil de adequabilidade média por latitude e porcentagem de área ideal.
    """
    try:
        with rasterio.open(arquivo_tif) as src:
            data = src.read(1)
            
            # Tratar NoData (converter para NaN para não afetar a média)
            if src.nodata is not None:
                data_float = np.where(data == src.nodata, np.nan, data)
            else:
                data_float = np.where(data == -9999, np.nan, data) # Fallback comum

            # --- KPI 1: Porcentagem de Área de Alta Qualidade (>0.6) ---
            valid_pixels = data_float[~np.isnan(data_float)]
            if valid_pixels.size == 0: return None
            
            pct_alta = (np.sum(valid_pixels > 0.6) / valid_pixels.size) * 100
            
            # --- KPI 2: Perfil Latitudinal ---
            # Calcula a média de cada linha (eixo 1) ignorando NaNs
            with np.errstate(divide='ignore', invalid='ignore'):
                lat_means = np.nanmean(data_float, axis=1)
            
            # Cria array de latitudes correspondentes às linhas
            height = data.shape[0]
            bounds = src.bounds
            # Latitudes vão do Topo (bounds.top) para Baixo (bounds.bottom)
            lats = np.linspace(bounds.top, bounds.bottom, height)
            
            # Filtrar linhas que ficaram totalmente NaN
            valid_rows = ~np.isnan(lat_means)
            lat_means_valid = lat_means[valid_rows]
            lats_valid = lats[valid_rows]

            if lats_valid.size == 0: return None
            
            # Encontrar a Latitude com o pico máximo de adequabilidade média
            idx_max = np.argmax(lat_means_valid)
            lat_pico = lats_valid[idx_max]
            valor_pico = lat_means_valid[idx_max]

            return {
                "lats": lats_valid,
                "means": lat_means_valid,
                "lat_pico": lat_pico,
                "valor_pico": valor_pico,
                "pct_alta": pct_alta
            }
    except Exception as e:
        return None

abas_periodos = st.tabs(list(arquivos_predicao.keys()))

for aba, (periodo, arquivo_tif) in zip(abas_periodos, arquivos_predicao.items()):
    with aba:
        st.subheader(f"Cenário {periodo}")
        stats = analisar_perfil_latitudinal(arquivo_tif)
        
        if stats and hist_lat is not None:
            # --- KPIs ---
            col_k1, col_k2, col_k3 = st.columns(3)
            
            with col_k1:
                st.metric("Latitude Média Histórica", f"{hist_lat:.2f}°", help="Latitude média de todos os registros no CSV.")
                
            with col_k2:
                # Mostra onde será o "melhor lugar" (Pico da curva)
                diff = stats['lat_pico'] - hist_lat
                direcao = "Norte" if diff > 0 else "Sul"
                st.metric("Latitude Ideal Prevista", f"{stats['lat_pico']:.2f}°", 
                         delta=f"{abs(diff):.2f}° ({direcao})", delta_color="off",
                         help="Latitude onde a adequabilidade média atinge seu ponto máximo neste cenário.")
                
            with col_k3:
                st.metric("Área de Alta Qualidade (>0.6)", f"{stats['pct_alta']:.1f}%",
                         help="Porcentagem do território analisado com índice de adequabilidade superior a 0.6.")

            st.markdown("---")

            # --- VISUALIZAÇÃO: GRÁFICO DE LINHA (PERFIL LATITUDINAL) ---
            col_graf, col_desc = st.columns([2, 1])
            
            with col_graf:
                st.markdown("**Perfil Latitudinal de Adequabilidade**")
                fig_lat, ax_lat = plt.subplots(figsize=(8, 4))
                
                # Plot da curva (Latitude no Eixo Y, Adequabilidade no Eixo X)
                ax_lat.plot(stats['means'], stats['lats'], color='#2E8B57', linewidth=2, label='Adequabilidade Média')
                
                # Linha de referência histórica
                ax_lat.axhline(y=hist_lat, color='#FF4B4B', linestyle='--', label='Média Histórica')
                
                # Linha de pico previsto
                ax_lat.axhline(y=stats['lat_pico'], color='#2E8B57', linestyle=':', label='Pico Previsto')

                ax_lat.set_ylabel("Latitude (Graus)")
                ax_lat.set_xlabel("Índice Médio de Adequabilidade")
                ax_lat.set_title(f"Distribuição Norte-Sul da Qualidade do Habitat ({periodo})")
                ax_lat.legend()
                ax_lat.grid(True, alpha=0.3)
                ax_lat.spines['top'].set_visible(False)
                ax_lat.spines['right'].set_visible(False)
                
                # Inverter eixo Y se necessário (mas latitudes negativas já ordenam corretamente no plot padrão)
                # Se o gráfico parecer invertido (Norte embaixo), descomente a linha abaixo:
                # ax_lat.invert_yaxis() 
                
                st.pyplot(fig_lat, use_container_width=True)

            with col_desc:
                st.info(f"""
                **Interpretação do Gráfico:**
                
                Este gráfico mostra onde estão as melhores condições para a *Trigona spinipes* ao longo do eixo Norte-Sul.
                
                * **Eixo Vertical:** Latitude (Quanto mais alto, mais ao Norte; quanto mais baixo/negativo, mais ao Sul).
                * **Eixo Horizontal:** Qualidade média do clima.
                * **Linha Tracejada Vermelha:** Onde a espécie costumava estar historicamente.
                * **Linha Pontilhada Verde:** Onde o clima será melhor neste cenário futuro.
                
                Se a curva verde se deslocar para baixo em relação à linha vermelha, indica uma tendência de **migração para o Sul**.
                """)

        else:
            st.error("Erro ao processar o arquivo de previsão ou dados históricos indisponíveis.")

st.divider()

# --- 4. MAPAS DE CALOR ESTÁTICOS ---
st.header("Comparativo Visual: Evolução do Habitat")
st.markdown("Visualização dos mapas de calor para cada período.")

def plot_clean_heatmap(file_path, title, ax):
    try:
        with rasterio.open(file_path) as src:
            data = src.read(1)
            if src.nodata is not None:
                data = np.where(data == src.nodata, np.nan, data)
            
            im = ax.imshow(data, cmap='YlGn')
            
            ax.set_title(title, fontsize=14)
            ax.axis('off')
            ax.set_frame_on(False)
            
            cbar = plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
            cbar.ax.tick_params(labelsize=9)
            cbar.outline.set_visible(False)
            cbar.set_label('Índice de Adequamento', rotation=270, labelpad=15, fontsize=10)
            return True
    except Exception:
        ax.text(0.5, 0.5, "Erro na leitura", ha='center', va='center')
        ax.axis('off')
        return False

periodos_info = [
    {"periodo": "2021-2040", "arquivo": arquivos_predicao["2021-2040"], "texto": "Cenário inicial de curto prazo."},
    {"periodo": "2041-2060", "arquivo": arquivos_predicao["2041-2060"], "texto": "Projeção de médio prazo (Cenário SSP2-4.5)."},
    {"periodo": "2061-2080", "arquivo": arquivos_predicao["2061-2080"], "texto": "Projeção de longo prazo."},
    {"periodo": "2081-2100", "arquivo": arquivos_predicao["2081-2100"], "texto": "Cenário final (secular)."}
]

texto_generico = """
**Sobre a Trigona spinipes e Adaptação Climática**

A *Trigona spinipes* (Irapuá) é uma espécie chave para a polinização de flora nativa.

O mapa ao lado representa o modelo para o período futuro selecionado. As áreas em **verde escuro** indicam regiões favoráveis, enquanto **amarelo claro** indica baixo adequamento.
"""

for info in periodos_info:
    st.subheader(f"Período: {info['periodo']}")
    col_mapa, col_texto = st.columns([1, 1])
    
    with col_mapa:
        fig, ax = plt.subplots(figsize=(8, 6), facecolor='none')
        plot_clean_heatmap(info["arquivo"], f"Predição {info['periodo']}", ax)
        st.pyplot(fig, use_container_width=True)
        
    with col_texto:
        st.markdown(f"#### Detalhes do Período {info['periodo']}")
        st.markdown(info['texto'])
        st.markdown("---")
        st.markdown(texto_generico)
    st.write("---")

st.divider()

# --- 5. MAPA INTERATIVO ---
st.header("Mapa Interativo de Adequabilidade (Zoom e Detalhes)")
st.markdown("Utilize as abas abaixo para explorar o mapa com zoom e interatividade.")

def colorize_raster_interactive(data, cmap_name='RdYlGn'):
    valid_data = data[~np.isnan(data)]
    if valid_data.size == 0: return np.zeros((data.shape[0], data.shape[1], 4))
    min_val, max_val = valid_data.min(), valid_data.max()
    if max_val == min_val:
        norm_data = np.zeros_like(data)
    else:
        norm_data = (data - min_val) / (max_val - min_val)
    
    cmap = plt.get_cmap(cmap_name)
    colored_data = cmap(norm_data)
    colored_data[np.isnan(data), 3] = 0
    return colored_data

def get_bounds(src):
    return [[src.bounds.bottom, src.bounds.left], [src.bounds.top, src.bounds.right]]

tabs = st.tabs(list(arquivos_predicao.keys()))

for i, (periodo, arquivo) in enumerate(arquivos_predicao.items()):
    with tabs[i]:
        try:
            with rasterio.open(arquivo) as src:
                data = src.read(1)
                if src.nodata is not None: data = np.where(data == src.nodata, np.nan, data)
                img = colorize_raster_interactive(data)
                m = folium.Map(location=[-14.2350, -51.9253], zoom_start=4)
                folium.raster_layers.ImageOverlay(image=img, bounds=get_bounds(src), opacity=0.7).add_to(m)
                st_folium(m, width=800, height=500)
        except Exception as e:
            st.error(f"Erro ao carregar o arquivo {arquivo}.")