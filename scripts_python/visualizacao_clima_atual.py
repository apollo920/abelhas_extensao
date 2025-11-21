import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.mask import mask
import geopandas as gpd
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap
import glob
import os
import warnings
warnings.filterwarnings('ignore')

class VisualizadorClima:
    def __init__(self, base_path='data', brasil_shape_path=None):
        self.base_path = Path(base_path)
        self.brasil_shapefile = None
        self.clima_path = self.base_path / 'clima_atual'
        
        # Usar BRASIL_SHAPE_PATH global se não for fornecido
        if brasil_shape_path is None:
            self.brasil_shape_path = BRASIL_SHAPE_PATH
        else:
            self.brasil_shape_path = brasil_shape_path
        
        # Dicionário com informações das variáveis
        self.bio_info = {
            'bio1': {
                'nome': 'Temperatura Média Anual',
                'unidade': '°C',
                'escala': 10,  # WorldClim usa *10
                'cmap': 'RdYlBu_r',
                'descricao': 'Média das temperaturas mensais do ano'
            },
            'bio2': {
                'nome': 'Variação Diurna Média',
                'unidade': '°C',
                'escala': 10,
                'cmap': 'YlOrRd',
                'descricao': 'Média mensal (temp. máx - temp. mín)'
            },
            'bio3': {
                'nome': 'Isotermalidade',
                'unidade': '%',
                'escala': 1,
                'cmap': 'viridis',
                'descricao': '(BIO2/BIO7) × 100'
            },
            'bio4': {
                'nome': 'Sazonalidade de Temperatura',
                'unidade': 'CV',
                'escala': 100,
                'cmap': 'plasma',
                'descricao': 'Desvio padrão × 100'
            },
            'bio5': {
                'nome': 'Temp. Máxima do Mês Mais Quente',
                'unidade': '°C',
                'escala': 10,
                'cmap': 'hot',
                'descricao': 'Temperatura máxima registrada'
            },
            'bio6': {
                'nome': 'Temp. Mínima do Mês Mais Frio',
                'unidade': '°C',
                'escala': 10,
                'cmap': 'coolwarm',
                'descricao': 'Temperatura mínima registrada'
            },
            'bio7': {
                'nome': 'Variação Anual de Temperatura',
                'unidade': '°C',
                'escala': 10,
                'cmap': 'RdYlGn_r',
                'descricao': 'BIO5 - BIO6'
            },
            'bio8': {
                'nome': 'Temp. Média Trimestre Úmido',
                'unidade': '°C',
                'escala': 10,
                'cmap': 'RdYlBu_r',
                'descricao': 'Temp. média no trimestre mais úmido'
            },
            'bio9': {
                'nome': 'Temp. Média Trimestre Seco',
                'unidade': '°C',
                'escala': 10,
                'cmap': 'RdYlBu_r',
                'descricao': 'Temp. média no trimestre mais seco'
            },
            'bio10': {
                'nome': 'Temp. Média Trimestre Quente',
                'unidade': '°C',
                'escala': 10,
                'cmap': 'RdYlBu_r',
                'descricao': 'Temp. média no trimestre mais quente'
            },
            'bio11': {
                'nome': 'Temp. Média Trimestre Frio',
                'unidade': '°C',
                'escala': 10,
                'cmap': 'RdYlBu_r',
                'descricao': 'Temp. média no trimestre mais frio'
            },
            'bio12': {
                'nome': 'Precipitação Anual',
                'unidade': 'mm',
                'escala': 1,
                'cmap': 'Blues',
                'descricao': 'Soma das precipitações mensais'
            },
            'bio13': {
                'nome': 'Precipitação Mês Mais Úmido',
                'unidade': 'mm',
                'escala': 1,
                'cmap': 'Blues',
                'descricao': 'Precipitação máxima mensal'
            },
            'bio14': {
                'nome': 'Precipitação Mês Mais Seco',
                'unidade': 'mm',
                'escala': 1,
                'cmap': 'YlGnBu',
                'descricao': 'Precipitação mínima mensal'
            },
            'bio15': {
                'nome': 'Sazonalidade de Precipitação',
                'unidade': 'CV',
                'escala': 1,
                'cmap': 'PuBuGn',
                'descricao': 'Coeficiente de variação'
            },
            'bio16': {
                'nome': 'Precipitação Trimestre Úmido',
                'unidade': 'mm',
                'escala': 1,
                'cmap': 'Blues',
                'descricao': 'Precip. no trimestre mais úmido'
            },
            'bio17': {
                'nome': 'Precipitação Trimestre Seco',
                'unidade': 'mm',
                'escala': 1,
                'cmap': 'YlGnBu',
                'descricao': 'Precip. no trimestre mais seco'
            },
            'bio18': {
                'nome': 'Precipitação Trimestre Quente',
                'unidade': 'mm',
                'escala': 1,
                'cmap': 'BuPu',
                'descricao': 'Precip. no trimestre mais quente'
            },
            'bio19': {
                'nome': 'Precipitação Trimestre Frio',
                'unidade': 'mm',
                'escala': 1,
                'cmap': 'GnBu',
                'descricao': 'Precip. no trimestre mais frio'
            }
        }
    
    def carregar_shapefile(self):
        """Carrega shapefile do Brasil usando a função do notebook"""
        print("📍 Carregando limites do Brasil...")
        
        try:
            # Usar a função do notebook
            self.brasil_shapefile = load_brazil_map_notebook(self.brasil_shape_path)
            print(f"✅ Shapefile carregado de: {self.brasil_shape_path}")
            print(f"✅ CRS: {self.brasil_shapefile.crs}")
            print(f"✅ Total de {len(self.brasil_shapefile)} estados\n")
            
        except IndexError:
            raise FileNotFoundError(
                f"Nenhum arquivo .shp encontrado em: {self.brasil_shape_path}\n"
                f"💡 Verifique se o caminho está correto e contém um shapefile."
            )
        except Exception as e:
            raise Exception(f"Erro ao carregar shapefile: {e}")
    
    def carregar_e_recortar_raster(self, bio_num):
        """
        Carrega raster e recorta para os limites do Brasil
        
        Parâmetros:
        - bio_num: número da variável (1-19)
        """
        # Encontrar arquivo
        pattern = f'*bio_{bio_num}.tif'
        tif_files = list(self.clima_path.glob(pattern))
        
        if len(tif_files) == 0:
            pattern = f'*bio{bio_num}.tif'
            tif_files = list(self.clima_path.glob(pattern))
        
        if len(tif_files) == 0:
            raise FileNotFoundError(f"Arquivo não encontrado para bio{bio_num}")
        
        tif_file = tif_files[0]
        
        # Carregar raster
        with rasterio.open(tif_file) as src:
            # Recortar para Brasil
            out_image, out_transform = mask(
                src, 
                self.brasil_shapefile.geometry, 
                crop=True,
                nodata=np.nan
            )
            
            data = out_image[0]
            
        return data, out_transform
    
    def plotar_variavel(self, bio_num, ax=None, adicionar_estados=True):
        """
        Plota uma variável bioclimática
        
        Parâmetros:
        - bio_num: número da variável (1-19)
        - ax: eixo matplotlib (cria novo se None)
        - adicionar_estados: se True, adiciona contornos dos estados
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 8))
        
        bio_key = f'bio{bio_num}'
        info = self.bio_info[bio_key]
        
        # Carregar dados
        data, transform = self.carregar_e_recortar_raster(bio_num)
        
        # Converter para unidades reais
        data_real = data / info['escala']
        
        # Plotar
        im = ax.imshow(
            data_real, 
            cmap=info['cmap'],
            aspect='auto',
            interpolation='bilinear'
        )
        
        # Adicionar contornos dos estados
        if adicionar_estados:
            self.brasil_shapefile.boundary.plot(
                ax=ax, 
                linewidth=0.5, 
                color='black', 
                alpha=0.5
            )
        
        # Título e labels
        ax.set_title(
            f"{info['nome']}\n{info['descricao']}", 
            fontsize=12, 
            fontweight='bold',
            pad=10
        )
        ax.axis('off')
        
        # Colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(info['unidade'], fontsize=10, fontweight='bold')
        
        # Estatísticas
        stats_text = f"Min: {np.nanmin(data_real):.1f}\n"
        stats_text += f"Média: {np.nanmean(data_real):.1f}\n"
        stats_text += f"Max: {np.nanmax(data_real):.1f}"
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        return ax, data_real
    
    def plotar_todas_variaveis(self, salvar=True):
        """Cria figura com todas as 19 variáveis"""
        print("🎨 Criando visualização de todas as variáveis...\n")
        
        # Criar grid de subplots
        fig = plt.figure(figsize=(24, 28))
        
        # 19 variáveis em grid 5x4
        for i in range(1, 20):
            ax = plt.subplot(5, 4, i)
            self.plotar_variavel(i, ax=ax, adicionar_estados=True)
            print(f"✓ BIO{i:02d} plotado")
        
        # Título geral
        fig.suptitle(
            'Variáveis Bioclimáticas do Brasil (WorldClim v2.1)',
            fontsize=20,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        
        if salvar:
            output_path = self.base_path / 'visualizacoes' / 'clima_atual_completo.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Figura salva: {output_path}")
        
        plt.show()
    
    def plotar_variaveis_temperatura(self, salvar=True):
        """Plota apenas variáveis de temperatura"""
        print("🌡️ Criando visualização de temperatura...\n")
        
        fig = plt.figure(figsize=(20, 14))
        
        vars_temp = [1, 2, 5, 6, 7, 8, 9, 10, 11]
        
        for idx, bio_num in enumerate(vars_temp, 1):
            ax = plt.subplot(3, 3, idx)
            self.plotar_variavel(bio_num, ax=ax)
            print(f"✓ BIO{bio_num} plotado")
        
        fig.suptitle(
            'Variáveis de Temperatura no Brasil',
            fontsize=18,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        
        if salvar:
            output_path = self.base_path / 'visualizacoes' / 'clima_temperatura.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Figura salva: {output_path}")
        
        plt.show()
    
    def plotar_variaveis_precipitacao(self, salvar=True):
        """Plota apenas variáveis de precipitação"""
        print("💧 Criando visualização de precipitação...\n")
        
        fig = plt.figure(figsize=(16, 12))
        
        vars_precip = [12, 13, 14, 15, 16, 17, 18, 19]
        
        for idx, bio_num in enumerate(vars_precip, 1):
            ax = plt.subplot(3, 3, idx)
            self.plotar_variavel(bio_num, ax=ax)
            print(f"✓ BIO{bio_num} plotado")
        
        fig.suptitle(
            'Variáveis de Precipitação no Brasil',
            fontsize=18,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        
        if salvar:
            output_path = self.base_path / 'visualizacoes' / 'clima_precipitacao.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Figura salva: {output_path}")
        
        plt.show()
    
    def plotar_principais_variaveis(self, salvar=True):
        """Plota as 4 variáveis mais importantes para espécies"""
        print("⭐ Criando visualização das principais variáveis...\n")
        
        fig = plt.figure(figsize=(16, 10))
        
        # BIO1, BIO12, BIO4, BIO15 (geralmente as mais importantes)
        vars_principais = [1, 12, 4, 15]
        
        for idx, bio_num in enumerate(vars_principais, 1):
            ax = plt.subplot(2, 2, idx)
            self.plotar_variavel(bio_num, ax=ax)
            print(f"✓ BIO{bio_num} plotado")
        
        fig.suptitle(
            'Principais Variáveis Climáticas do Brasil',
            fontsize=16,
            fontweight='bold',
            y=0.995
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.995])
        
        if salvar:
            output_path = self.base_path / 'visualizacoes' / 'clima_principais.png'
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"\n✅ Figura salva: {output_path}")
        
        plt.show()
    
    def criar_mapa_interativo(self, bio_num):
        """Cria visualização interativa de uma variável"""
        print(f"🗺️ Criando mapa interativo de BIO{bio_num}...\n")
        
        bio_key = f'bio{bio_num}'
        info = self.bio_info[bio_key]
        
        # Carregar dados
        data, transform = self.carregar_e_recortar_raster(bio_num)
        data_real = data / info['escala']
        
        # Criar figura interativa
        fig, ax = plt.subplots(figsize=(14, 10))
        
        im = ax.imshow(
            data_real, 
            cmap=info['cmap'],
            aspect='auto',
            interpolation='bilinear'
        )
        
        # Adicionar estados
        self.brasil_shapefile.boundary.plot(
            ax=ax, 
            linewidth=1, 
            color='black', 
            alpha=0.7
        )
        
        # Adicionar nomes dos estados (centroides)
        for idx, row in self.brasil_shapefile.iterrows():
            centroid = row.geometry.centroid
            ax.annotate(
                text=row['SIGLA'] if 'SIGLA' in row else '',
                xy=(centroid.x, centroid.y),
                fontsize=8,
                ha='center',
                color='black',
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
            )
        
        ax.set_title(
            f"{info['nome']}\n{info['descricao']}", 
            fontsize=14, 
            fontweight='bold',
            pad=15
        )
        ax.axis('off')
        
        # Colorbar aprimorada
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(info['unidade'], fontsize=12, fontweight='bold')
        
        # Estatísticas detalhadas
        stats_text = f"Estatísticas:\n"
        stats_text += f"Mínimo: {np.nanmin(data_real):.2f} {info['unidade']}\n"
        stats_text += f"Média: {np.nanmean(data_real):.2f} {info['unidade']}\n"
        stats_text += f"Mediana: {np.nanmedian(data_real):.2f} {info['unidade']}\n"
        stats_text += f"Máximo: {np.nanmax(data_real):.2f} {info['unidade']}\n"
        stats_text += f"Desvio Padrão: {np.nanstd(data_real):.2f} {info['unidade']}"
        
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9)
        )
        
        plt.tight_layout()
        
        # Salvar
        output_path = self.base_path / 'visualizacoes' / f'mapa_interativo_bio{bio_num}.png'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Mapa salvo: {output_path}")
        
        plt.show()
    
    def gerar_relatorio_clima(self):
        """Gera relatório PDF com todas as visualizações"""
        print("📄 Gerando relatório completo do clima...\n")
        
        from matplotlib.backends.backend_pdf import PdfPages
        
        output_path = self.base_path / 'visualizacoes' / 'relatorio_clima_brasil.pdf'
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with PdfPages(output_path) as pdf:
            # Página de título
            fig = plt.figure(figsize=(8.5, 11))
            fig.text(0.5, 0.7, 'Clima Atual do Brasil', 
                    ha='center', fontsize=28, fontweight='bold')
            fig.text(0.5, 0.6, 'Variáveis Bioclimáticas WorldClim v2.1', 
                    ha='center', fontsize=16)
            fig.text(0.5, 0.5, 'Resolução: 10 minutos (~20km)', 
                    ha='center', fontsize=12, color='gray')
            
            from datetime import datetime
            fig.text(0.5, 0.3, f'Gerado em: {datetime.now().strftime("%d/%m/%Y")}', 
                    ha='center', fontsize=10, color='gray')
            
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
            # Página para cada variável
            for bio_num in range(1, 20):
                print(f"  Gerando página BIO{bio_num}...")
                
                fig, ax = plt.subplots(figsize=(8.5, 11))
                self.plotar_variavel(bio_num, ax=ax, adicionar_estados=True)
                
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
        
        print(f"\n✅ Relatório completo salvo: {output_path}")
        print(f"   Total de páginas: 20 (1 título + 19 variáveis)")

# ============================================================================
# CONSTANTES CONFIGURÁVEIS
# ============================================================================

# Caminho para a pasta contendo o shapefile do Brasil
# Ajusta automaticamente se estiver rodando de notebooks/ ou da raiz
import os
if os.path.basename(os.getcwd()) == 'notebooks':
    BRASIL_SHAPE_PATH = "../data/BR_UF_2024"
    CLIMA_ATUAL_PATH = "../data/clima_atual"
else:
    BRASIL_SHAPE_PATH = "data/BR_UF_2024"
    CLIMA_ATUAL_PATH = "data/clima_atual"

# ============================================================================
# FUNÇÕES AUXILIARES
# ============================================================================

def load_brazil_map_notebook(path):
    """Função auxiliar para carregar o GeoDataFrame do Brasil."""
    shapefile_brasil = glob.glob(os.path.join(path, "*.shp"))[0]
    brasil_gdf = gpd.read_file(shapefile_brasil)
    # Garante que o CRS seja compatível com os rasters (EPSG:4326)
    if brasil_gdf.crs != "EPSG:4326":
        brasil_gdf = brasil_gdf.to_crs("EPSG:4326")
    return brasil_gdf

# ============================================================================
# CLASSE PRINCIPAL
# ============================================================================

def menu_visualizacao():
    """Menu interativo para escolher visualizações"""
    print("="*70)
    print("  🗺️  VISUALIZAÇÃO DO CLIMA ATUAL DO BRASIL  🗺️")
    print("="*70)
    print("\nEscolha uma opção:\n")
    print("  1 - Visualizar TODAS as 19 variáveis")
    print("  2 - Visualizar apenas variáveis de TEMPERATURA")
    print("  3 - Visualizar apenas variáveis de PRECIPITAÇÃO")
    print("  4 - Visualizar PRINCIPAIS variáveis (4 mais importantes)")
    print("  5 - Visualizar uma variável ESPECÍFICA (interativo)")
    print("  6 - Gerar RELATÓRIO PDF completo")
    print("  0 - Sair\n")
    
    while True:
        try:
            opcao = input("Digite o número da opção: ").strip()
            if opcao in ['0', '1', '2', '3', '4', '5', '6']:
                return opcao
            print("❌ Opção inválida. Tente novamente.")
        except KeyboardInterrupt:
            print("\n\n👋 Execução cancelada.")
            return '0'

# ============================================================================
# EXECUTAR
# ============================================================================

if __name__ == "__main__":
    try:
        # Inicializar usando a constante BRASIL_SHAPE_PATH
        viz = VisualizadorClima()
        viz.carregar_shapefile()
        
        # Menu
        opcao = menu_visualizacao()
        
        if opcao == '0':
            print("\n👋 Até logo!")
        
        elif opcao == '1':
            viz.plotar_todas_variaveis(salvar=True)
        
        elif opcao == '2':
            viz.plotar_variaveis_temperatura(salvar=True)
        
        elif opcao == '3':
            viz.plotar_variaveis_precipitacao(salvar=True)
        
        elif opcao == '4':
            viz.plotar_principais_variaveis(salvar=True)
        
        elif opcao == '5':
            print("\nVariáveis disponíveis:")
            print("  Temperatura: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11")
            print("  Precipitação: 12, 13, 14, 15, 16, 17, 18, 19")
            
            bio_num = int(input("\nDigite o número da variável (1-19): "))
            
            if 1 <= bio_num <= 19:
                viz.criar_mapa_interativo(bio_num)
            else:
                print("❌ Número inválido! Use valores entre 1 e 19.")
        
        elif opcao == '6':
            viz.gerar_relatorio_clima()
        
        print("\n✅ Visualização concluída!")
        print(f"📁 Resultados salvos em: abelhas_extensao/visualizacoes/\n")
        
    except FileNotFoundError as e:
        print(f"\n❌ Erro: {e}")
        print("\n💡 Certifique-se de que:")
        print("  1. Os dados climáticos foram baixados")
        print("  2. O shapefile do Brasil está presente")
        print("  3. Você está executando do diretório correto\n")
    
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
        import traceback
        traceback.print_exc()
    