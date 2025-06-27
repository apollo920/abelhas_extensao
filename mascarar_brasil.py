import rasterio
from rasterio.mask import mask
import geopandas as gpd
import os
import glob
from tqdm import tqdm

def aplicar_mascara(mapa_path, shapefile_brasil, saida_path):
    print(f"🗺️ Aplicando máscara ao arquivo: {mapa_path}")

    # Verifica se os arquivos existem
    if not os.path.exists(mapa_path):
        print(f"❌ Arquivo de mapa não encontrado: {mapa_path}")
        return False
    if not os.path.exists(shapefile_brasil):
        print(f"❌ Shapefile não encontrado: {shapefile_brasil}")
        return False

    # Lê o shapefile do Brasil
    try:
        print(f"📂 Lendo shapefile: {shapefile_brasil}")
        brasil = gpd.read_file(shapefile_brasil)
        geoms = brasil.geometry.values
        print(f"✅ Shapefile carregado com {len(brasil)} feições")
    except Exception as e:
        print(f"❌ Erro ao ler o shapefile: {str(e)}")
        return False

    # Abre o raster de entrada
    try:
        print(f"📊 Lendo raster: {mapa_path}")
        with rasterio.open(mapa_path) as src:
            print(f"📏 Dimensões do raster: {src.width}x{src.height}")
            print("🔍 Aplicando máscara...")
            out_image, out_transform = mask(src, geoms, crop=True)
            out_meta = src.meta.copy()
    except Exception as e:
        print(f"❌ Erro ao processar o raster: {str(e)}")
        return False

    # Atualiza os metadados
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
        "nodata": src.nodata or -9999
    })

    # Salva o novo arquivo mascarado
    try:
        with rasterio.open(saida_path, "w", **out_meta) as dest:
            dest.write(out_image)
        print(f"✅ Mapa salvo com máscara: {saida_path}")
        return True
    except Exception as e:
        print(f"❌ Erro ao salvar o resultado: {str(e)}")
        return False

def detectar_mapas_gerados():
    """Detecta automaticamente os mapas gerados pelo gerar_mapas_ajustados.py"""
    padroes = [
        "mapa_predito_atual_*.tif",
        "mapa_predito_futuro*.tif", 
        "mapa_mudanca_*.tif"
    ]
    
    arquivos_encontrados = []
    for padrao in padroes:
        arquivos = glob.glob(padrao)
        arquivos_encontrados.extend(arquivos)
    
    # Remove duplicatas e ordena
    return sorted(list(set(arquivos_encontrados)))

if __name__ == "__main__":
    # Caminho para o shapefile
    shapefile_brasil = os.path.join("BR_UF_2024", "BR_UF_2024.shp")
    
    # Detectar mapas gerados automaticamente
    arquivos = detectar_mapas_gerados()
    
    if not arquivos:
        print("❌ Nenhum mapa gerado encontrado!")
        print("Execute primeiro: python gerar_mapas_ajustados.py")
        exit(1)
    
    print(f"📋 Mapas encontrados para processar:")
    for arquivo in arquivos:
        print(f"  - {arquivo}")
    
    # Processa cada arquivo
    sucessos = 0
    for arquivo in tqdm(arquivos, desc="🗺️ Processando mapas"):
        # Criar nome de saída
        nome_base = os.path.splitext(arquivo)[0]
        saida = f"{nome_base}_brasil.tif"
        
        if aplicar_mascara(arquivo, shapefile_brasil, saida):
            sucessos += 1
        print()  # Linha em branco para separar
    
    # Resumo
    print(f"✅ Concluído: {sucessos}/{len(arquivos)} mapas processados com sucesso")
    
    if sucessos > 0:
        print("\n📂 Arquivos gerados com máscara do Brasil:")
        arquivos_brasil = glob.glob("*_brasil.tif")
        for arquivo in sorted(arquivos_brasil):
            print(f"  ✅ {arquivo}")
        
        print("\n🎯 Próximos passos:")
        print("1. Execute: python visualizar_mapas.py")
