import numpy as np
import rasterio
import joblib
import os
import glob
from tqdm import tqdm

def carregar_arquivos():
    """Carrega modelo, scaler e lista de variáveis"""
    if not os.path.exists("modelo_ajustado.pkl"):
        print("❌ Modelo ajustado não encontrado. Execute ajustar_modelo.py primeiro.")
        return None, None, None
    
    modelo = joblib.load("modelo_ajustado.pkl")
    scaler = joblib.load("scaler_ajustado.pkl")
    
    if os.path.exists("variaveis_usadas.pkl"):
        variaveis = joblib.load("variaveis_usadas.pkl")
    else:
        # Fallback para arquivo de texto
        with open("variaveis_usadas.txt", "r") as f:
            variaveis = [line.strip() for line in f.readlines()]
    
    return modelo, scaler, variaveis

def stack_rasters(raster_dir, selected_files):
    """Empilha múltiplos rasters em um único array 3D"""
    files = [os.path.join(raster_dir, f) for f in selected_files]
    arrays = []
    meta = None
    shape_ref = None
    
    # Primeiro, verificar as dimensões de todos os arquivos
    print(f"🔍 Verificando consistência dos arquivos em {raster_dir}...")
    for file in files:
        with rasterio.open(file) as src:
            if shape_ref is None:
                shape_ref = src.shape
                print(f"  📏 Dimensões esperadas: {shape_ref}")
            elif src.shape != shape_ref:
                print(f"  ⚠️ ALERTA: {os.path.basename(file)} tem dimensões diferentes: {src.shape}")
                return None, None
    
    # Se passou na verificação, empilhar os rasters
    for file in tqdm(files, desc=f"📚 Empilhando rasters em {raster_dir}"):
        with rasterio.open(file) as src:
            if meta is None:
                meta = src.meta.copy()
            data = src.read(1)
            # Verificar se há valores válidos
            if np.all(np.isnan(data)):
                print(f"  ⚠️ ALERTA: {os.path.basename(file)} contém apenas valores NaN")
            arrays.append(data)
    
    stacked = np.stack(arrays, axis=-1)
    return stacked, meta

def predizer_mapa(modelo, scaler, clima, meta, nome_saida):
    """Prediz probabilidade de ocorrência para cada pixel"""
    print(f"🗺️ Projetando distribuição para: {nome_saida}")
    
    # Preparar dados
    shape_original = clima.shape[:2]
    clima_reshaped = clima.reshape(-1, clima.shape[2])
    
    # Identificar valores válidos (não-NaN)
    valid_mask = ~np.isnan(clima_reshaped).any(axis=1)
    valid_indices = np.where(valid_mask)[0]
    
    # Inicializar mapa de predição com NaN
    pred_map = np.full(clima_reshaped.shape[0], np.nan)
    
    # Aplicar scaler e prever apenas para pixels válidos
    if len(valid_indices) > 0:
        print(f"✅ Predizendo {len(valid_indices)} pixels válidos...")
        valid_data = clima_reshaped[valid_indices]
        valid_data_scaled = scaler.transform(valid_data)
        
        # Processar em lotes para economizar memória
        batch_size = 100000
        n_batches = (len(valid_indices) + batch_size - 1) // batch_size
        
        for i in tqdm(range(n_batches), desc="🔍 Processando em lotes"):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(valid_indices))
            batch_data = valid_data_scaled[start_idx:end_idx]
            batch_preds = modelo.predict_proba(batch_data)[:, 1]
            pred_map[valid_indices[start_idx:end_idx]] = batch_preds
    
    # Redimensionar para formato original
    pred_map = pred_map.reshape(shape_original)
    
    # Salvar resultado
    print(f"💾 Salvando mapa em: {nome_saida}")
    with rasterio.open(nome_saida, "w", **meta) as dst:
        dst.write(pred_map, 1)
    
    return pred_map

def gerar_mapas():
    # Carregar modelo e parâmetros
    print("🔍 Carregando modelo ajustado...")
    modelo, scaler, variaveis = carregar_arquivos()
    
    if modelo is None:
        return
    
    print(f"✅ Modelo carregado com sucesso. Usando {len(variaveis)} variáveis bioclimáticas:")
    for var in variaveis:
        print(f"  - {var}")
    
    # Empilhar clima atual
    print("\n📚 Empilhando clima atual...")
    clima_atual, meta_atual = stack_rasters("clima_atual", variaveis)
    if clima_atual is None:
        print("❌ Erro ao empilhar clima atual. Verifique as dimensões dos arquivos.")
        return
    print(f"✅ Clima atual empilhado: {clima_atual.shape}")
    
    # Escolher qual conjunto de dados futuros usar
    print("\n🔄 Escolha o conjunto de dados futuros:")
    print("1. clima_futuro2 (variáveis do bio1.tif)")
    print("2. clima_futuro3 (variáveis do bio2.tif)")
    opcao = input("Opção (1 ou 2): ").strip()
    
    if opcao == "1":
        dir_futuro = "clima_futuro/clima_futuro2"
        sufixo = "futuro2"
    elif opcao == "2":
        dir_futuro = "clima_futuro/clima_futuro3"
        sufixo = "futuro3"
    else:
        print("❌ Opção inválida!")
        return
    
    # Empilhar clima futuro
    print(f"\n📚 Empilhando clima futuro de {dir_futuro}...")
    clima_futuro, meta_futuro = stack_rasters(dir_futuro, variaveis)
    if clima_futuro is None:
        print("❌ Erro ao empilhar clima futuro. Verifique as dimensões dos arquivos.")
        return
    print(f"✅ Clima futuro empilhado: {clima_futuro.shape}")
    
    # Verificar se as dimensões são compatíveis
    if clima_atual.shape != clima_futuro.shape:
        print(f"❌ Erro: Dimensões incompatíveis entre clima atual ({clima_atual.shape}) e futuro ({clima_futuro.shape})")
        return
    
    # Predizer mapas
    mapa_atual = predizer_mapa(modelo, scaler, clima_atual, meta_atual, f"mapa_predito_atual_{sufixo}.tif")
    mapa_futuro = predizer_mapa(modelo, scaler, clima_futuro, meta_futuro, f"mapa_predito_{sufixo}.tif")
    
    # Calcular mudança
    if mapa_atual is not None and mapa_futuro is not None:
        print("📊 Calculando mapa de mudança (futuro - atual)...")
        mapa_mudanca = mapa_futuro - mapa_atual
        with rasterio.open(f"mapa_mudanca_{sufixo}.tif", "w", **meta_atual) as dst:
            dst.write(mapa_mudanca, 1)
    
    print("\n✅ Mapas gerados com sucesso! Para continuar o fluxo:")
    print(f"1. Execute: python mascarar_brasil.py (usando os mapas com sufixo _{sufixo})")
    print("2. Execute: python visualizar_mapas.py")

if __name__ == "__main__":
    gerar_mapas() 