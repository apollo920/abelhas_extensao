import pandas as pd
import numpy as np
import rasterio
import os
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from tqdm import tqdm
from geopy.distance import geodesic

def obter_variaveis_disponiveis():
    """Identifica todas as variáveis bioclimáticas disponíveis em ambos cenários"""
    cli_atual = set(os.path.basename(f) for f in glob.glob("clima_atual/*.tif"))
    cli_futuro = set(os.path.basename(f) for f in glob.glob("clima_futuro/clima_futuro2/*.tif"))
    comuns = sorted(list(cli_atual & cli_futuro))
    print(f"✅ Variáveis em comum: {len(comuns)}")
    for var in comuns:
        print(f"  - {var}")

    # NOVO: Verifica se os arquivos futuros estão corretos e tem o mesmo formato
    try:
        with rasterio.open(os.path.join("clima_atual", comuns[0])) as src_atual:
            shape_atual = src_atual.shape
        with rasterio.open(os.path.join("clima_futuro", comuns[0])) as src_futuro:
            shape_futuro = src_futuro.shape
        
        if shape_atual != shape_futuro:
            print(f"⚠️ ALERTA: Arquivos atuais ({shape_atual}) e futuros ({shape_futuro}) têm dimensões diferentes!")
    except Exception as e:
        print(f"⚠️ Erro ao verificar dimensões dos arquivos: {str(e)}")
    
    return comuns

def stack_rasters(raster_dir, selected_files):
    """Empilha múltiplos rasters em um único array 3D"""
    files = [os.path.join(raster_dir, f) for f in selected_files]
    arrays = []
    meta = None
    for file in tqdm(files, desc=f"📚 Empilhando rasters em {raster_dir}"):
        with rasterio.open(file) as src:
            if meta is None:
                meta = src.meta.copy()
            arrays.append(src.read(1))
    stacked = np.stack(arrays, axis=-1)
    return stacked, meta

def extract_values(coords, raster_stack, meta):
    """Extrai valores de raster para cada coordenada"""
    results = []
    for lat, lon in tqdm(coords, desc="📌 Extraindo valores"):
        row, col = rasterio.transform.rowcol(meta['transform'], lon, lat)
        if 0 <= row < raster_stack.shape[0] and 0 <= col < raster_stack.shape[1]:
            results.append(raster_stack[row, col])
        else:
            results.append([np.nan]*raster_stack.shape[2])
    return np.array(results)

def generate_pseudo_absences(clima, meta, pres_coords, n_samples, min_dist_km=30):
    """Gera pontos de pseudo-ausência longe das presenças conhecidas"""
    absences = []
    attempts = 0
    max_attempts = n_samples * 30  # Aumentado para mais tentativas
    
    print(f"🎯 Gerando {n_samples} pseudo-ausências")
    with tqdm(total=n_samples, desc="🚫 Gerando pseudo-ausências") as pbar:
        while len(absences) < n_samples and attempts < max_attempts:
            # Estratégia para melhor distribuição espacial: dividir em quadrantes
            row = np.random.randint(0, clima.shape[0])
            col = np.random.randint(0, clima.shape[1])
            vals = clima[row, col]
            if np.isnan(vals).any():
                attempts += 1
                continue
            lon, lat = rasterio.transform.xy(meta['transform'], row, col)
            
            # Verifica distância mínima das presenças (agora 30km em vez de 20km)
            if all(geodesic((lat, lon), pc).km > min_dist_km for pc in pres_coords):
                absences.append(vals)
                pbar.update(1)
            attempts += 1
    
    return np.array(absences)

def ajustar_modelo():
    print("🔄 Iniciando ajuste do modelo para reduzir overfitting...")
    
    # Obter variáveis disponíveis
    variaveis = obter_variaveis_disponiveis()
    
    if len(variaveis) <= 2:
        print("⚠️ Apenas 2 variáveis bioclimáticas disponíveis. Para melhor desempenho, adicione mais variáveis.")
    
    # Ler ocorrências
    print("📊 Lendo dados de ocorrência...")
    df = pd.read_csv("ocorrencias.csv")
    coords = list(zip(df["latitude"], df["longitude"]))
    print(f"✅ Presenças: {len(coords)}")
    
    # Empilhar rasters
    print("📚 Empilhando rasters...")
    clima_atual, meta_atual = stack_rasters("clima_atual", variaveis)
    print(f"✅ Clima atual empilhado: {clima_atual.shape}")
    
    # Extrair valores para presenças
    presenças = extract_values(coords, clima_atual, meta_atual)
    
    # Remover presenças com NaN
    nan_mask = np.isnan(presenças).any(axis=1)
    if nan_mask.any():
        print(f"⚠️ Removendo {nan_mask.sum()} presenças com valores NaN")
        presenças = presenças[~nan_mask]
        presenca_coords = [coord for i, coord in enumerate(coords) if not nan_mask[i]]
    else:
        presenca_coords = coords
    
    # Gerar pseudo-ausências
    n_ausencias = len(presenças)
    ausencias = generate_pseudo_absences(clima_atual, meta_atual, presenca_coords, n_ausencias, min_dist_km=30)
    
    # Preparar dados para treinamento
    X = np.vstack((presenças, ausencias))
    y = np.array([1]*len(presenças) + [0]*len(ausencias))
    
    # Normalizar dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Dividir em treino/teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2)
    
    # Treinar modelo com parâmetros para prevenir overfitting
    print("🔧 Treinando modelo com parâmetros para reduzir overfitting...")
    
    # Parâmetros mais conservadores para evitar overfitting
    # - Aumentando min_samples_leaf ainda mais
    # - Limitando max_depth para evitar árvores muito profundas
    # - Reduzindo número de árvores para evitar memorização
    rf = RandomForestClassifier(
        n_estimators=50,          # Menos árvores (era 100)
        min_samples_leaf=30,      # Muito maior (era 10)
        max_depth=10,             # Mais limitado (era 15)
        max_features='sqrt',
        class_weight='balanced',
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    
    # Avaliar modelo
    test_acc = rf.score(X_test, y_test)
    print(f"✅ Acurácia no teste: {test_acc:.3f}")
    
    # Salvar modelo
    print("💾 Salvando modelo e scaler...")
    joblib.dump(rf, "modelo_ajustado.pkl")
    joblib.dump(scaler, "scaler_ajustado.pkl")
    joblib.dump(variaveis, "variaveis_usadas.pkl")
    
    # Salvar lista de variáveis
    with open("variaveis_usadas.txt", "w") as f:
        for var in variaveis:
            f.write(f"{var}\n")
    
    print("✅ Modelo ajustado! Para gerar os novos mapas, execute:")
    print("1. python gerar_mapas_ajustados.py")
    print("2. python mascarar_brasil.py")
    print("3. python visualizar_mapas.py")

if __name__ == "__main__":
    ajustar_modelo() 