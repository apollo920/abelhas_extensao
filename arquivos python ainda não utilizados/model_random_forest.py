import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel
import joblib
import glob
from tqdm import tqdm
from geopy.distance import geodesic
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance

def stack_rasters(raster_dir, selected_files=None):
    all_files = sorted(glob.glob(os.path.join(raster_dir, "*.tif")))
    if selected_files:
        files = [f for f in all_files if os.path.basename(f) in selected_files]
    else:
        files = all_files
    arrays = []
    meta = None
    for file in files:
        with rasterio.open(file) as src:
            if meta is None:
                meta = src.meta.copy()
            arrays.append(src.read(1))
    stacked = np.stack(arrays, axis=-1)
    return stacked, meta, [os.path.basename(f) for f in files]

def extract_raster_values(coords, raster_stack, meta):
    results = []
    for lat, lon in tqdm(coords, desc="📌 Extraindo valores ambientais"):
        row, col = rasterio.transform.rowcol(meta['transform'], lon, lat)
        if 0 <= row < raster_stack.shape[0] and 0 <= col < raster_stack.shape[1]:
            results.append(raster_stack[row, col])
        else:
            results.append([np.nan]*raster_stack.shape[2])
    return np.array(results)

def generate_pseudo_absences(clima, meta, pres_coords, n_samples, min_dist_km=20):
    absences = []
    attempts = 0
    max_attempts = n_samples * 20
    
    print(f"🎯 Meta: gerar {n_samples} pseudo-ausências")
    with tqdm(total=n_samples, desc="🚫 Gerando pseudo-ausências") as pbar:
        while len(absences) < n_samples and attempts < max_attempts:
            row = np.random.randint(0, clima.shape[0])
            col = np.random.randint(0, clima.shape[1])
            vals = clima[row, col]
            if np.isnan(vals).any():
                attempts += 1
                continue
            lon, lat = rasterio.transform.xy(meta['transform'], row, col)
            if all(geodesic((lat, lon), pc).km > min_dist_km for pc in pres_coords):
                absences.append(vals)
                pbar.update(1)
            attempts += 1
            pbar.set_postfix({'tentativas': attempts, 'taxa_sucesso': f"{len(absences)/attempts*100:.1f}%"})
    
    if len(absences) < n_samples:
        print(f"⚠️  Apenas {len(absences)} pseudo-ausências foram geradas após {attempts} tentativas")
    else:
        print(f"✅ {len(absences)} pseudo-ausências geradas com sucesso!")
    
    return np.array(absences)

# Nova função para realizar validação espacial por blocos
def spatial_block_cv(coords, n_folds=5):
    """
    Divide os dados em blocos espaciais para validação cruzada.
    Isso ajuda a evitar vazamento espacial entre treino e teste.
    """
    # Obter limites geográficos
    lats, lons = zip(*coords)
    lat_min, lat_max = min(lats), max(lats)
    lon_min, lon_max = min(lons), max(lons)
    
    # Criar blocos espaciais
    lat_bins = np.linspace(lat_min, lat_max, int(np.sqrt(n_folds))+1)
    lon_bins = np.linspace(lon_min, lon_max, int(np.sqrt(n_folds))+1)
    
    # Atribuir cada ponto a um bloco
    blocks = []
    for i in range(len(coords)):
        lat, lon = coords[i]
        lat_idx = np.digitize(lat, lat_bins) - 1
        lon_idx = np.digitize(lon, lon_bins) - 1
        block_id = lat_idx * (len(lon_bins)-1) + lon_idx
        blocks.append(block_id)
    
    return np.array(blocks)

# Nova função para avaliar importância de features
def plot_feature_importance(model, feature_names, output_file="feature_importance.png"):
    """
    Gera gráfico de importância das variáveis para interpretação do modelo
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Importância das Variáveis Bioclimáticas")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig(output_file)
    
    # Também salva como texto
    with open("importancia_variaveis.txt", "w") as f:
        for i in indices:
            f.write(f"{feature_names[i]}: {importances[i]:.4f}\n")

def main():
    ocorrencias_csv = "ocorrencias.csv"
    rasters_atual = "clima_atual"
    rasters_futuro = "clima_futuro"

    print("🔍 Lendo dados de ocorrência...")
    df = pd.read_csv(ocorrencias_csv)
    df = df.dropna(subset=["longitude", "latitude"])
    coords = list(zip(df["latitude"], df["longitude"]))
    print(f"Presenças: {len(coords)}")

    print("📚 Verificando variáveis em comum...")
    arquivos_atual = {os.path.basename(f) for f in glob.glob(os.path.join(rasters_atual, "*.tif"))}
    arquivos_futuro = {os.path.basename(f) for f in glob.glob(os.path.join(rasters_futuro, "*.tif"))}
    arquivos_comuns = sorted(list(arquivos_atual & arquivos_futuro))
    if len(arquivos_comuns) < 1:
        raise ValueError("❌ Nenhuma variável em comum entre clima atual e futuro.")
    print(f"✅ Variáveis em comum: {len(arquivos_comuns)} -> {arquivos_comuns}")

    # Salva as variáveis utilizadas
    with open("variaveis_usadas.txt", "w") as f:
        for nome in arquivos_comuns:
            f.write(f"{nome}\n")

    print("📚 Empilhando clima atual...")
    clima_atual, meta_atual, nomes_variaveis = stack_rasters(rasters_atual, arquivos_comuns)
    print("✅ Clima atual empilhado:", clima_atual.shape)

    print("📌 Extraindo valores ambientais para presenças...")
    presenças = extract_raster_values(coords, clima_atual, meta_atual)

    # Verificando e removendo presenças com valores NaN
    nan_mask = np.isnan(presenças).any(axis=1)
    if nan_mask.any():
        print(f"⚠️ Removendo {nan_mask.sum()} presenças com valores NaN")
        presenças = presenças[~nan_mask]
        presenca_coords = [coord for i, coord in enumerate(coords) if not nan_mask[i]]
    else:
        presenca_coords = coords
    
    print(f"✅ Presenças válidas: {len(presenças)}")

    print("🚫 Gerando pseudo-ausências com distância mínima de 20 km...")
    # Balanceamento entre presenças e ausências (1:1 para grandes volumes de dados)
    n_ausencias = len(presenças)  # Proporção 1:1 para maior balanceamento
    ausencias = generate_pseudo_absences(clima_atual, meta_atual, presenca_coords, n_ausencias, min_dist_km=20)

    print("📊 Preparando dados para modelagem...")
    X = np.vstack((presenças, ausencias))
    y = np.array([1]*len(presenças) + [0]*len(ausencias))
    
    # Normalização dos dados (importante para variáveis em escalas diferentes)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Divisão treino/teste considerando a estrutura espacial (se tiver coordenadas suficientes)
    print("🧮 Dividindo dados para treinamento e teste...")
    if len(presenca_coords) >= 50:  # Se tiver coordenadas suficientes para blocos espaciais
        todos_coords = presenca_coords + [(0, 0)] * len(ausencias)  # Pseudo-ausências com coords dummy
        blocks = spatial_block_cv(todos_coords)
        block_ids = np.unique(blocks)
        
        # Garantir que há pelo menos 2 blocos diferentes para teste
        if len(block_ids) < 5:
            print("⚠️ Poucos blocos espaciais distintos, usando divisão aleatória...")
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)
        else:
            # Calcular número mínimo de blocos para teste (pelo menos 20% dos dados)
            min_test_size = int(0.2 * len(X_scaled))
            test_blocks = []
            test_mask = np.zeros(len(blocks), dtype=bool)
            
            # Adicionar blocos ao conjunto de teste até atingir o tamanho mínimo
            while np.sum(test_mask) < min_test_size and len(test_blocks) < len(block_ids):
                # Escolher bloco aleatório que ainda não está no conjunto de teste
                remaining_blocks = [b for b in block_ids if b not in test_blocks]
                if not remaining_blocks:
                    break
                new_block = np.random.choice(remaining_blocks)
                test_blocks.append(new_block)
                
                # Atualizar máscara de teste
                new_mask = np.isin(blocks, test_blocks)
                test_mask = new_mask
            
            # Verificar se conseguimos amostras suficientes para teste
            if np.sum(test_mask) < min_test_size * 0.5:  # Se menos da metade do desejado
                print("⚠️ Blocos espaciais não forneceram amostras suficientes, usando divisão aleatória...")
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)
            else:
                X_train, X_test = X_scaled[~test_mask], X_scaled[test_mask]
                y_train, y_test = y[~test_mask], y[test_mask]
                print(f"✅ Divisão por blocos espaciais: {len(X_train)} treino, {len(X_test)} teste")
    else:
        # Fallback para divisão aleatória estratificada tradicional
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

    # Configurar GridSearchCV para otimização de hiperparâmetros
    print("🔍 Otimizando hiperparâmetros do modelo...")
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    
    # Usando uma versão simplificada para execução mais rápida
    param_grid_simples = {
        'n_estimators': [100],
        'max_depth': [None, 15],
        'min_samples_split': [5],
        'min_samples_leaf': [2],
        'max_features': ['sqrt']
    }
    
    # Escolha qual grid usar com base no tamanho do conjunto de dados
    grid = param_grid_simples if len(X_train) > 1000 else param_grid
    
    rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=grid,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Melhores parâmetros encontrados: {grid_search.best_params_}")
    model = grid_search.best_estimator_
    
    # Avaliação no conjunto de teste
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)
    test_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"✅ Desempenho no conjunto de teste:")
    print(f"   AUC: {test_auc:.3f}")
    print("\nMatriz de confusão:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred))
    
    # Calcular importância das variáveis
    print("📊 Calculando importância das variáveis...")
    plot_feature_importance(model, nomes_variaveis)
    
    # Salvar modelo e scaler para uso posterior
    joblib.dump(model, "modelo_random_forest.pkl")
    joblib.dump(scaler, "scaler.pkl")
    
    # Projetar para clima atual
    print("🗺️ Projetando distribuição atual...")
    # Redimensionar e normalizar os dados antes da predição
    clima_atual_reshaped = clima_atual.reshape(-1, clima_atual.shape[2])
    # Criar máscara para valores válidos (não-NaN)
    valid_mask = ~np.isnan(clima_atual_reshaped).any(axis=1)
    valid_indices = np.where(valid_mask)[0]
    
    # Inicializar mapa de predição com NaN
    pred_map = np.full(clima_atual_reshaped.shape[0], np.nan)
    
    # Aplicar scaler e prever apenas para pixels válidos
    if len(valid_indices) > 0:
        valid_data = clima_atual_reshaped[valid_indices]
        valid_data_scaled = scaler.transform(valid_data)
        valid_preds = model.predict_proba(valid_data_scaled)[:, 1]
        pred_map[valid_indices] = valid_preds
    
    # Redimensionar para formato original
    pred_map = pred_map.reshape(clima_atual.shape[0], clima_atual.shape[1])
    
    # Salvar mapa
    with rasterio.open("mapa_predito_atual_brasil.tif", "w", **meta_atual) as dst:
        dst.write(pred_map, 1)
    
    # Processar clima futuro
    print("📦 Empilhando clima futuro...")
    clima_futuro, meta_futuro, _ = stack_rasters(rasters_futuro, arquivos_comuns)
    
    print("🗺️ Projetando distribuição futura...")
    # Mesmo processo de predição para clima futuro
    clima_futuro_reshaped = clima_futuro.reshape(-1, clima_futuro.shape[2])
    valid_mask_futuro = ~np.isnan(clima_futuro_reshaped).any(axis=1)
    valid_indices_futuro = np.where(valid_mask_futuro)[0]
    
    pred_futuro = np.full(clima_futuro_reshaped.shape[0], np.nan)
    
    if len(valid_indices_futuro) > 0:
        valid_data_futuro = clima_futuro_reshaped[valid_indices_futuro]
        valid_data_futuro_scaled = scaler.transform(valid_data_futuro)
        valid_preds_futuro = model.predict_proba(valid_data_futuro_scaled)[:, 1]
        pred_futuro[valid_indices_futuro] = valid_preds_futuro
    
    pred_futuro = pred_futuro.reshape(clima_futuro.shape[0], clima_futuro.shape[1])
    
    with rasterio.open("mapa_predito_futuro_brasil.tif", "w", **meta_futuro) as dst:
        dst.write(pred_futuro, 1)
    
    # Calcular mudança entre atual e futuro
    if pred_map.shape == pred_futuro.shape:
        print("📊 Calculando mapa de mudança (futuro - atual)...")
        mapa_mudanca = pred_futuro - pred_map
        with rasterio.open("mapa_mudanca_brasil.tif", "w", **meta_atual) as dst:
            dst.write(mapa_mudanca, 1)
    
    print("✅ Modelagem concluída!")

if __name__ == "__main__":
    main()
