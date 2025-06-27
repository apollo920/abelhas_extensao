import rasterio
import numpy as np
import matplotlib.pyplot as plt
import os

def analisar_mapa(tif_path, titulo):
    if not os.path.exists(tif_path):
        print(f"❌ Arquivo não encontrado: {tif_path}")
        return None
    
    print(f"📊 Analisando: {titulo}")
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(float)
        
        # Trata valores nodata
        if src.nodata is not None:
            data[data == src.nodata] = np.nan
        
        # Estatísticas básicas
        dados_validos = data[~np.isnan(data)]
        if len(dados_validos) == 0:
            print("❌ Nenhum dado válido encontrado no mapa")
            return None
        
        min_val = np.nanmin(data)
        max_val = np.nanmax(data)
        mean_val = np.nanmean(data)
        
        print(f"   - Valor mínimo: {min_val:.4f}")
        print(f"   - Valor máximo: {max_val:.4f}")
        print(f"   - Média: {mean_val:.4f}")
        
        # Calcular histograma
        valores, bins = np.histogram(dados_validos, bins=10, range=(0, 1))
        for i, (inicio, fim) in enumerate(zip(bins[:-1], bins[1:])):
            perc = valores[i] / len(dados_validos) * 100
            print(f"   - {inicio:.1f}-{fim:.1f}: {perc:.1f}% dos pixels")
        
        # Identificar overfitting
        if mean_val > 0.9:
            print("⚠️ ALERTA: Média muito alta, possível overfitting (modelo prevê presença em excesso)")
        elif mean_val < 0.1:
            print("⚠️ ALERTA: Média muito baixa, possível overfitting (modelo prevê ausência em excesso)")
        
        return data

def comparar_mapas():
    # Mapas após máscara
    atual = analisar_mapa("mapa_predito_atual_brasil_brasil.tif", "Distribuição Atual")
    futuro = analisar_mapa("mapa_predito_futuro_brasil_brasil.tif", "Distribuição Futura")
    
    # Também verificar os originais
    print("\n📊 Verificando mapas originais (antes da máscara):")
    atual_orig = analisar_mapa("mapa_predito_atual_brasil.tif", "Original Atual")
    futuro_orig = analisar_mapa("mapa_predito_futuro_brasil.tif", "Original Futuro")
    
    if atual is not None and futuro is not None:
        # Verificar diferença
        diff = futuro - atual
        diff_valid = diff[~np.isnan(diff)]
        if len(diff_valid) > 0:
            print("\n📈 Análise de mudança (Futuro - Atual):")
            print(f"   - Média da diferença: {np.mean(diff_valid):.4f}")
            print(f"   - Diferença máxima positiva: {np.max(diff_valid):.4f}")
            print(f"   - Diferença máxima negativa: {np.min(diff_valid):.4f}")
    
    print("\n💡 Recomendações:")
    print("1. Se todos os mapas mostram probabilidade 1.0 (ou próximo), o modelo está com overfitting severo.")
    print("2. Possíveis soluções:")
    print("   - Ajustar hiperparâmetros do Random Forest (aumentar min_samples_leaf)")
    print("   - Incluir mais variáveis ambientais (apenas 2 foram usadas)")
    print("   - Verificar geração de pseudo-ausências")
    print("   - Verificar dados de presença (possíveis duplicatas ou viés espacial)")

if __name__ == "__main__":
    comparar_mapas() 