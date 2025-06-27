import matplotlib.pyplot as plt
import rasterio
import numpy as np
import os
import glob

def detectar_mapas_brasil():
    """Detecta automaticamente os mapas mascarados do Brasil"""
    mapas_encontrados = {}
    
    # Procurar mapas de distribuição atual
    atual_files = glob.glob("mapa_predito_atual_*_brasil.tif")
    if atual_files:
        mapas_encontrados["atual"] = atual_files[0]
    
    # Procurar mapas de distribuição futura
    futuro_files = glob.glob("mapa_predito_futuro*_brasil.tif")
    if futuro_files:
        mapas_encontrados["futuro"] = futuro_files[0]
    
    # Procurar mapas de mudança
    mudanca_files = glob.glob("mapa_mudanca_*_brasil.tif")
    if mudanca_files:
        mapas_encontrados["mudanca"] = mudanca_files[0]
    
    return mapas_encontrados

def visualizar_mapa(tif_path, titulo, salvar=True, escala_padronizada=True):
    with rasterio.open(tif_path) as src:
        data = src.read(1).astype(float)

        # Trata valores nodata
        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        # Obtém valores mínimo e máximo reais para estatísticas
        valid_data = data[~np.isnan(data)]
        if len(valid_data) == 0:
            print(f"❌ Nenhum dado válido encontrado em {titulo}")
            return
            
        v_min_real = np.nanmin(valid_data)
        v_max_real = np.nanmax(valid_data)
        
        # Calcula estatísticas básicas
        mean = np.nanmean(valid_data)
        std = np.nanstd(valid_data)
        
        print(f"📊 Estatísticas para {titulo}:")
        print(f"   - Min: {v_min_real:.4f}, Max: {v_max_real:.4f}, Média: {mean:.4f}, Desvio: {std:.4f}")
        
        # Define escala de visualização
        if escala_padronizada:
            if "mudança" in titulo.lower() or "mudanca" in titulo.lower():
                # Para mapas de mudança: escala simétrica baseada no máximo absoluto global
                abs_max = max(abs(v_min_real), abs(v_max_real))
                # Usar uma escala padrão de -0.8 a +0.8 para mudanças
                v_min, v_max = -0.8, 0.8
                cmap = plt.cm.RdBu_r  # Vermelho = perda, Azul = ganho
                label = 'Mudança na Probabilidade'
                print(f"   📏 Escala padronizada para mudança: {v_min} a {v_max}")
            else:
                # Para mapas de probabilidade: escala padronizada 0-1
                v_min, v_max = 0.0, 1.0
                cmap = plt.cm.viridis
                label = 'Probabilidade de Adequabilidade'
                print(f"   📏 Escala padronizada para probabilidade: {v_min} a {v_max}")
        else:
            # Escala dinâmica baseada nos dados
            v_min, v_max = v_min_real, v_max_real
            if "mudança" in titulo.lower() or "mudanca" in titulo.lower():
                cmap = plt.cm.RdBu_r
                label = 'Mudança na Probabilidade'
            else:
                cmap = plt.cm.viridis
                label = 'Probabilidade de Adequabilidade'
            print(f"   📏 Escala dinâmica: {v_min:.4f} a {v_max:.4f}")
        
        # Configura visualização
        plt.figure(figsize=(12, 10))
        
        # Cria o mapa
        im = plt.imshow(data, cmap=cmap, vmin=v_min, vmax=v_max)
        
        # Adiciona título e legenda
        plt.title(titulo, fontsize=16, pad=20)
        plt.axis('off')
        cbar = plt.colorbar(im, label=label, shrink=0.7)
        cbar.ax.tick_params(labelsize=12)
        
        # Adiciona informações sobre valores reais e escala
        info_text = f"Valores reais: {v_min_real:.4f} a {v_max_real:.4f}\nMédia: {mean:.4f} ± {std:.4f}"
        if escala_padronizada:
            info_text += f"\nEscala padronizada: {v_min} a {v_max}"
        
        plt.annotate(info_text, 
                     xy=(0.02, 0.02), xycoords='figure fraction', 
                     fontsize=11, color='black', backgroundcolor='white',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Salva o mapa em alta resolução para o trabalho acadêmico
        if salvar:
            nome_arquivo = f"{titulo.replace(' ', '_').replace('(', '').replace(')', '')}.png"
            plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
            print(f"💾 Mapa salvo como: {nome_arquivo}")
        
        plt.show()

def visualizar_mapa_diferenca(atual_path, futuro_path, escala_padronizada=True):
    """Cria um mapa de diferença entre distribuição futura e atual"""
    
    if not os.path.exists(atual_path) or not os.path.exists(futuro_path):
        print("❌ Arquivos de mapa não encontrados para calcular diferença")
        return
    
    with rasterio.open(atual_path) as src_atual, rasterio.open(futuro_path) as src_futuro:
        data_atual = src_atual.read(1).astype(float)
        data_futuro = src_futuro.read(1).astype(float)
        
        # Trata valores nodata
        if src_atual.nodata is not None:
            data_atual[data_atual == src_atual.nodata] = np.nan
        if src_futuro.nodata is not None:
            data_futuro[data_futuro == src_futuro.nodata] = np.nan
        
        # Calcula diferença
        data_diff = data_futuro - data_atual
        
        # Configurar visualização
        plt.figure(figsize=(12, 10))
        
        # Paleta divergente para mudanças (vermelho = perda, azul = ganho)
        cmap = plt.cm.RdBu_r
        
        # Determinar limite para visualização
        valid_diff = data_diff[~np.isnan(data_diff)]
        if len(valid_diff) == 0:
            print("❌ Nenhum dado válido para calcular diferença")
            return
        
        v_min_real = np.nanmin(data_diff)
        v_max_real = np.nanmax(data_diff)
        
        if escala_padronizada:
            # Escala padronizada para mudanças
            v_min, v_max = -0.8, 0.8
        else:
            # Escala dinâmica simétrica
            abs_max = max(abs(v_min_real), abs(v_max_real))
            v_min, v_max = -abs_max, abs_max
        
        im = plt.imshow(data_diff, cmap=cmap, vmin=v_min, vmax=v_max)
        
        plt.title("Mudança na Distribuição (Futuro - Atual)", fontsize=16, pad=20)
        plt.axis('off')
        cbar = plt.colorbar(im, label='Mudança na Probabilidade', shrink=0.7)
        cbar.ax.tick_params(labelsize=12)
        
        # Calcular estatísticas da diferença
        mean_diff = np.nanmean(data_diff)
        perc_loss = np.sum(data_diff < -0.1) / np.sum(~np.isnan(data_diff)) * 100
        perc_gain = np.sum(data_diff > 0.1) / np.sum(~np.isnan(data_diff)) * 100
        
        # Adiciona informações sobre valores
        info_text = (f"Valores reais: {v_min_real:.4f} a {v_max_real:.4f}\n"
                     f"Média da mudança: {mean_diff:.4f}\n"
                     f"Área com perda (< -0.1): {perc_loss:.1f}%\n"
                     f"Área com ganho (> 0.1): {perc_gain:.1f}%")
        
        if escala_padronizada:
            info_text += f"\nEscala padronizada: {v_min} a {v_max}"
        
        plt.annotate(info_text, xy=(0.02, 0.02), xycoords='figure fraction', 
                     fontsize=11, color='black', backgroundcolor='white',
                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.tight_layout()
        
        # Salva o mapa
        plt.savefig("Mudanca_na_Distribuicao.png", dpi=300, bbox_inches='tight')
        print(f"💾 Mapa de mudança salvo como: Mudanca_na_Distribuicao.png")
        
        plt.show()

if __name__ == "__main__":
    # Detectar mapas automaticamente
    mapas = detectar_mapas_brasil()
    
    if not mapas:
        print("❌ Nenhum mapa mascarado encontrado!")
        print("Execute primeiro:")
        print("1. python gerar_mapas_ajustados.py")
        print("2. python mascarar_brasil.py")
        exit(1)
    
    print("📂 Mapas detectados:")
    for tipo, arquivo in mapas.items():
        print(f"  - {tipo}: {arquivo}")
    
    # Perguntar sobre padronização
    print("\n🎨 Escolha o tipo de escala para visualização:")
    print("1. Escala padronizada (0-1 para probabilidades, -0.8 a +0.8 para mudanças)")
    print("2. Escala dinâmica (baseada nos valores dos dados)")
    opcao_escala = input("Opção (1 ou 2) [padrão=1]: ").strip() or "1"
    
    escala_padronizada = opcao_escala == "1"
    
    if escala_padronizada:
        print("✅ Usando escalas padronizadas para comparação científica")
    else:
        print("✅ Usando escalas dinâmicas para visualização detalhada")
    
    # Visualizar mapas de distribuição
    if "atual" in mapas:
        print(f"\n📍 Exibindo: Distribuição Atual")
        # Extrair sufixo do arquivo para identificar o cenário
        nome_arquivo = os.path.basename(mapas["atual"])
        if "futuro2" in nome_arquivo:
            sufixo = " (cenário futuro2)"
        elif "futuro3" in nome_arquivo:
            sufixo = " (cenário futuro3)"
        else:
            sufixo = ""
        visualizar_mapa(mapas["atual"], f"Distribuição Atual{sufixo}", escala_padronizada=escala_padronizada)
    
    if "futuro" in mapas:
        print(f"\n📍 Exibindo: Distribuição Futura")
        # Extrair sufixo do arquivo para identificar o cenário
        nome_arquivo = os.path.basename(mapas["futuro"])
        if "futuro2" in nome_arquivo:
            sufixo = " (cenário futuro2)"
        elif "futuro3" in nome_arquivo:
            sufixo = " (cenário futuro3)"
        else:
            sufixo = ""
        visualizar_mapa(mapas["futuro"], f"Distribuição Futura{sufixo}", escala_padronizada=escala_padronizada)
    
    # Visualizar mapa de mudança se disponível
    if "mudanca" in mapas:
        print(f"\n📍 Exibindo: Mapa de Mudança")
        nome_arquivo = os.path.basename(mapas["mudanca"])
        if "futuro2" in nome_arquivo:
            sufixo = " (cenário futuro2)"
        elif "futuro3" in nome_arquivo:
            sufixo = " (cenário futuro3)"
        else:
            sufixo = ""
        visualizar_mapa(mapas["mudanca"], f"Mudança na Distribuição{sufixo}", escala_padronizada=escala_padronizada)
    
    # Se temos atual e futuro, calcular diferença manualmente se não existe mapa de mudança
    elif "atual" in mapas and "futuro" in mapas:
        print(f"\n📍 Calculando mapa de diferença entre distribuição futura e atual")
        visualizar_mapa_diferenca(mapas["atual"], mapas["futuro"], escala_padronizada=escala_padronizada)
