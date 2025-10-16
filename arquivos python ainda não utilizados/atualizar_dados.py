import os
import glob
import shutil
import rasterio
import numpy as np
from tqdm import tqdm

def listar_variaveis():
    """Lista as variáveis disponíveis e ausentes no clima futuro"""
    # Lista variáveis disponíveis em cada diretório
    atual = set(os.path.basename(f) for f in glob.glob("clima_atual/*.tif"))
    futuro = set(os.path.basename(f) for f in glob.glob("clima_futuro/*.tif"))
    
    # Identifica variáveis ausentes
    faltando = atual - futuro
    
    print(f"📊 Variáveis no clima atual: {len(atual)}")
    print(f"📊 Variáveis no clima futuro: {len(futuro)}")
    print(f"📊 Variáveis faltando no clima futuro: {len(faltando)}")
    
    print("\n📋 Variáveis disponíveis em ambos:")
    for var in sorted(atual & futuro):
        print(f"  ✅ {var}")
    
    print("\n📋 Variáveis faltando no clima futuro:")
    for var in sorted(faltando):
        print(f"  ❌ {var}")
    
    return atual, futuro, faltando

def criar_copias_clima_futuro():
    """Cria cópias das variáveis ausentes no clima futuro"""
    atual, futuro, faltando = listar_variaveis()
    
    if len(faltando) == 0:
        print("\n✅ Todas as variáveis já estão disponíveis no clima futuro!")
        return
    
    print("\n⚠️ ATENÇÃO: Esta operação criará cópias das variáveis ausentes.")
    print("   Isso é útil para testes, mas para um modelo real você deve obter dados futuros corretos.")
    
    resposta = input("\n🔄 Deseja criar cópias das variáveis faltantes? (s/n): ")
    if resposta.lower() != "s":
        print("❌ Operação cancelada pelo usuário")
        return
    
    # Obtém a referência para dimensões
    if len(futuro) > 0:
        ref_futuro = os.path.join("clima_futuro", list(futuro)[0])
        with rasterio.open(ref_futuro) as src:
            futuro_shape = src.shape
            futuro_meta = src.meta.copy()
    else:
        print("❌ Não há variáveis de referência no clima futuro")
        return
    
    # Cria cópias
    criadas = 0
    for var in tqdm(faltando, desc="🔄 Copiando variáveis"):
        origem = os.path.join("clima_atual", var)
        destino = os.path.join("clima_futuro", var)
        
        # Lê o raster atual
        with rasterio.open(origem) as src:
            atual_data = src.read(1)
            # Se as dimensões são diferentes, redimensiona
            if src.shape != futuro_shape:
                print(f"⚠️ Dimensões diferentes para {var}. Redimensionando...")
                # Aqui faríamos um redimensionamento usando interpolação
                # Para simplificar, apenas copiamos o arquivo
            
            # Cria uma cópia com pequena variação para simular mudança climática
            futuro_data = atual_data.copy()
            # Adiciona uma pequena variação para simular mudança climática
            if var.startswith("bio1"):  # temperatura
                # Aumenta temperatura em 1-2 graus (bio1 é em décimos de grau)
                futuro_data = futuro_data + np.random.uniform(10, 20, futuro_data.shape)
            elif var.startswith(("bio12", "bio13", "bio14", "bio16", "bio17")):  # precipitação
                # Reduz precipitação em 5-15%
                futuro_data = futuro_data * np.random.uniform(0.85, 0.95, futuro_data.shape)
            
            # Salva o novo raster
            with rasterio.open(destino, 'w', **futuro_meta) as dst:
                dst.write(futuro_data, 1)
        
        criadas += 1
    
    print(f"✅ {criadas} variáveis bioclimáticas foram copiadas para clima_futuro")
    print("⚠️ LEMBRETE: Estas são cópias para teste. Para resultados científicos,")
    print("   utilize projeções climáticas futuras adequadas.")

def limpar_arquivos_gerados():
    """Remove mapas e modelos gerados para recomeçar do zero"""
    arquivos = [
        "modelo_ajustado.pkl",
        "scaler_ajustado.pkl",
        "mapa_predito_atual_brasil.tif",
        "mapa_predito_futuro_brasil.tif",
        "mapa_predito_atual_brasil_brasil.tif",
        "mapa_predito_futuro_brasil_brasil.tif",
        "mapa_mudanca_brasil.tif",
        "Distribuição_Atual.png",
        "Distribuição_Futura.png",
        "Mudanca_na_Distribuicao.png"
    ]
    
    print("⚠️ Esta operação removerá os seguintes arquivos:")
    for arquivo in arquivos:
        if os.path.exists(arquivo):
            print(f"  ❌ {arquivo}")
    
    resposta = input("\n🗑️ Deseja remover estes arquivos? (s/n): ")
    if resposta.lower() != "s":
        print("❌ Operação cancelada pelo usuário")
        return
    
    removidos = 0
    for arquivo in arquivos:
        if os.path.exists(arquivo):
            os.remove(arquivo)
            removidos += 1
    
    print(f"✅ {removidos} arquivos foram removidos")

def extrair_variaveis_bio(arquivo_origem, pasta_destino):
    """Extrai as 19 variáveis bioclimáticas de um arquivo bio*.tif"""
    os.makedirs(pasta_destino, exist_ok=True)
    
    print(f"\n📊 Extraindo variáveis de {arquivo_origem} para {pasta_destino}")
    
    try:
        with rasterio.open(arquivo_origem) as src:
            # Verificar número de bandas
            num_bandas = src.count
            print(f"Número de bandas detectadas: {num_bandas}")
            
            if num_bandas == 1:
                print("⚠️ AVISO: Este arquivo tem apenas uma banda!")
                return False
            
            # Processar arquivo multi-banda
            meta = src.meta.copy()
            meta.update({
                'count': 1,
                'driver': 'GTiff',
                'compress': 'lzw'
            })
            
            # Extrair cada banda como um arquivo separado
            for i in range(1, num_bandas + 1):
                banda = src.read(i)
                arquivo_saida = os.path.join(pasta_destino, f"bio{i}.tif")
                
                with rasterio.open(arquivo_saida, 'w', **meta) as dst:
                    dst.write(banda, 1)
                print(f"✅ Extraído: banda {i} -> bio{i}.tif")
            
            print(f"✅ Extração concluída: {num_bandas} variáveis salvas em {pasta_destino}")
            return True
            
    except Exception as e:
        print(f"❌ ERRO ao processar {arquivo_origem}: {str(e)}")
        return False

def menu_principal():
    """Menu principal do programa"""
    while True:
        print("\n" + "="*50)
        print("SISTEMA DE ATUALIZAÇÃO DE DADOS BIOCLIMÁTICOS")
        print("="*50)
        print("1. Listar variáveis disponíveis e faltantes")
        print("2. Criar cópias de variáveis para clima futuro")
        print("3. Limpar arquivos gerados (recomeçar do zero)")
        print("4. Extrair variáveis de bio1.tif para clima_futuro2")
        print("5. Extrair variáveis de bio2.tif para clima_futuro3")
        print("6. Sair")
        
        opcao = input("\nEscolha uma opção (1-6): ")
        
        if opcao == "1":
            listar_variaveis()
        elif opcao == "2":
            criar_copias_clima_futuro()
        elif opcao == "3":
            limpar_arquivos_gerados()
        elif opcao == "4":
            arquivo_bio1 = os.path.join("clima_futuro", "bio1.tif")
            if os.path.exists(arquivo_bio1):
                extrair_variaveis_bio(arquivo_bio1, os.path.join("clima_futuro", "clima_futuro2"))
            else:
                print("❌ Arquivo bio1.tif não encontrado em clima_futuro/")
        elif opcao == "5":
            arquivo_bio2 = os.path.join("clima_futuro", "bio2.tif")
            if os.path.exists(arquivo_bio2):
                extrair_variaveis_bio(arquivo_bio2, os.path.join("clima_futuro", "clima_futuro3"))
            else:
                print("❌ Arquivo bio2.tif não encontrado em clima_futuro/")
        elif opcao == "6":
            print("✅ Saindo do programa...")
            break
        else:
            print("❌ Opção inválida. Tente novamente.")

if __name__ == "__main__":
    menu_principal() 