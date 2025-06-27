#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import shutil
import rasterio
import numpy as np
from tqdm import tqdm
from rasterio.warp import calculate_default_transform, reproject, Resampling

def extrair_variaveis_bioclimaticas(arquivo_tif, diretorio_saida):
    """
    Extrai as 19 variáveis bioclimáticas de um único arquivo .tif do WorldClim
    e salva cada uma como um arquivo separado.
    
    Args:
        arquivo_tif: Caminho para o arquivo .tif multi-banda do WorldClim
        diretorio_saida: Diretório onde salvar os arquivos individuais
    """
    os.makedirs(diretorio_saida, exist_ok=True)
    
    print(f"Processando arquivo: {arquivo_tif}")
    
    try:
        with rasterio.open(arquivo_tif) as src:
            # Verificar número de bandas
            num_bandas = src.count
            print(f"Número de bandas detectadas: {num_bandas}")
            
            if num_bandas == 1:
                print("AVISO: Este arquivo tem apenas uma banda. Pode não ser um arquivo multi-variável.")
                
                # Extrair nome da variável do nome do arquivo
                nome_arquivo = os.path.basename(arquivo_tif)
                if "bio" in nome_arquivo.lower():
                    # Tentar extrair número da variável bioclimática
                    partes = nome_arquivo.lower().split("bio")
                    if len(partes) > 1:
                        num_var = ''.join(filter(str.isdigit, partes[1]))
                        if num_var:
                            novo_nome = f"bio{num_var}.tif"
                            arquivo_saida = os.path.join(diretorio_saida, novo_nome)
                            
                            # Copiar arquivo
                            shutil.copy(arquivo_tif, arquivo_saida)
                            print(f"Copiado: {nome_arquivo} -> {novo_nome}")
                        else:
                            print(f"Não foi possível extrair número da variável de {nome_arquivo}")
                else:
                    print(f"Arquivo não parece ser uma variável bioclimática: {nome_arquivo}")
            else:
                # Processar arquivo multi-banda
                meta = src.meta.copy()
                
                # Atualizar metadados para arquivos de saída (uma banda)
                meta.update({
                    'count': 1,
                    'driver': 'GTiff',
                    'compress': 'lzw'
                })
                
                # Extrair cada banda como um arquivo separado
                for i in range(1, num_bandas + 1):
                    banda = src.read(i)
                    
                    # Nome do arquivo de saída
                    arquivo_saida = os.path.join(diretorio_saida, f"bio{i}.tif")
                    
                    # Salvar banda como arquivo separado
                    with rasterio.open(arquivo_saida, 'w', **meta) as dst:
                        dst.write(banda, 1)
                    
                    print(f"Extraído: banda {i} -> bio{i}.tif")
                
                print(f"Extração concluída: {num_bandas} variáveis salvas em {diretorio_saida}")
    
    except Exception as e:
        print(f"ERRO ao processar {arquivo_tif}: {str(e)}")

def padronizar_nomes_arquivos(diretorio_origem, diretorio_destino):
    """
    Padroniza os nomes dos arquivos .tif para o formato bio1.tif, bio2.tif, etc.
    
    Args:
        diretorio_origem: Diretório com os arquivos originais
        diretorio_destino: Diretório onde salvar os arquivos renomeados
    """
    os.makedirs(diretorio_destino, exist_ok=True)
    
    print(f"Padronizando nomes dos arquivos em {diretorio_origem}")
    
    # Padrões de nomes conhecidos do WorldClim
    padroes = [
        # WorldClim v2.1
        {"prefixo": "wc2.1_", "separador": "bio"},
        # CMIP6
        {"prefixo": "ssp", "separador": "bio"},
        {"prefixo": "rcp", "separador": "bio"},
        # Outros formatos possíveis
        {"prefixo": "", "separador": "bio"}
    ]
    
    arquivos_processados = 0
    
    for arquivo in glob.glob(os.path.join(diretorio_origem, "*.tif")):
        nome_arquivo = os.path.basename(arquivo)
        nome_base = os.path.splitext(nome_arquivo)[0]
        
        # Tentar extrair número da variável bioclimática
        num_var = None
        
        for padrao in padroes:
            if padrao["separador"] in nome_base.lower():
                partes = nome_base.lower().split(padrao["separador"])
                if len(partes) > 1:
                    # Extrair dígitos do final
                    num_var = ''.join(filter(str.isdigit, partes[1]))
                    if num_var:
                        break
        
        if num_var:
            novo_nome = f"bio{num_var}.tif"
            arquivo_saida = os.path.join(diretorio_destino, novo_nome)
            
            # Copiar arquivo com novo nome
            shutil.copy(arquivo, arquivo_saida)
            print(f"Renomeado: {nome_arquivo} -> {novo_nome}")
            arquivos_processados += 1
        else:
            print(f"Não foi possível extrair número da variável de {nome_arquivo}")
    
    print(f"Padronização concluída: {arquivos_processados} arquivos processados")

def verificar_consistencia(dir_atual, dir_futuro):
    """
    Verifica se os mesmos arquivos existem em ambos os diretórios e têm as mesmas dimensões
    
    Args:
        dir_atual: Diretório com variáveis do clima atual
        dir_futuro: Diretório com variáveis do clima futuro
    
    Returns:
        Lista de variáveis em comum
    """
    print(f"Verificando consistência entre {dir_atual} e {dir_futuro}")
    
    arquivos_atual = set(os.path.basename(f) for f in glob.glob(os.path.join(dir_atual, "*.tif")))
    arquivos_futuro = set(os.path.basename(f) for f in glob.glob(os.path.join(dir_futuro, "*.tif")))
    
    # Verificar arquivos em comum
    comuns = arquivos_atual & arquivos_futuro
    print(f"Variáveis em comum: {len(comuns)}")
    
    if len(comuns) == 0:
        print("ERRO: Nenhuma variável em comum encontrada!")
        return []
    
    # Verificar dimensões
    problemas = []
    for arquivo in comuns:
        try:
            with rasterio.open(os.path.join(dir_atual, arquivo)) as src_atual:
                with rasterio.open(os.path.join(dir_futuro, arquivo)) as src_futuro:
                    if src_atual.shape != src_futuro.shape:
                        print(f"ALERTA: {arquivo} tem dimensões diferentes entre cenários")
                        print(f"  Atual: {src_atual.shape}, Futuro: {src_futuro.shape}")
                        problemas.append(arquivo)
                    if src_atual.crs != src_futuro.crs:
                        print(f"ALERTA: {arquivo} tem CRS diferentes entre cenários")
                        problemas.append(arquivo)
        except Exception as e:
            print(f"ERRO ao verificar {arquivo}: {str(e)}")
            problemas.append(arquivo)
    
    # Remover arquivos problemáticos da lista
    comuns_ok = [arq for arq in comuns if arq not in problemas]
    
    if problemas:
        print(f"ALERTA: {len(problemas)} variáveis com problemas foram removidas da lista")
    
    print(f"Variáveis consistentes: {len(comuns_ok)}")
    return sorted(list(comuns_ok))

def criar_metadados(variaveis, arquivo_saida="metadados_variaveis.md"):
    """
    Cria um arquivo de metadados para as variáveis bioclimáticas
    
    Args:
        variaveis: Lista de nomes de arquivos das variáveis
        arquivo_saida: Nome do arquivo de saída
    """
    descricoes = {
        "bio1.tif": "Temperatura Média Anual",
        "bio2.tif": "Amplitude Média Diurna (média mensal de (temp max - temp min))",
        "bio3.tif": "Isotermalidade (bio2/bio7) (×100)",
        "bio4.tif": "Sazonalidade da Temperatura (desvio padrão ×100)",
        "bio5.tif": "Temperatura Máxima do Mês Mais Quente",
        "bio6.tif": "Temperatura Mínima do Mês Mais Frio",
        "bio7.tif": "Amplitude Térmica Anual (bio5-bio6)",
        "bio8.tif": "Temperatura Média do Trimestre Mais Úmido",
        "bio9.tif": "Temperatura Média do Trimestre Mais Seco",
        "bio10.tif": "Temperatura Média do Trimestre Mais Quente",
        "bio11.tif": "Temperatura Média do Trimestre Mais Frio",
        "bio12.tif": "Precipitação Anual",
        "bio13.tif": "Precipitação do Mês Mais Úmido",
        "bio14.tif": "Precipitação do Mês Mais Seco",
        "bio15.tif": "Sazonalidade da Precipitação (coeficiente de variação)",
        "bio16.tif": "Precipitação do Trimestre Mais Úmido",
        "bio17.tif": "Precipitação do Trimestre Mais Seco",
        "bio18.tif": "Precipitação do Trimestre Mais Quente",
        "bio19.tif": "Precipitação do Trimestre Mais Frio"
    }
    
    with open(arquivo_saida, "w") as f:
        f.write("# Metadados das Variáveis Bioclimáticas\n\n")
        f.write("| Arquivo | Descrição | Unidade |\n")
        f.write("|---------|-----------|--------|\n")
        
        for var in sorted(variaveis):
            nome_base = os.path.basename(var)
            descricao = descricoes.get(nome_base, "Descrição não disponível")
            unidade = "°C" if int(nome_base.replace("bio", "").split(".")[0]) < 12 else "mm"
            f.write(f"| {nome_base} | {descricao} | {unidade} |\n")
    
    print(f"Metadados criados: {arquivo_saida}")

def main():
    """Função principal para processar arquivos bioclimáticos"""
    # Diretórios
    dir_downloads = "downloads"
    dir_clima_atual = "clima_atual"
    dir_clima_futuro = "clima_futuro"
    
    # Criar diretórios se não existirem
    os.makedirs(dir_downloads, exist_ok=True)
    os.makedirs(dir_clima_atual, exist_ok=True)
    os.makedirs(dir_clima_futuro, exist_ok=True)
    
    # Perguntar ao usuário o que deseja fazer
    print("\n===== PROCESSADOR DE VARIÁVEIS BIOCLIMÁTICAS =====\n")
    print("Escolha uma opção:")
    print("1. Extrair variáveis de arquivo .tif multi-banda")
    print("2. Padronizar nomes de arquivos existentes")
    print("3. Verificar consistência entre clima atual e futuro")
    print("4. Criar metadados para as variáveis")
    print("5. Executar todo o processo")
    print("6. Processar pastas que já contêm arquivos bio*.tif")
    print("0. Sair")
    
    opcao = input("\nOpção: ")
    
    if opcao == "1":
        arquivo_tif = input("Caminho para o arquivo .tif multi-banda: ")
        diretorio_saida = input("Diretório de saída [clima_atual]: ") or "clima_atual"
        extrair_variaveis_bioclimaticas(arquivo_tif, diretorio_saida)
    
    elif opcao == "2":
        diretorio_origem = input("Diretório com arquivos originais: ")
        diretorio_destino = input("Diretório de destino para arquivos padronizados: ")
        padronizar_nomes_arquivos(diretorio_origem, diretorio_destino)
    
    elif opcao == "3":
        dir_atual = input("Diretório com clima atual [clima_atual]: ") or "clima_atual"
        dir_futuro = input("Diretório com clima futuro [clima_futuro]: ") or "clima_futuro"
        variaveis_comuns = verificar_consistencia(dir_atual, dir_futuro)
        
        # Salvar lista de variáveis em comum
        if variaveis_comuns:
            with open("variaveis_usadas.txt", "w") as f:
                for var in variaveis_comuns:
                    f.write(f"{var}\n")
            print(f"Lista de variáveis salva em variaveis_usadas.txt")
    
    elif opcao == "4":
        dir_variaveis = input("Diretório com variáveis [clima_atual]: ") or "clima_atual"
        variaveis = [os.path.basename(f) for f in glob.glob(os.path.join(dir_variaveis, "*.tif"))]
        arquivo_saida = input("Nome do arquivo de saída [metadados_variaveis.md]: ") or "metadados_variaveis.md"
        criar_metadados(variaveis, arquivo_saida)
    
    elif opcao == "5":
        # Perguntar caminhos dos arquivos
        arquivo_atual = input("Caminho para o arquivo .tif do clima atual: ")
        arquivo_futuro = input("Caminho para o arquivo .tif do clima futuro: ")
        
        # Extrair variáveis
        if os.path.exists(arquivo_atual):
            if os.path.isdir(arquivo_atual):
                print(f"AVISO: {arquivo_atual} é um diretório, não um arquivo .tif")
            else:
                extrair_variaveis_bioclimaticas(arquivo_atual, dir_clima_atual)
        else:
            print(f"ERRO: Arquivo {arquivo_atual} não encontrado")
        
        if os.path.exists(arquivo_futuro):
            if os.path.isdir(arquivo_futuro):
                print(f"AVISO: {arquivo_futuro} é um diretório, não um arquivo .tif")
            else:
                extrair_variaveis_bioclimaticas(arquivo_futuro, dir_clima_futuro)
        else:
            print(f"ERRO: Arquivo {arquivo_futuro} não encontrado")
        
        # Verificar consistência
        variaveis_comuns = verificar_consistencia(dir_clima_atual, dir_clima_futuro)
        
        # Salvar lista de variáveis em comum
        if variaveis_comuns:
            with open("variaveis_usadas.txt", "w") as f:
                for var in variaveis_comuns:
                    f.write(f"{var}\n")
            print(f"Lista de variáveis salva em variaveis_usadas.txt")
            
            # Criar metadados
            criar_metadados(variaveis_comuns)
    
    elif opcao == "6":
        print("\nProcessando pastas existentes com arquivos bio*.tif")
        dir_atual = input("Diretório com clima atual [clima_atual]: ") or "clima_atual"
        dir_futuro = input("Diretório com clima futuro [clima_futuro]: ") or "clima_futuro"
        
        # Verificar se as pastas existem
        if not os.path.isdir(dir_atual):
            print(f"ERRO: Diretório {dir_atual} não encontrado")
            return
        if not os.path.isdir(dir_futuro):
            print(f"ERRO: Diretório {dir_futuro} não encontrado")
            return
            
        # Verificar consistência
        variaveis_comuns = verificar_consistencia(dir_atual, dir_futuro)
        
        # Salvar lista de variáveis em comum
        if variaveis_comuns:
            with open("variaveis_usadas.txt", "w") as f:
                for var in variaveis_comuns:
                    f.write(f"{var}\n")
            print(f"Lista de variáveis salva em variaveis_usadas.txt")
            
            # Criar metadados
            criar_metadados(variaveis_comuns)
    
    elif opcao == "0":
        print("Saindo...")
    
    else:
        print("Opção inválida!")

if __name__ == "__main__":
    main()
