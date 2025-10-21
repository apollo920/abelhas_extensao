# 🐝 Modelagem de Distribuição de Espécies com Mudanças Climáticas

Sistema completo para modelagem de distribuição de espécies usando variáveis bioclimáticas e projeções de mudanças climáticas.

## 📋 Descrição

Este projeto implementa um workflow completo para:

- Processar variáveis bioclimáticas do WorldClim
- Treinar modelos de distribuição de espécies (Random Forest)
- Gerar mapas de adequabilidade climática atual e futura
- Visualizar mudanças na distribuição geográfica

## 🛠️ Requisitos

### Clonando o Repositório

Este projeto usa Git LFS para gerenciar arquivos de dados climáticos grandes. Antes de clonar:

**Linux:**
```bash
sudo apt install git-lfs  # Ubuntu/Debian
git lfs install
git clone <url-do-repositorio>
```

**Windows:**
Baixe e instale Git LFS de [git-lfs.github.com](https://git-lfs.github.com/), depois:
```bash
git lfs install
git clone <url-do-repositorio>
```

Se você já clonou o repositório antes de instalar o Git LFS:
```bash
git lfs pull
```

### Dependências Python

```bash
pip install pandas numpy rasterio scikit-learn joblib tqdm geopy matplotlib geopandas
```

### Dados Necessários

1. **Variáveis Bioclimáticas**: Dados do WorldClim (bio1.tif a bio19.tif) - incluídos via Git LFS
2. **Ocorrências da Espécie**: Arquivo CSV com colunas `latitude` e `longitude`
3. **Shapefile do Brasil**: Para aplicar máscara geográfica

## 📁 Estrutura do Projeto

```
abelhas_extensao/
├── processar_variaveis.py     # Processamento inicial dos dados
├── ajustar_modelo.py          # Treinamento do modelo
├── gerar_mapas_ajustados.py   # Geração de mapas de distribuição
├── mascarar_brasil.py         # Aplicação de máscara geográfica
├── visualizar_mapas.py        # Visualização dos resultados
├── atualizar_dados.py         # Utilitários para dados
├── clima_atual/               # Variáveis bioclimáticas atuais
├── clima_futuro/              # Variáveis bioclimáticas futuras
├── BR_UF_2024/               # Shapefile do Brasil
└── ocorrencias.csv           # Dados de ocorrência da espécie
```

## 🚀 Como Usar

### 1. Preparação dos Dados

```bash
python processar_variaveis.py
```

- Opção 6: Para processar pastas existentes com arquivos bio\*.tif

### 2. Treinamento do Modelo

```bash
python ajustar_modelo.py
```

- Treina um Random Forest com parâmetros anti-overfitting
- Gera pseudo-ausências automaticamente
- Salva modelo, scaler e lista de variáveis

### 3. Geração de Mapas

```bash
python gerar_mapas_ajustados.py
```

- Gera mapas de distribuição atual e futura
- Calcula mapas de mudança (futuro - atual)
- Escolha entre diferentes cenários climáticos

### 4. Aplicação de Máscara

```bash
python mascarar_brasil.py
```

- Aplica máscara do território brasileiro
- Processa automaticamente todos os mapas gerados

### 5. Visualização

```bash
python visualizar_mapas.py
```

- Cria visualizações padronizadas
- Escalas científicas consistentes
- Salva mapas em alta resolução (300 DPI)

## 📊 Outputs

### Arquivos Gerados

- `mapa_predito_atual_*.tif`: Distribuição atual
- `mapa_predito_futuro*.tif`: Distribuição futura
- `mapa_mudanca_*.tif`: Mudanças na distribuição
- `*_brasil.tif`: Versões com máscara do Brasil
- `*.png`: Visualizações em alta resolução

### Visualizações

- **Distribuição Atual**: Probabilidade de adequabilidade climática
- **Distribuição Futura**: Projeções para cenários climáticos
- **Mapas de Mudança**: Diferenças entre atual e futuro

## 🎯 Características Técnicas

### Modelo

- **Algoritmo**: Random Forest
- **Parâmetros Anti-overfitting**: min_samples_leaf=30, max_depth=10
- **Pseudo-ausências**: Geradas automaticamente com distância mínima de 30km
- **Normalização**: StandardScaler para todas as variáveis

### Visualização

- **Escalas Padronizadas**: 0-1 para probabilidades, -0.8 a +0.8 para mudanças
- **Paletas de Cores**: Viridis (probabilidade), RdBu_r (mudanças)
- **Resolução**: 300 DPI para publicação científica

## 📈 Interpretação dos Resultados

### Mapas de Probabilidade

- **0.0**: Inadequado climaticamente
- **1.0**: Altamente adequado
- **0.5+**: Áreas potencialmente favoráveis

### Mapas de Mudança

- **Valores Negativos**: Perda de adequabilidade
- **Valores Positivos**: Ganho de adequabilidade
- **Vermelho**: Áreas com redução
- **Azul**: Áreas com aumento

## 🔧 Utilitários

### atualizar_dados.py

```bash
python atualizar_dados.py
```

- Extrair variáveis de arquivos multi-banda
- Limpar arquivos gerados
- Verificar consistência dos dados

## 📝 Notas Importantes

1. **Tamanho dos Dados**: Arquivos .tif são excluídos do Git (muito pesados)
2. **Dependências Espaciais**: Requires GDAL e bibliotecas geoespaciais
3. **Memória**: Processamento pode requerer 8GB+ RAM para dados globais
4. **Tempo**: Geração de pseudo-ausências pode levar várias horas

## 🤝 Contribuição

Para contribuir:

1. Fork o repositório
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Push para a branch
5. Abra um Pull Request

## 📄 Licença

Este projeto está sob licença MIT. Veja o arquivo LICENSE para detalhes.

## 📚 Referências

- [WorldClim](https://worldclim.org/): Dados bioclimáticos
- [IBGE](https://www.ibge.gov.br/): Shapefiles do Brasil
- [scikit-learn](https://scikit-learn.org/): Machine learning
- [rasterio](https://rasterio.readthedocs.io/): Processamento de rasters
