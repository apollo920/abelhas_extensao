import pandas as pd

# Caminho para o arquivo GBIF
arquivo = "0024588-250426092105405/occurrence.txt"

# Lê o arquivo com separador TAB (padrão Darwin Core)
df = pd.read_csv(arquivo, sep="\t", low_memory=False)

# Filtra apenas registros com coordenadas válidas
df = df[["decimalLatitude", "decimalLongitude", "scientificName"]]
df = df.dropna(subset=["decimalLatitude", "decimalLongitude"])

# Filtra só a espécie desejada (Trigona spinipes)
df = df[df["scientificName"].str.contains("Trigona spinipes", case=False, na=False)]

# Renomeia colunas como seu script espera
df = df.rename(columns={"decimalLatitude": "latitude", "decimalLongitude": "longitude"})

# Salva como CSV simples
df[["latitude", "longitude"]].to_csv("ocorrencias.csv", index=False)

print("✅ ocorrencias.csv salvo com sucesso!")