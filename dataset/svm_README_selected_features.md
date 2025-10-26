# Features Selecionadas - Análise de Ganho de Informação

**Data de criação:** 2025-10-23 21:16:41.445651

**Estratégia utilizada:** above_median

## Resumo

- **Features originais:** 48
- **Features selecionadas:** 19
- **Redução:** 60.4%

## Arquivos Gerados

1. **selected_features.txt** - Lista simples das features selecionadas
2. **selected_datasets.pkl** - Arquivo pickle com todos os datasets reduzidos e metadados
3. **selected_features_summary.pkl** - Scores e rankings detalhados das features (pickle)
4. **README_selected_features.md** - Este arquivo de documentação

## Conteúdo do Arquivo Pickle Principal

O arquivo `selected_datasets.pkl` contém:
- **X_selected**: Dataset principal com features selecionadas
- **selected_features_list**: Lista das features selecionadas
- **selected_features_summary**: DataFrame com scores e rankings
- **Conjuntos treino/teste/validação**: Versões reduzidas dos conjuntos
- **Metadados**: Estratégia usada, data de criação, informações de redução

## Top 10 Features

| Rank | Feature | Information Gain | Mutual Information |
|------|---------|------------------|--------------------|
|  4 | mean_container_net_rx_packets | 0.019351 | 0.072516 |
|  2 | mean_container_net_tx_packets | 0.009718 | 0.079438 |
|  7 | mean_os_disk_write_io | 0.007512 | 0.047747 |
|  3 | mean_os_mem_nr_mapped | 0.005773 | 0.073238 |
| 22 | mean_os_disk_write_merge | 0.004950 | 0.013181 |
| 15 | mean_container_mem_mapped_file | 0.004915 | 0.026477 |
|  8 | mean_container_cpu_user | 0.003533 | 0.053143 |
|  9 | mean_container_cpu_system | 0.003258 | 0.042540 |
| 21 | mean_os_mem_nr_inactive_anon | 0.002461 | 0.016893 |
|  5 | mean_os_mem_pgpgout | 0.002220 | 0.062524 |

## Como Usar

```python
import pandas as pd
import numpy as np
import pickle

# Carregar features selecionadas (texto)
with open('selected_features.txt', 'r') as f:
    selected_features = [line.strip() for line in f if not line.startswith('#') and line.strip()]

# Carregar todos os datasets e metadados (pickle)
with open('selected_datasets.pkl', 'rb') as f:
    data = pickle.load(f)

# Acessar datasets
X_selected = data['X_selected']
selected_features = data['selected_features_list']
summary = data['selected_features_summary']

# Acessar conjuntos de treino/teste (se disponíveis)
if 'X_train_selected' in data:
    X_train_selected = data['X_train_selected']
    y_train = data.get('y_train', None)

# Carregar summary separado (pickle)
with open('selected_features_summary.pkl', 'rb') as f:
    summary_df = pickle.load(f)
```
