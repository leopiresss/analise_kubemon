# Features Selecionadas - Análise de Ganho de Informação

**Data de criação:** 2025-10-15 12:26:12.091541

**Estratégia utilizada:** above_median

## Resumo

- **Features originais:** 52
- **Features selecionadas:** 15
- **Redução:** 71.2%

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
| 50 | mean_os_net_bytes_sent | 0.607395 | 0.658197 |
| 14 | mean_os_net_packets_sent | 0.412268 | 0.288321 |
| 23 | mean_os_net_packets_recv | 0.243639 | 0.224137 |
| 46 | mean_os_cpu_interrupts | 0.210774 | 0.188144 |
|  1 | mean_os_cpu_ctx_switches | 0.187970 | 0.194073 |
| 11 | mean_os_cpu_soft_interrupts | 0.186976 | 0.177688 |
| 29 | mean_os_cpu_softirq | 0.135683 | 0.124693 |
| 17 | mean_os_cpu_system | 0.133248 | 0.213352 |
| 24 | mean_os_cpu_user | 0.084585 | 0.125483 |
| 12 | mean_os_net_bytes_recv | 0.046311 | 0.270387 |

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
