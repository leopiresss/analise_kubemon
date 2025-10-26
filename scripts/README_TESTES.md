# 🧪 Testes Unitários - UtilClassificadores

Este documento descreve os testes unitários implementados para a classe `UtilClassificadores`.

## 📋 Estrutura dos Testes

### Classes de Teste Implementadas

1. **TestUtilClassificadoresInit**
   - Testa inicialização da classe
   - Validação de títulos padrão e customizados
   - Filtragem de títulos inválidos
   - Verificação de atributos inicializados
   - Teste do método `__repr__()`

2. **TestCreateClassifiers**
   - Criação de classificadores com configuração padrão
   - Validação de todos os classificadores disponíveis
   - Verificação de métodos `fit` e `predict`
   - Correspondência entre `valid_titles` e `methods`

3. **TestTrainClassifiers**
   - Treinamento bem-sucedido de classificadores
   - Armazenamento de modelos treinados
   - Validação do status de treinamento
   - Verificação de atributos pós-treinamento

4. **TestEvaluateClassifiers**
   - Avaliação de classificadores treinados
   - Validação de intervalo de acurácia [0, 1]
   - Estrutura dos resultados retornados
   - Teste de scores e resultados

5. **TestGetMethodsTrainedModels**
   - Retorno de tupla (methods, valid_titles)
   - Filtragem de modelos com sucesso
   - Exclusão de modelos com erro

6. **TestGenerateModelMetricsDataset**
   - Geração de DataFrame com métricas
   - Validação de colunas obrigatórias
   - Salvamento de arquivo CSV
   - Métricas calculadas corretamente

7. **TestGenerateConfusionMatrices**
   - Geração de matrizes de confusão
   - Cálculo de acurácia
   - Validação de estrutura de resultados

8. **TestGenerateRocAucCurves**
   - Geração de curvas ROC/AUC
   - Execução sem erros
   - Suporte para classificação binária

9. **TestUtilityMethods**
   - `get_titles_trained_models()`
   - `get_trained_models()`
   - `get_successful_models()`

10. **TestIntegration**
    - Fluxo completo de trabalho
    - Integração entre métodos
    - Pipeline completo: criar → treinar → avaliar → métricas

## 🚀 Como Executar os Testes

### Opção 1: Executar Todos os Testes

```powershell
# No diretório scripts/
python lib_classificador_testes.py
```

### Opção 2: Executar com unittest

```powershell
# Executar todos os testes
python -m unittest lib_classificador_testes -v

# Executar uma classe específica de testes
python -m unittest lib_classificador_testes.TestTrainClassifiers -v

# Executar um teste específico
python -m unittest lib_classificador_testes.TestTrainClassifiers.test_train_classifiers_success -v
```

### Opção 3: Executar com pytest (se instalado)

```powershell
# Instalar pytest
pip install pytest pytest-cov

# Executar testes
pytest lib_classificador_testes.py -v

# Executar com cobertura
pytest lib_classificador_testes.py --cov=lib_classificador --cov-report=html
```

## 📊 Cobertura de Testes

Os testes cobrem os seguintes métodos críticos:

| Método | Cobertura | Descrição |
|--------|-----------|-----------|
| `__init__()` | ✅ Alta | Inicialização e validação |
| `create_classifiers()` | ✅ Alta | Criação de classificadores |
| `train_classifiers()` | ✅ Alta | Treinamento de modelos |
| `evaluate_classifiers()` | ✅ Alta | Avaliação de desempenho |
| `get_methods_trained_models()` | ✅ Alta | Extração de modelos |
| `generate_model_metrics_dataset()` | ✅ Média | Geração de métricas |
| `generate_confusion_matrices()` | ✅ Média | Matrizes de confusão |
| `generate_roc_auc_curves()` | ✅ Média | Curvas ROC/AUC |
| `get_titles_trained_models()` | ✅ Alta | Métodos utilitários |
| `get_trained_models()` | ✅ Alta | Métodos utilitários |
| `get_successful_models()` | ✅ Alta | Métodos utilitários |

## 🎯 Casos de Teste Importantes

### Inicialização
- ✅ Títulos padrão funcionam corretamente
- ✅ Títulos customizados são aceitos
- ✅ Títulos inválidos são filtrados
- ✅ Todos os atributos são inicializados

### Treinamento
- ✅ Modelos são treinados com sucesso
- ✅ Modelos treinados são armazenados
- ✅ Status de treinamento é registrado
- ✅ Erros são tratados adequadamente

### Avaliação
- ✅ Acurácia está no intervalo correto [0, 1]
- ✅ Resultados têm estrutura esperada
- ✅ Múltiplos modelos podem ser avaliados

### Métricas
- ✅ DataFrame é gerado corretamente
- ✅ Colunas obrigatórias estão presentes
- ✅ CSV pode ser salvo
- ✅ Métricas são calculadas (acurácia, precisão, recall, F1)

## 🔧 Dependências para Testes

```python
unittest          # Built-in Python
numpy            # Manipulação de arrays
pandas           # DataFrames
scikit-learn     # Algoritmos ML e datasets
matplotlib       # Visualizações (mockado em testes)
```

## 📝 Exemplo de Saída

```
======================================================================
EXECUTANDO TESTES UNITÁRIOS - UtilClassificadores
======================================================================
test_init_default_titles (lib_classificador_testes.TestUtilClassificadoresInit) ... ok
test_init_custom_titles (lib_classificador_testes.TestUtilClassificadoresInit) ... ok
...
----------------------------------------------------------------------
Ran 40 tests in 12.345s

======================================================================
RESUMO DOS TESTES
======================================================================
✅ Testes executados: 40
✅ Sucessos: 40
❌ Falhas: 0
❌ Erros: 0

🎉 TODOS OS TESTES PASSARAM COM SUCESSO!
```

## 🐛 Troubleshooting

### Erro: ModuleNotFoundError: No module named 'deslib'
```powershell
pip install deslib
```

### Erro: ModuleNotFoundError: No module named 'xgboost'
```powershell
pip install xgboost
```

### Erro: Gráficos sendo exibidos durante testes
Os testes utilizam `@patch('matplotlib.pyplot.show')` para evitar exibição de gráficos.
Certifique-se de que os decorators estão presentes.

## 📈 Melhorias Futuras

- [ ] Adicionar testes de performance
- [ ] Adicionar testes de carga (datasets grandes)
- [ ] Implementar testes para métodos de persistência
- [ ] Adicionar testes de validação cruzada
- [ ] Implementar testes de comparação de modelos
- [ ] Adicionar testes de tratamento de dados desbalanceados

## 📚 Referências

- [unittest Documentation](https://docs.python.org/3/library/unittest.html)
- [scikit-learn Testing](https://scikit-learn.org/stable/developers/develop.html)
- [pytest Documentation](https://docs.pytest.org/)

---

**Autor**: Sistema de Testes  
**Data**: 22/10/2025  
**Versão**: 1.0
