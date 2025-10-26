import pickle
import os
from sklearn.preprocessing import PowerTransformer
from sklearn import datasets
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, PowerTransformer
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

nome_dataset_default = 'svm'
def get_info_modelo(nome_dataset = nome_dataset_default):
    if nome_dataset == 'svm':
        return {
            'nome_dataset': 'svm',
            'parametros': {
                'arq_dataset_csv': '../dataset/svm.csv',
                'arq_dataset_pkl': '../dataset/svm.pkl'
            }
        }
    if nome_dataset == 'terasort':
        return {
            'nome_dataset': 'terasort',
            'parametros': {
                'arq_dataset_csv': '../dataset/terasort.csv',
                'arq_dataset_pkl': '../dataset/terasort.pkl'
            }
        }


def save_informacao_analise(datasets = None, nome_data_set = nome_dataset_default):
    try:
        info_modelo = get_info_modelo(nome_data_set)
        
        # Verificar se o arquivo PKL existe
        arquivo_pkl = info_modelo['parametros']['arq_dataset_pkl']
        
        with open(arquivo_pkl, 'wb') as f:
            datasets['X_train_scaled'] = datasets['X_train_scaled'] if 'X_train_scaled' in datasets else None
            datasets['X_test_scaled'] = datasets['X_test_scaled'] if 'X_test_scaled' in datasets else None
            datasets['X_val_scaled'] = datasets['X_val_scaled'] if 'X_val_scaled' in datasets else None           
            pickle.dump(datasets, f)
        print(f"✅ Dataset salvo com sucesso em {arquivo_pkl}")
        return datasets
    except Exception as e:
        print(f"❌ Erro ao salvar dataset: {e}")
        raise
    
def print_informacao_analise(nome_data_set = nome_dataset_default):    
    try:
        info_modelo = get_info_modelo(nome_data_set)
        
        # Verificar se o arquivo PKL existe
        arquivo_pkl = info_modelo['parametros']['arq_dataset_pkl']
        print('Nome do dataset: ', nome_data_set)
        with open(arquivo_pkl, 'rb') as f:
            datasets = pickle.load(f)
            print('X_train.shape', datasets['X_train'].shape)
            print('X_test.shape', datasets['X_test'].shape)
            print('X_val.shape', datasets['X_val'].shape)            
            print('X_train_scaled.shape', datasets['X_train_scaled'].shape if 'X_train_scaled' in datasets and datasets['X_train_scaled'] is not None else None)
            print('X_test_scaled.shape', datasets['X_test_scaled'].shape if 'X_test_scaled' in datasets and datasets['X_test_scaled'] is not None else None)
            print('X_val_scaled.shape', datasets['X_val_scaled'].shape if 'X_val_scaled' in datasets and datasets['X_val_scaled'] is not None else None)
            print('classes_mapping', datasets['classes_mapping'] if 'classes_mapping' in datasets else None)
            print('features_ganho_informacao', datasets['features_ganho_informacao'] if 'features_ganho_informacao' in datasets else None)  
            print('qtd features_ganho_informacao: ', len(datasets['features_ganho_informacao']) if 'features_ganho_informacao' in datasets else None)  
       
        
    except FileNotFoundError as e:
        print(f"❌ Erro de arquivo: {e}")
        raise
    except pickle.UnpicklingError as e:
        print(f"❌ Erro ao carregar o arquivo PKL: {e}")
        raise
    except Exception as e:
        print(f"❌ Erro inesperado ao carregar dataset: {e}")
        raise

def get_dataset_analise(nome_data_set = nome_dataset_default,analise_ganho_de_informacao=False):    
    try:
        info_modelo = get_info_modelo(nome_data_set)
        # Verificar se o arquivo PKL existe
        arquivo_pkl = info_modelo['parametros']['arq_dataset_pkl']
        
        with open(arquivo_pkl, 'rb') as f:
            datasets = pickle.load(f)
            
            if analise_ganho_de_informacao:
                if 'features_ganho_informacao' not in datasets or datasets['features_ganho_informacao'] is None:
                    raise ValueError("O dataset não contém 'features_ganho_informacao' ou ela é None. Execute primeiro a análise de ganho de informação no notebook correspondente.")
                # Aplicar seleção de features baseada no ganho de informação
                print("============>", len(datasets['features_ganho_informacao']))
                datasets['X_train'] = datasets['X_train'][datasets['features_ganho_informacao']]
                datasets['X_test'] = datasets['X_test'][datasets['features_ganho_informacao']]  
                datasets['X_val'] = datasets['X_val'][datasets['features_ganho_informacao']]
                
                # Aplicar seleção apenas se os dados escalados existirem
                if datasets['X_train_scaled'] is not None:
                    datasets['X_train_scaled'] = datasets['X_train_scaled'][datasets['features_ganho_informacao']]
                if datasets['X_test_scaled'] is not None:
                    datasets['X_test_scaled'] = datasets['X_test_scaled'][datasets['features_ganho_informacao']]
                if datasets['X_val_scaled'] is not None:
                    datasets['X_val_scaled'] = datasets['X_val_scaled'][datasets['features_ganho_informacao']]
        return datasets
        
    except FileNotFoundError as e:
        print(f"❌ Erro de arquivo: {e}")
        raise
    except pickle.UnpicklingError as e:
        print(f"❌ Erro ao carregar o arquivo PKL: {e}")
        raise
    except KeyError as e:
        print(f"❌ Erro: Chave não encontrada no dataset: {e}")
        raise
    except Exception as e:
        print(f"❌ Erro inesperado ao carregar dataset: {e}")
        raise


def atualizar_features_dataset_analise(datasets = None,features = None):    
    if datasets is None or features is None:
        raise ValueError("Os parâmetros 'datasets' e 'features' não podem ser None.")
    # Aplicar seleção de features baseada no ganho de informação
    datasets['X_train'] = datasets['X_train'][features]
    datasets['X_test'] = datasets['X_test'][features]
    datasets['X_val'] = datasets['X_val'][features]
    datasets['features_ganho_informacao'] = features
    return datasets


# Normalização dos dados usando Yeo-Johnson
def normalization_dataset(datasets):
    print(f"\n⚖️ Aplicando transformação Yeo-Johnson...")
    datasets['yeo_johnson_transformer'] = PowerTransformer(method='yeo-johnson', standardize=True)
    # O fit é feito apenas no conjunto de treino para evitar data leakage
    datasets['X_train_scaled'] = datasets['yeo_johnson_transformer'].fit_transform(datasets['X_train'])
    datasets['X_test_scaled'] = datasets['yeo_johnson_transformer'].transform(datasets['X_test'])
    datasets['X_val_scaled'] = datasets['yeo_johnson_transformer'].transform(datasets['X_val'])
    save_informacao_analise(datasets = datasets)
    print(f"   ✅ Transformação Yeo-Johnson aplicada com StandardScaler integrado")
    print(f"   • Média treino antes: {datasets['X_train'].mean().mean():.3f} | depois: {datasets['X_train_scaled'].mean():.3f}")
    print(f"   • Std treino antes: {datasets['X_train'].std().mean():.3f} | depois: {datasets['X_train_scaled'].std().mean():.3f}")
    print(f"   • Média teste antes: {datasets['X_test'].mean().mean():.3f} | depois: {datasets['X_test_scaled'].mean():.3f}")
    print(f"   • Std teste antes: {datasets['X_test'].std().mean():.3f} | depois: {datasets['X_test_scaled'].std().mean():.3f}")
    print(f"   • Média validação antes: {datasets['X_val'].mean().mean():.3f} | depois: {datasets['X_val_scaled'].mean():.3f}")
    print(f"   • Std validação antes: {datasets['X_val'].std().mean():.3f} | depois: {datasets['X_val_scaled'].std().mean():.3f}")
    print(f"   • Transformação aplicada: Yeo-Johnson + Padronização")
    return datasets



def preparar_dados(nome_dataset):
    info_modelo = get_info_modelo(nome_dataset)  # para garantir que a função está carregada da
    print(info_modelo)
    arq_dataset_csv = info_modelo['parametros']['arq_dataset_csv']
    arq_dataset_pkl = info_modelo['parametros']['arq_dataset_pkl']
    df = pd.read_csv(arq_dataset_csv)
    # Preparação dos dados
    print("🔧 Preparando dados para treinamento...")

    # Separar features e target
    colunas_excluir = ['os_timestamp', 'node_name', 'iteration', 'target']
    colunas_excluir = [col for col in colunas_excluir if col in df.columns]

    # Selecionar apenas features numéricas
    features_numericas = df.select_dtypes(include=[np.number]).columns.tolist()
    features_para_modelo = [col for col in features_numericas if col not in colunas_excluir]

    print(f"📊 Features para o modelo:")
    print(f"   • Total de features: {len(features_para_modelo)}")
    print(f"   • Colunas excluídas: {colunas_excluir}")

    # Preparar X e y
    X = df[features_para_modelo].copy()
    y = df['target'].copy()

    # Tratar valores ausentes
    valores_ausentes = X.isnull().sum().sum()
    if valores_ausentes > 0:
        print(f"   • Preenchendo {valores_ausentes:,} valores ausentes com a mediana...")
        X = X.fillna(X.median())


    # Codificar target se necessário
    le = LabelEncoder()
    if y.dtype == 'object':
        y_encoded = le.fit_transform(y)
        classes_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        print(f"   • Target codificado: {classes_mapping}")
    else:
        y_encoded = y.values
        classes_mapping = None

    print(f"\n✅ Dados preparados:")
    print(f"   • Shape X: {X.shape}")
    print(f"   • Shape y: {len(y_encoded)}")
    print(f"   • Classes únicas: {np.unique(y_encoded)}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, 
        test_size=0.6, 
        random_state=42, 
        stratify=y_encoded
    )

    # Dividindo o teste em teste e validação (50%/50%)
    X_test, X_val, y_test, y_val = train_test_split(
        X_test, y_test, 
        test_size=0.5, 
        random_state=42, 
        stratify=y_test
    )


    datasets = {}
    datasets['nome_dataset'] = nome_dataset
    datasets['X_train'] = X_train
    datasets['X_test'] = X_test
    datasets['X_val'] = X_val
    datasets['y_train'] = y_train
    datasets['y_test'] = y_test
    datasets['y_val'] = y_val
    datasets['classes_mapping'] = classes_mapping
    datasets['features_ganho_informacao'] = features_para_modelo
    save_informacao_analise(nome_data_set = nome_dataset, datasets = datasets)
    print_informacao_analise(nome_data_set = nome_dataset)
    return datasets


def main():
    """
    Função principal para demonstrar o uso da biblioteca lib_analise
    """
    print("🔍 Biblioteca de Análise KubeMon")
    print("=" * 50)
    
    try:
        # Obter informações do modelo padrão
        info_modelo = get_info_modelo()
        print(f"📊 Dataset: {info_modelo['nome_dataset']}")
        print(f"📁 Arquivo CSV: {info_modelo['parametros']['arq_dataset_csv']}")
        print(f"📦 Arquivo PKL: {info_modelo['parametros']['arq_dataset_pkl']}")
        
        # Carregar dataset básico
        print("\n🔄 Carregando dataset básico...")
        datasets_basico = get_dataset_analise(analise_ganho_de_informacao=False)
        
        print(f"✅ Dataset básico carregado:")
        print(f"   • X_train shape: {datasets_basico['X_train'].shape}")
        print(f"   • X_test shape: {datasets_basico['X_test'].shape}")
        print(f"   • X_val shape: {datasets_basico['X_val'].shape}")
        print(f"   • Classes: {list(datasets_basico['classes_mapping'].keys())}")
        
        # Tentar carregar dataset com análise de ganho de informação
        print("\n🧠 Tentando carregar dataset com análise de ganho de informação...")
        try:
            datasets_gi = get_dataset_analise(analise_ganho_de_informacao=True)
            print(f"✅ Dataset com ganho de informação carregado:")
            print(f"   • Features selecionadas: {len(datasets_gi['features_ganho_informacao'])}")
            print(f"   • X_train shape reduzido: {datasets_gi['X_train'].shape}")
        except ValueError as e:
            print(f"⚠️ {e}")
        
    except Exception as e:
        print(f"❌ Erro: {e}")
        print("   Verifique se os arquivos de dataset estão disponíveis.")
    
    print("\n🎯 Biblioteca pronta para uso!")


if __name__ == "__main__":
    main()
