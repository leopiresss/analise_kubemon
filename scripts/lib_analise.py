import pickle

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

def get_dataset_analise(nome_data_set = nome_dataset_default,analise_ganho_de_informacao=False):    
    info_modelo = get_info_modelo(nome_data_set)
    with open(info_modelo['parametros']['arq_dataset_pkl'], 'rb') as f:
        datasets = pickle.load(f)
        if analise_ganho_de_informacao:
            if 'features_ganho_informacao' not in datasets or datasets['features_ganho_informacao'] is None:
                raise ValueError("O dataset não contém 'features_ganho_informacao' ou ela é None. Execute primeiro a análise de ganho de informação no notebook correspondente.")
            
            # Aplicar seleção de features baseada no ganho de informação
            datasets['X_train'] = datasets['X_train'][datasets['features_ganho_informacao']]
            datasets['X_test'] = datasets['X_test'][datasets['features_ganho_informacao']]
            datasets['X_val'] = datasets['X_val'][datasets['features_ganho_informacao']]
            datasets['y_train'] = datasets['y_train']   
            datasets['y_test'] = datasets['y_test']
            datasets['y_val'] = datasets['y_val']
            datasets['X_train_scaled'] = datasets['X_train_scaled']
            datasets['X_test_scaled'] = datasets['X_test_scaled']
            datasets['X_val_scaled'] = datasets['X_val_scaled']
            datasets['classes_mapping'] = datasets['classes_mapping']
            datasets['features_ganho_informacao'] = datasets['features_ganho_informacao']
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
