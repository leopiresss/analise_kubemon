
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