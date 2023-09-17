import torch
from utils import *
from model.seq2sql import Seq2SQL
from model.sqlnet import SQLNet

model_config = {
    'model': 'SQLNet',              # SQLNet or Seq2SQL
    'batch_size': 64,
    'dataset_id': 4,                # 0 for original, 1 for re-split dataset, 2 for altered questions 1, 3 for altered questions 2, 4 for altered questions 3
    'use_column_attention': True,   # Seq2SQL does not support column attention
    'trainable_embedding': False,   # Seq2SQL does not support trainable embedding
    'embedding_path': 'glove/glove.42B.300d.txt'
}

class ModelTester:
    def __init__(self, config):
        self.config = config
        self.batch_size = config['batch_size']
        self.model_name = config['model']

        _, _, _, _, self.test_sql_data, self.test_table_data = load_dataset(config['dataset_id'])
        word_embedding = load_word_embeddings(config['embedding_path'], load_used=config['trainable_embedding'])

        if self.model_name == "Seq2SQL":
            self.model = Seq2SQL(word_embedding, 300, use_gpu=True)
        elif self.model_name == "SQLNet":
            self.model = SQLNet(word_embedding, 300, use_gpu=True, use_column_attention=config['use_column_attention'], 
                                trainable_embedding=config['trainable_embedding'])

    def load_model(self, model_type):
        getattr(self.model, f"{model_type}Predictor").load_state_dict(
            torch.load(f"saved_models/{self.model_name}_best_{model_type.lower()}_model"))
        if self.config['trainable_embedding'] == True:
            getattr(self.model, f"{model_type.lower()}_embedding_layer").load_state_dict(
                torch.load(f"saved_models/{self.model_name}_best_{model_type.lower()}_embed_model"))
    
    def test(self):
        self.load_model('Aggregate')
        self.load_model('Select')
        self.load_model('Condition')
        test_accuracy = epoch_accuracy(self.model, self.batch_size, self.test_sql_data, self.test_table_data)
        print(f"Test overall accuracy: {test_accuracy[0]}\nBreakdown on (agg, sel, where): {test_accuracy[1]}")

model_tester = ModelTester(model_config)
model_tester.test()