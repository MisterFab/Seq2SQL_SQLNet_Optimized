import torch
import datetime
from model.seq2sql import Seq2SQL
from model.sqlnet import SQLNet
from utils import *

model_config = {
    'model': 'SQLNet',              # SQLNet or Seq2SQL
    'num_epochs': 100,
    'batch_size': 64,
    'dataset_id': 0,                # 0 for original, 1 for re-split dataset
    'use_column_attention': True,   # Seq2SQL does not support column attention
    'trainable_embedding': False,   # Seq2SQL does not support trainable embedding
    'embedding_path': 'glove/glove.42B.300d.txt',
    'learning_rate': 1e-3
}

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.num_epochs = config['num_epochs']
        self.batch_size = config['batch_size']
        self.model_name = config['model']

        use_gpu = torch.cuda.is_available()

        print("Using CUDA GPU") if use_gpu else print("Using CPU")

        self.train_sql_data, self.train_table_data, self.val_sql_data, self.val_table_data, _, _ = load_dataset(config['dataset_id'])
        word_embedding = load_word_embeddings(config['embedding_path'], load_used=config['trainable_embedding'])

        if self.model_name == "Seq2SQL":
            assert not config['trainable_embedding'], "Seq2SQL does not support trainable embedding"
            assert not config['use_column_attention'], "Seq2SQL does not support column attention"
            self.model = Seq2SQL(word_embedding, 300, use_gpu=use_gpu)
        elif self.model_name == "SQLNet":
            self.model = SQLNet(word_embedding, 300, use_gpu=use_gpu, use_column_attention=config['use_column_attention'], 
                                trainable_embedding=config['trainable_embedding'])

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay = 0)

        self.epoch_loss = []
        self.best_agg_epoch, self.best_sel_epoch, self.best_cond_epoch = 0, 0, 0

    def load_model(self, model_type):
        getattr(self.model, f"{model_type}Predictor").load_state_dict(
            torch.load(f"saved_models/{self.model_name}_best_{model_type.lower()}_model"))

    def train(self):
        if self.config['trainable_embedding'] == True:
            print("Loading from pretrained model")
            self.load_model("Aggregate")
            self.load_model("Select")
            self.load_model("Condition")
    
        initial_accuracy = epoch_accuracy(self.model, self.batch_size, self.val_sql_data, self.val_table_data)
        print(f"Initial overall accuracy: {initial_accuracy[0]}\nBreakdown on (agg, sel, where): {initial_accuracy[1]}")

        best_agg_accuracy, best_sel_accuracy, best_cond_accuracy = initial_accuracy[1]
        
        for i in range(self.num_epochs):
            print(f'Epoch: {i+1} / {datetime.datetime.now()}')

            loss = epoch_train(self.model, self.optimizer, self.batch_size, self.train_sql_data, self.train_table_data)
            self.epoch_loss.append(loss)
            print(f"Loss: {loss}")

            validation_accuracy = epoch_accuracy(self.model, self.batch_size, self.val_sql_data, self.val_table_data)
            print(f"Validation overall accuracy: {validation_accuracy[0]}\nBreakdown on (agg, sel, where): {validation_accuracy[1]}")

            agg_accuracy, sel_accuracy, cond_accuracy = validation_accuracy[1]

            if agg_accuracy > best_agg_accuracy:
                best_agg_accuracy = agg_accuracy
                self.best_agg_epoch = i + 1
                self.save_model('Aggregate')

            if sel_accuracy > best_sel_accuracy:
                best_sel_accuracy = sel_accuracy
                self.best_sel_epoch = i + 1
                self.save_model('Select')

            if cond_accuracy > best_cond_accuracy:
                best_cond_accuracy = cond_accuracy
                self.best_cond_epoch = i + 1
                self.save_model('Condition')

            print(f"Best validation accuracy: {best_agg_accuracy, best_sel_accuracy, best_cond_accuracy}", 
                  f"Best epoch: {self.best_agg_epoch, self.best_sel_epoch, self.best_cond_epoch}")

    def save_model(self, model_type):
        torch.save(getattr(self.model, f"{model_type}Predictor").state_dict(), 
                   f"saved_models/{self.model_name}_best_{model_type.lower()}_model")
        if self.config['trainable_embedding'] == True:
            torch.save(getattr(self.model, f"{model_type.lower()}_embedding_layer").state_dict(), 
                       f"saved_models/{self.model_name}_best_{model_type.lower()}_embed_model")
    
model_trainer = ModelTrainer(model_config)
model_trainer.train()