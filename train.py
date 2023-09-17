import torch
import datetime
import threading
import time
import psutil
import GPUtil
from model.seq2sql import Seq2SQL
from model.sqlnet import SQLNet
from utils import *

model_config = {
    'model': 'SQLNet',              # SQLNet or Seq2SQL
    'num_epochs': 30,
    'batch_size': 64,
    'dataset_id': 0,                # 0 for original, 1 for re-split dataset, 2 for altered questions 1, 3 for altered questions 2, 4 for altered questions 3
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
        self.cpu_usage_list = []
        self.gpu_usage_list = []
        self.gpu_memory_usage_list = []
        self.memory_usage_list = []
        self.total_accuracy_list = [0]
        self.execution_time_list = []
        
        use_gpu = torch.cuda.is_available()
        print("Using CUDA GPU") if use_gpu else print("Using CPU")
        
        self.train_sql_data, self.train_table_data, self.val_sql_data, self.val_table_data, _, _ = load_dataset(config['dataset_id'])
        word_embedding = load_word_embeddings(config['embedding_path'], load_used=config['trainable_embedding'])
        
        if self.model_name == "Seq2SQL":
            self.model = Seq2SQL(word_embedding, 300, use_gpu=use_gpu)
        elif self.model_name == "SQLNet":
            self.model = SQLNet(word_embedding, 300, use_gpu=use_gpu, use_column_attention=config['use_column_attention'], 
                                trainable_embedding=config['trainable_embedding'])
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['learning_rate'], weight_decay = 0)
        self.best_agg_epoch, self.best_sel_epoch, self.best_cond_epoch = 0, 0, 0
    
    def sample_usage(self, sample_interval=1.0):
        while True:
            self.cpu_usage_list.append(psutil.cpu_percent())
            self.memory_usage_list.append(psutil.virtual_memory().used / (1024 ** 3))  # Convert to GB
                
            GPUs = GPUtil.getGPUs()
            if GPUs:
                self.gpu_usage_list.append(GPUs[0].load * 100)
                self.gpu_memory_usage_list.append(GPUs[0].memoryUsed / 1024)
                
            time.sleep(sample_interval)

    def train(self):
        usage_thread = threading.Thread(target=self.sample_usage, daemon=True)
        usage_thread.start()
        
        best_agg_accuracy, best_sel_accuracy, best_cond_accuracy = 0, 0, 0
        
        for i in range(self.num_epochs):
            print(f'Epoch: {i+1} / {datetime.datetime.now()}')
            print("\n")

            start_time = time.time()
            
            loss = epoch_train(self.model, self.optimizer, self.batch_size, self.train_sql_data, self.train_table_data)
            print(f"Loss: {loss}")
            print("\n")
 
            validation_accuracy = epoch_accuracy(self.model, self.batch_size, self.val_sql_data, self.val_table_data)
            print(f"Validation overall accuracy: {validation_accuracy[0]}\nBreakdown on (agg, sel, where): {validation_accuracy[1]}")
            print("\n")

            execution_time = time.time() - start_time
            self.execution_time_list.append(execution_time)
            
            self.total_accuracy_list.append(validation_accuracy[0])
            
            agg_accuracy, sel_accuracy, cond_accuracy = validation_accuracy[1]
            if agg_accuracy > best_agg_accuracy:
                best_agg_accuracy = agg_accuracy
                self.best_agg_epoch = i + 1
            if sel_accuracy > best_sel_accuracy:
                best_sel_accuracy = sel_accuracy
                self.best_sel_epoch = i + 1
            if cond_accuracy > best_cond_accuracy:
                best_cond_accuracy = cond_accuracy
                self.best_cond_epoch = i + 1
            
            print(f"Best validation accuracy: {best_agg_accuracy, best_sel_accuracy, best_cond_accuracy}", 
                  f"Best epoch: {self.best_agg_epoch, self.best_sel_epoch, self.best_cond_epoch}")
            print("\n")

        usage_thread.join(timeout=1)

model_trainer = ModelTrainer(model_config)
model_trainer.train()