import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple


class WordEmbedding(nn.Module):
    def __init__(self,
                 word_embedding: dict,
                 embedding_dimension: int,
                 use_gpu: bool,
                 sql_tokens: List[str],
                 SQLNet: bool,
                 trainable: bool = False):
        super().__init__()

        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')
        self.trainable = trainable
        self.embedding_dimension = embedding_dimension
        self.sql_tokens = sql_tokens
        self.SQLNet = SQLNet

        if trainable:
            self.w2i, word_emb_val = word_embedding
            self.embedding = nn.Embedding(len(self.w2i), embedding_dimension)
            self.embedding.weight = nn.Parameter(torch.from_numpy(word_emb_val.astype(np.float32)))
        else:
            self.word_embedding = word_embedding
        
        self.to(self.device)

    def generate_batch(self,
                       questions: List[List[str]],
                       columns: List[List[List[str]]]) -> Tuple[torch.Tensor, np.array]:
        batch_size = len(questions)
        values_embeddings = []
        values_lengths = np.zeros(batch_size, dtype=np.int64)

        for i, (question, column) in enumerate(zip(questions, columns)):
            if self.trainable:
                question_embedding = [self.w2i.get(word, 0) for word in question]
            else:
                question_embedding = [self.word_embedding.get(word, np.zeros(self.embedding_dimension))
                                      for word in question]

            if self.SQLNet:
                if self.trainable:
                    values_embeddings.append([1] + question_embedding + [2])
                else:
                    values_embeddings.append(
                        [np.zeros(self.embedding_dimension)] + question_embedding + [np.zeros(self.embedding_dimension)])
                values_lengths[i] = 1 + len(question_embedding) + 1

            else:
                one_col_all = [x for toks in column for x in toks + [',']]
                if self.trainable:
                    column_embedding = [self.w2i.get(word, 0) for word in one_col_all]
                    values_embeddings.append(
                        [0 for _ in self.sql_tokens] + column_embedding + [0] + question_embedding + [0])
                else:
                    column_embedding = [self.word_embedding.get(word, np.zeros(self.embedding_dimension))
                                        for word in one_col_all]
                    values_embeddings.append(
                        [np.zeros(self.embedding_dimension) for _ in self.sql_tokens] + column_embedding +
                        [np.zeros(self.embedding_dimension)] + question_embedding + [np.zeros(self.embedding_dimension)])
                values_lengths[i] = len(self.sql_tokens) + len(column_embedding) + 1 + len(question_embedding) + 1

        max_length = max(values_lengths)
        embedding_tensor = self._create_embedding_tensor(batch_size, max_length, values_embeddings)

        return embedding_tensor, values_lengths

    def generate_column_batch(self, columns: List[List[str]]) -> Tuple[torch.Tensor, np.array, np.array]:
        column_lengths = np.array([len(column) for column in columns])
        names = [name for column in columns for name in column]

        embeding_tensor, name_lenghts = self.str_list_to_batch(names)
        return embeding_tensor, name_lenghts, column_lengths

    def str_list_to_batch(self, str_list: List[str]) -> Tuple[torch.Tensor, np.array]:
        batch_size = len(str_list)

        values_embeddings = []
        name_lenghts = np.zeros(batch_size, dtype=np.int64)

        for i, one_str in enumerate(str_list):
            if self.trainable:
                embedding = [self.w2i.get(x, 0) for x in one_str]
            else:
                embedding = [self.word_embedding.get(x, np.zeros(self.embedding_dimension)) for x in one_str]
            values_embeddings.append(embedding)
            name_lenghts[i] = len(embedding)

        max_length = max(name_lenghts)
        embedding_tensor = self._create_embedding_tensor(batch_size, max_length, values_embeddings)

        return embedding_tensor, name_lenghts

    def _create_embedding_tensor(self, batch_size: int, max_length: int, values_embeddings: List[List[str]]):
        if self.trainable:
            embedding_tensor = np.zeros((batch_size, max_length), dtype=np.int64)
            for i in range(batch_size):
                for t, value in enumerate(values_embeddings[i]):
                    embedding_tensor[i, t] = value
            embedding_tensor = torch.from_numpy(embedding_tensor).to(self.device)
            embedding_tensor = self.embedding(embedding_tensor)
        else:
            embedding_tensor = np.zeros((batch_size, max_length, self.embedding_dimension), dtype=np.float32)
            for i in range(batch_size):
                for t, value in enumerate(values_embeddings[i]):
                    embedding_tensor[i, t, :] = value
            embedding_tensor = torch.from_numpy(embedding_tensor).to(self.device)
        
        return embedding_tensor