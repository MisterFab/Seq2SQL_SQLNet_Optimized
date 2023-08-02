import torch
import torch.nn as nn
from modules.network_utils import run_lstm, encode_column_names
from typing import List

class SelectPredictor(nn.Module):
    def __init__(self, word_dimensions: int, hidden_size: int, num_layers: int, use_gpu: bool = False, use_column_attention: bool = False):
        super().__init__()

        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.use_column_attention = use_column_attention

        self.select_lstm = nn.LSTM(input_size=word_dimensions, hidden_size=hidden_size//2,
                                   num_layers=num_layers, batch_first=True,
                                   dropout=0.3, bidirectional=True)
        if use_column_attention:
            self.select_attention = nn.Linear(hidden_size, hidden_size)
        else:
            self.select_attention = nn.Linear(hidden_size, 1)
        self.select_column_name_encoder = nn.LSTM(input_size=word_dimensions, hidden_size=hidden_size//2,
                                                  num_layers=num_layers, batch_first=True,
                                                  dropout=0.3, bidirectional=True)
        self.select_out_K = nn.Linear(hidden_size, hidden_size)
        self.select_out_col = nn.Linear(hidden_size, hidden_size)
        self.select_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_size, 1))
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)

    def forward(self, 
                question_embeddings: torch.Tensor, 
                question_lengths: List[int], 
                column_input: torch.Tensor,
                column_name_lengths: List[int], 
                column_lengths: List[int], 
                num_of_columns: List[int]) -> torch.Tensor:

        batch_size = len(question_embeddings)
        max_question_len = max(question_lengths)

        encoded_columns, _ = encode_column_names(column_input, column_name_lengths, column_lengths, self.select_column_name_encoder)
        encoded_questions, _ = run_lstm(self.select_lstm, question_embeddings, question_lengths)

        if self.use_column_attention:
            attention_values = torch.bmm(encoded_columns, self.select_attention(encoded_questions).transpose(1, 2))

            for idx, question_len in enumerate(question_lengths):
                if question_len < max_question_len:
                    attention_values[idx, :, question_len:].fill_(-100)

            attention_weights = self.softmax(attention_values.view((-1, max_question_len))).view(batch_size, -1, max_question_len)
            encoded_question_weighted = (encoded_questions.unsqueeze(1) * attention_weights.unsqueeze(3)).sum(2)

        else:
            attention_values = self.select_attention(encoded_questions).squeeze()

            for idx, question_len in enumerate(question_lengths):
                if question_len < max_question_len:
                    attention_values[idx, question_len:].fill_(-100)

            attention_weights = self.softmax(attention_values)
            encoded_question_weighted = (encoded_questions * attention_weights.unsqueeze(2).expand_as(encoded_questions)).sum(1)
            encoded_question_weighted = encoded_question_weighted.unsqueeze(1)

        selection_scores = self.select_out(self.select_out_K(encoded_question_weighted) + self.select_out_col(encoded_columns)).squeeze()

        max_num_of_columns = max(num_of_columns)

        for idx, num in enumerate(num_of_columns):
            if num < max_num_of_columns:
                selection_scores[idx, num:].fill_(-100)

        return selection_scores