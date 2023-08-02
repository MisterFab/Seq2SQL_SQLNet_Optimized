import torch
import torch.nn as nn
from modules.network_utils import run_lstm, encode_column_names
from typing import List

class AggregatePredictor(nn.Module):
    def __init__(self, word_dimensions: int, hidden_size: int, num_layers: int, use_gpu: bool = False, use_column_attention: bool = False):
        super().__init__()

        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.use_column_attention = use_column_attention

        self.aggregate_lstm = nn.LSTM(input_size=word_dimensions, hidden_size=hidden_size // 2,
                                      num_layers=num_layers, batch_first=True,
                                      dropout=0.3, bidirectional=True)

        if use_column_attention:
            self.aggregate_column_name_encoded = nn.LSTM(input_size=word_dimensions,
                                                         hidden_size=hidden_size // 2, num_layers=num_layers,
                                                         batch_first=True, dropout=0.3, bidirectional=True)
            self.aggregate_attention = nn.Linear(hidden_size, hidden_size)
        else:
            self.aggregate_attention = nn.Linear(hidden_size, 1)

        self.aggregate_output = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 6)
        )
        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)

    def forward(self, 
                question_embedded: torch.Tensor, 
                question_lengths: List[int], 
                column_input: torch.Tensor,
                column_name_lengths: List[int], 
                column_lengths: List[int], 
                ground_truth_selection: List[int]) -> torch.Tensor:
        
        max_question_len = max(question_lengths)
        encoded_question, _ = run_lstm(self.aggregate_lstm, question_embedded, question_lengths)

        if self.use_column_attention:
            encoded_column, _ = encode_column_names(column_input, column_name_lengths, column_lengths, self.aggregate_column_name_encoded)
            chosen_sel_indices = torch.tensor(ground_truth_selection, dtype=torch.long, device=self.device)
            batch_indices = torch.arange(len(chosen_sel_indices), dtype=torch.long, device=self.device)
            chosen_encoded_column = encoded_column[batch_indices, chosen_sel_indices]
            attention_values = torch.bmm(self.aggregate_attention(encoded_question), chosen_encoded_column.unsqueeze(2)).squeeze()
        else:
            attention_values = self.aggregate_attention(encoded_question).squeeze()

        for idx, question_len in enumerate(question_lengths):
            if question_len < max_question_len:
                attention_values[idx, question_len:].fill_(-100)

        attention_weights = self.softmax(attention_values)
        weighted_encoding = (encoded_question * attention_weights.unsqueeze(2).expand_as(encoded_question)).sum(1)
        agg_scores = self.aggregate_output(weighted_encoding)

        return agg_scores