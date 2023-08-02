import torch.nn as nn
import numpy as np
from modules.network_utils import run_lstm
from typing import List, Tuple, Optional, Union
import torch

class ConditionPredictor(nn.Module):
    def __init__(self, word_dimensions: int, hidden_size: int, num_layers: int, max_token_num: int, use_gpu: bool):
        super().__init__()

        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.hidden_dim = hidden_size
        self.max_token_num = max_token_num

        self.condition_lstm = nn.LSTM(input_size=word_dimensions, hidden_size=hidden_size//2,
                                      num_layers=num_layers, batch_first=True,
                                      dropout=0.3, bidirectional=True)

        self.condition_decoder = nn.LSTM(input_size=max_token_num,
                                         hidden_size=hidden_size, num_layers=num_layers,
                                         batch_first=True, dropout=0.3)

        self.condition_out_g = nn.Linear(hidden_size, hidden_size)
        self.condition_out_h = nn.Linear(hidden_size, hidden_size)
        self.condition_out = nn.Sequential(nn.Tanh(), nn.Linear(hidden_size, 1))

        self.softmax = nn.Softmax(dim=1)

        self.to(self.device)

    def forward(self, 
                question_embedded: torch.Tensor, 
                question_lengths: List[int], 
                ground_truth_sequences: Optional[List[List[str]]] = None) -> torch.Tensor:
        
        max_question_length = max(question_lengths)
        encoded_question, hidden_state = run_lstm(self.condition_lstm, question_embedded, question_lengths)
        decoder_hidden_state = tuple(torch.cat((state[:2], state[2:]),dim=2) for state in hidden_state)

        if ground_truth_sequences is not None:
            ground_truth_tokens, ground_truth_lengths = self.generate_ground_truth_batch(ground_truth_sequences, generate_input=True)
            decoder_output, _ = run_lstm(self.condition_decoder, ground_truth_tokens, ground_truth_lengths, decoder_hidden_state)
            condition_scores = self.calculate_condition_scores(encoded_question, decoder_output, question_lengths, max_question_length)
        else:
            condition_scores = self.perform_autoregressive_decode(encoded_question, decoder_hidden_state, question_lengths, max_question_length)
        
        return condition_scores

    def calculate_condition_scores(self, 
                                encoded_question: torch.Tensor, 
                                decoder_output: torch.Tensor, 
                                question_lengths: List[int], 
                                max_question_length: int) -> torch.Tensor:
        expanded_encoded_question = encoded_question.unsqueeze(1)
        expanded_decoder_output = decoder_output.unsqueeze(2)

        condition_scores = self.condition_out(self.condition_out_h(expanded_encoded_question) + self.condition_out_g(expanded_decoder_output)).squeeze()

        for idx, num in enumerate(question_lengths):
            if num < max_question_length:
                condition_scores[idx, :, num:].fill_(-100)

        return condition_scores 

    def perform_autoregressive_decode(self, 
                                    encoded_question: torch.Tensor, 
                                    decoder_hidden_state: Tuple[torch.Tensor, torch.Tensor], 
                                    question_lengths: List[int], 
                                    max_question_length: int) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.distributions.Categorical]]]:
        batch_size = len(question_lengths)
        expanded_encoded_question = encoded_question.unsqueeze(1)

        decoded_scores = []
        completed_set = set()
        time_step = 0

        current_input = torch.zeros((batch_size, 1, self.max_token_num), dtype=torch.float32, device=self.device)
        current_input[:,0,7] = 1
        current_hidden_state = decoder_hidden_state
        output_hidden_state = self.condition_out_h(expanded_encoded_question)

        while len(completed_set) < batch_size and time_step < 100:
            decoder_output, current_hidden_state = self.condition_decoder(current_input, current_hidden_state)
            expanded_decoder_output = decoder_output.unsqueeze(2)
            current_condition_score = self.condition_out(output_hidden_state + self.condition_out_g(expanded_decoder_output)).squeeze()

            for idx, question_length in enumerate(question_lengths):
                if question_length < max_question_length:
                    current_condition_score[idx, question_length:].fill_(-100)

            decoded_scores.append(current_condition_score)

            _, answer_token_var = current_condition_score.view(batch_size, max_question_length).max(1)

            answer_token = answer_token_var.unsqueeze(1).detach()

            current_input = torch.zeros((batch_size, self.max_token_num), device=self.device).zero_().scatter_(1, answer_token, 1)
            current_input = current_input.unsqueeze(1)

            for idx, token in enumerate(answer_token_var.squeeze()):
                if token == 1:
                    completed_set.add(idx)
            time_step += 1

        condition_scores = torch.stack(decoded_scores, 1)

        return condition_scores

    def generate_ground_truth_batch(self, 
                                    token_sequences: List[List[int]], 
                                    generate_input: bool = True) -> Tuple[torch.Tensor, np.array]:
        batch_size = len(token_sequences)
        sequence_lengths = np.array([len(sequence) - 1 for sequence in token_sequences])
        max_sequence_length = max(sequence_lengths)

        if generate_input:
            token_sequences = [sequence[:-1] for sequence in token_sequences]
        else:
            token_sequences = [sequence[1:] for sequence in token_sequences]

        one_hot_tensor = torch.zeros((batch_size, max_sequence_length, self.max_token_num), device=self.device)

        for idx, sequence in enumerate(token_sequences):
            for t, token in enumerate(sequence):
                one_hot_tensor[idx, t, token].fill_(1)

        return one_hot_tensor, sequence_lengths