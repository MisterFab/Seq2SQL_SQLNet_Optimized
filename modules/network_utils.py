import torch
import numpy as np
from torch import nn
from typing import Tuple, List

def run_lstm(lstm_network: nn.LSTM, input_tensor: torch.Tensor, input_lengths: List[int], hidden = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    device = input_tensor.device

    sorted_indices = np.argsort(-input_lengths)
    sorted_lengths = input_lengths[sorted_indices]
    sorted_indices = torch.tensor(sorted_indices, device=device)
    sorted_tensor = input_tensor[sorted_indices]

    packed_input = nn.utils.rnn.pack_padded_sequence(sorted_tensor, sorted_lengths, batch_first=True)
    hidden_state = (hidden[0][:, sorted_indices], hidden[1][:, sorted_indices]) if hidden is not None else None

    sorted_sequences_output, sorted_hidden_states = lstm_network(packed_input, hidden_state)
    inverse_sorted_indices = torch.argsort(sorted_indices)

    sequences_output = nn.utils.rnn.pad_packed_sequence(sorted_sequences_output, batch_first=True)[0][inverse_sorted_indices]
    hidden_states = (sorted_hidden_states[0][:, inverse_sorted_indices], sorted_hidden_states[1][:, inverse_sorted_indices])

    return sequences_output, hidden_states

def encode_column_names(name_input_var: torch.Tensor, name_lengths: List[int], column_lengths: List[int], encoder_lstm: nn.LSTM) -> Tuple[torch.Tensor, List[int]]:
    device = name_input_var.device

    name_hidden, _ = run_lstm(encoder_lstm, name_input_var, name_lengths)
    name_output = name_hidden[torch.arange(len(name_lengths)), name_lengths-1]

    encoded_output = torch.zeros((len(column_lengths), max(column_lengths), name_output.size()[1]), device=device)

    start_index = 0
    for idx, current_length in enumerate(column_lengths):
        encoded_output[idx, :current_length] = name_output[start_index : start_index+current_length]
        start_index += current_length

    return encoded_output, column_lengths