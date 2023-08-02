import torch
import torch.nn as nn
import numpy as np
from modules.network_utils import run_lstm, encode_column_names
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

class SQLNetConditionPredictor(nn.Module):
    def __init__(self, 
                 word_dimensions: int, 
                 hidden_size: int, 
                 num_layers: int, 
                 max_column_num: int, 
                 max_token_num: int, 
                 use_column_attention: bool, 
                 use_gpu: bool):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_token_num = max_token_num
        self.max_column_num = max_column_num
        self.use_column_attention = use_column_attention
        self.device = torch.device('cuda') if use_gpu else torch.device('cpu')

        lstm_params = {
            'input_size': word_dimensions, 
            'hidden_size': hidden_size//2,
            'num_layers': num_layers, 
            'batch_first': True,
            'dropout': 0.3, 
            'bidirectional': True
        }

        self.cond_num_lstm = nn.LSTM(**lstm_params)
        self.cond_num_att = nn.Linear(hidden_size, 1)
        self.cond_num_out = nn.Sequential(nn.Linear(hidden_size, hidden_size), 
                                          nn.Tanh(), nn.Linear(hidden_size, 5))
        self.cond_num_name_enc = nn.LSTM(**lstm_params)
        self.cond_num_col_att = nn.Linear(hidden_size, 1)
        self.cond_num_col2hid1 = nn.Linear(hidden_size, 2*hidden_size)
        self.cond_num_col2hid2 = nn.Linear(hidden_size, 2*hidden_size)

        self.cond_col_lstm = nn.LSTM(**lstm_params)
        self.cond_col_att = self._column_attention_layer(hidden_size, use_column_attention)
        self.cond_col_name_enc = nn.LSTM(**lstm_params)
        self.cond_col_out_K = nn.Linear(hidden_size, hidden_size)
        self.cond_col_out_col = nn.Linear(hidden_size, hidden_size)
        self.cond_col_out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 1))

        self.cond_op_lstm = nn.LSTM(**lstm_params)
        self.cond_op_att = self._column_attention_layer(hidden_size, use_column_attention)
        self.cond_op_out_K = nn.Linear(hidden_size, hidden_size)
        self.cond_op_name_enc = nn.LSTM(**lstm_params)
        self.cond_op_out_col = nn.Linear(hidden_size, hidden_size)
        self.cond_op_out = nn.Sequential(nn.Linear(hidden_size, hidden_size), 
                                         nn.Tanh(), nn.Linear(hidden_size, 3))

        self.cond_str_lstm = nn.LSTM(**lstm_params)
        self.cond_str_decoder = nn.LSTM(input_size=self.max_token_num, hidden_size=hidden_size, 
                                        num_layers=num_layers, batch_first=True, dropout=0.3)
        self.cond_str_name_enc = nn.LSTM(**lstm_params)
        self.cond_str_out_g = nn.Linear(hidden_size, hidden_size)
        self.cond_str_out_h = nn.Linear(hidden_size, hidden_size)
        self.cond_str_out_col = nn.Linear(hidden_size, hidden_size)
        self.cond_str_out = nn.Sequential(nn.ReLU(), nn.Linear(hidden_size, 1))

        self.softmax = nn.Softmax(dim=-1)

        self.to(self.device)

    def _column_attention_layer(self, hidden_size: int, use_column_attention: bool) -> nn.Linear:
        if use_column_attention:
            return nn.Linear(hidden_size, hidden_size)
        else:
            return nn.Linear(hidden_size, 1)

    def generate_ground_truth_batch(self, token_sequences: List[List[List[int]]]) -> Tuple[torch.Tensor, np.ndarray]:
        batch_size = len(token_sequences)
        max_seq_length = max([max([len(token) for token in token_seq]+[0]) for 
                token_seq in token_sequences]) - 1
        if max_seq_length < 1:
            max_seq_length = 1

        result_array = np.zeros((batch_size, 4, max_seq_length, self.max_token_num), dtype=np.float32)
        result_length = np.zeros((batch_size, 4))

        for batch_index, token_seq in enumerate(token_sequences):
            sequence_index = 0
            for sequence_index, single_token_seq in enumerate(token_seq):
                output_token_seq = single_token_seq[:-1]
                result_length[batch_index, sequence_index] = len(output_token_seq)
                for token_index, token_id in enumerate(output_token_seq):
                    result_array[batch_index, sequence_index, token_index, token_id] = 1
            if sequence_index < 3:
                result_array[batch_index, sequence_index+1:, 0, 1] = 1
                result_length[batch_index, sequence_index+1:] = 1

        result_input = torch.from_numpy(result_array).to(self.device)

        return result_input, result_length

    def forward(self, 
                question_embedding: torch.Tensor, 
                question_length: List[int], 
                column_name_embeddings: torch.Tensor, 
                column_name_length: List[int], 
                column_lengths: List[int], 
                num_columns: int, 
                ground_truth_where: torch.Tensor, 
                ground_truth_condition: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        
        batch_size = len(question_length)
        max_question_length = max(question_length)

        condition_number_scores = self.predict_condition_num(question_embedding, question_length, 
                                                             column_name_embeddings, column_name_length, 
                                                             column_lengths, num_columns, batch_size, max_question_length)
        
        condition_column_scores = self.predict_cond_col(question_embedding, question_length, 
                                                        column_name_embeddings, column_name_length, 
                                                        column_lengths, batch_size, max_question_length)
        
        chosen_column_ground_truth, condition_operator_scores = self.predict_cond_op(question_embedding, question_length, 
                                                                                     condition_number_scores, 
                                                                                     condition_column_scores, 
                                                                                     column_name_embeddings, 
                                                                                     column_name_length, column_lengths, 
                                                                                     ground_truth_condition, batch_size, 
                                                                                     max_question_length)
        
        condition_string_scores = self.predict_cond_str(question_embedding, question_length, 
                                                        column_name_embeddings, column_name_length, column_lengths, 
                                                        ground_truth_where, chosen_column_ground_truth, batch_size, 
                                                        max_question_length)

        return (condition_number_scores, condition_column_scores, condition_operator_scores, condition_string_scores)

    def predict_condition_num(self,
                              question_embedding: torch.Tensor, 
                              question_lengths: List[int], 
                              column_name_embeddings: torch.Tensor, 
                              column_name_lengths: List[int], 
                              column_lengths: List[int], 
                              num_columns: int, 
                              batch_size: int, 
                              max_question_length: int) -> torch.Tensor:
        encoded_num_col, col_num = encode_column_names(column_name_embeddings, column_name_lengths,
                column_lengths, self.cond_num_name_enc)
        num_col_attention_values = self.cond_num_col_att(encoded_num_col).squeeze()
        for idx, num in enumerate(num_columns):
            if num < max(num_columns):
                num_col_attention_values[idx, num:].fill_(-100)
        num_col_attention = self.softmax(num_col_attention_values)
        num_col_keys = (encoded_num_col * num_col_attention.unsqueeze(2)).sum(1)
        cond_num_hidden_1 = self.cond_num_col2hid1(num_col_keys).view(
                batch_size, 4, self.hidden_size//2).transpose(0, 1).contiguous()
        cond_num_hidden_2 = self.cond_num_col2hid2(num_col_keys).view(
                batch_size, 4, self.hidden_size//2).transpose(0, 1).contiguous()

        num_enc_hidden, _ = run_lstm(self.cond_num_lstm, question_embedding, question_lengths,
                hidden=(cond_num_hidden_1, cond_num_hidden_2))

        num_attention_values = self.cond_num_att(num_enc_hidden).squeeze()

        for idx, num in enumerate(question_lengths):
            if num < max_question_length:
                num_attention_values[idx, num:].fill_(-100)
        num_attention = self.softmax(num_attention_values)

        num_keys = (num_enc_hidden * num_attention.unsqueeze(2).expand_as(
            num_enc_hidden)).sum(1)
        cond_num_score = self.cond_num_out(num_keys)

        return cond_num_score

    def predict_cond_col(self, 
                         question_emb: torch.Tensor, 
                         question_len: List[int], 
                         column_input: torch.Tensor, 
                         column_name_len: List[int], 
                         column_len: List[int], 
                         batch_size: int, 
                         max_input_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        encoded_col, col_num = encode_column_names(column_input, column_name_len, column_len, self.cond_col_name_enc)

        encoded_col_lstm, _ = run_lstm(self.cond_col_lstm, question_emb, question_len)
        if self.use_column_attention:
            col_attention_values = torch.bmm(encoded_col,
                    self.cond_col_att(encoded_col_lstm).transpose(1, 2))
            for idx, num in enumerate(question_len):
                if num < max_input_len:
                    col_attention_values[idx, :, num:].fill_(-100)
            col_attention = self.softmax(col_attention_values.view(
                (-1, max_input_len))).view(batch_size, -1, max_input_len)
            cond_col_key = (encoded_col_lstm.unsqueeze(1) * col_attention.unsqueeze(3)).sum(2)
        else:
            col_attention_values = self.cond_col_att(encoded_col_lstm).squeeze()
            for idx, num in enumerate(question_len):
                if num < max_input_len:
                    col_attention_values[idx, num:].fill_(-100)
            col_attention = self.softmax(col_attention_values)
            cond_col_key = (encoded_col_lstm *
                    col_attention_values.unsqueeze(2)).sum(1).unsqueeze(1)

        cond_col_score = self.cond_col_out(self.cond_col_out_K(cond_col_key) +
                self.cond_col_out_col(encoded_col)).squeeze()
        max_column_num = max(col_num)
        for b, num in enumerate(col_num):
            if num < max_column_num:
                cond_col_score[b, num:].fill_(-100)

        return cond_col_score

    def predict_cond_op(self, 
                    question_emb: torch.Tensor, 
                    question_len: List[int], 
                    condition_num_score: torch.Tensor, 
                    condition_col_score: torch.Tensor, 
                    column_input: torch.Tensor, 
                    column_name_len: List[int], 
                    column_len: List[int], 
                    ground_truth_cond: Optional[List[List[Tuple[int]]]], 
                    batch_size: int, 
                    max_input_len: int) -> Tuple[List[List[int]], torch.Tensor]:
    
        chosen_columns_gt = []
        if ground_truth_cond is None:
            cond_nums = np.argmax(condition_num_score.cpu().detach().numpy(), axis=1)
            col_scores = condition_col_score.cpu().detach().numpy()
            chosen_columns_gt = [list(np.argsort(-col_scores[b])[:cond_nums[b]])
                    for b in range(len(cond_nums))]
        else:
            chosen_columns_gt = [[x[0] for x in one_gt_cond] for one_gt_cond in ground_truth_cond]

        encoded_col, _ = encode_column_names(column_input, column_name_len,
                column_len, self.cond_op_name_enc)
        column_embeddings = []
        for b in range(batch_size):
            cur_col_emb = torch.stack([encoded_col[b, x] 
                for x in chosen_columns_gt[b]] + [encoded_col[b, 0]] *
                (4 - len(chosen_columns_gt[b])))
            column_embeddings.append(cur_col_emb)
        column_embeddings = torch.stack(column_embeddings)

        op_lstm_enc, _ = run_lstm(self.cond_op_lstm, question_emb, question_len)
        if self.use_column_attention:
            op_att_val = torch.matmul(self.cond_op_att(op_lstm_enc).unsqueeze(1),
                    column_embeddings.unsqueeze(3)).squeeze()
            for idx, num in enumerate(question_len):
                if num < max_input_len:
                    op_att_val[idx, :, num:].fill_(-100)
            op_attention = self.softmax(op_att_val.view(batch_size*4, -1)).view(batch_size, 4, -1)
            cond_op_key = (op_lstm_enc.unsqueeze(1) * op_attention.unsqueeze(3)).sum(2)
        else:
            op_att_val = self.cond_op_att(op_lstm_enc).squeeze()
            for idx, num in enumerate(question_len):
                if num < max_input_len:
                    op_att_val[idx, num:].fill_(-100)
            op_attention = self.softmax(op_att_val)
            cond_op_key = (op_lstm_enc * op_attention.unsqueeze(2)).sum(1).unsqueeze(1)

        cond_op_score = self.cond_op_out(self.cond_op_out_K(cond_op_key) +
                self.cond_op_out_col(column_embeddings)).squeeze()
        
        return chosen_columns_gt, cond_op_score

    def predict_cond_str(self, 
                        question_emb: torch.Tensor, 
                        question_len: List[int],
                        column_input: torch.Tensor, 
                        column_name_len: List[int], 
                        column_len: List[int], 
                        ground_truth_where: Optional[List[List[str]]], 
                        chosen_columns_gt: List[List[int]], 
                        batch_size: int, 
                        max_input_len: int) -> torch.Tensor:
        str_lstm_enc, _ = run_lstm(self.cond_str_lstm, question_emb, question_len)
        encoded_col, _ = encode_column_names(column_input, column_name_len,
                column_len, self.cond_str_name_enc)
        column_embeddings = []
        for b in range(batch_size):
            cur_col_emb = torch.stack([encoded_col[b, x]
                for x in chosen_columns_gt[b]] +
                [encoded_col[b, 0]] * (4 - len(chosen_columns_gt[b])))
            column_embeddings.append(cur_col_emb)
        column_embeddings = torch.stack(column_embeddings)

        if ground_truth_where is not None:
            gt_tok_seq, gt_tok_len = self.generate_ground_truth_batch(ground_truth_where)
            str_decoder_out_flat, _ = self.cond_str_decoder(
                    gt_tok_seq.view(batch_size*4, -1, self.max_token_num))
            str_decoder_out = str_decoder_out_flat.contiguous().view(batch_size, 4, -1, self.hidden_size)

            str_lstm_extended = str_lstm_enc.unsqueeze(1).unsqueeze(1)
            str_decoder_extended = str_decoder_out.unsqueeze(3)
            column_emb_extended = column_embeddings.unsqueeze(2).unsqueeze(2)

            cond_str_score = self.cond_str_out(
                    self.cond_str_out_h(str_lstm_extended) + self.cond_str_out_g(str_decoder_extended) +
                    self.cond_str_out_col(column_emb_extended)).squeeze()
            for b, num in enumerate(question_len):
                if num < max_input_len:
                    cond_str_score[b, :, :, num:].fill_(-100)
        else:
            str_lstm_extended = str_lstm_enc.unsqueeze(1).unsqueeze(1)
            column_emb_extended = column_embeddings.unsqueeze(2).unsqueeze(2)
            scores = []

            t = 0
            cur_inp = torch.zeros((batch_size*4, 1, self.max_token_num), dtype=torch.float32, device=self.device)
            cur_inp[:,0,0].fill_(1)

            current_h = None
            while t < 50:
                if current_h:
                    str_decoder_out_flat, current_h = self.cond_str_decoder(cur_inp, current_h)
                else:
                    str_decoder_out_flat, current_h = self.cond_str_decoder(cur_inp)
                str_decoder_out = str_decoder_out_flat.view(batch_size, 4, 1, self.hidden_size)
                str_decoder_extended = str_decoder_out.unsqueeze(3)

                cur_cond_str_score = self.cond_str_out(
                        self.cond_str_out_h(str_lstm_extended) + self.cond_str_out_g(str_decoder_extended)
                        + self.cond_str_out_col(column_emb_extended)).squeeze()
                for b, num in enumerate(question_len):
                    if num < max_input_len:
                        cur_cond_str_score[b, :, num:].fill_(-100)
                scores.append(cur_cond_str_score)

                _, ans_tok_var = cur_cond_str_score.view(batch_size*4, max_input_len).max(1)
                ans_tok = ans_tok_var.detach()
                cur_inp = torch.zeros((batch_size*4, self.max_token_num), device=self.device).scatter_(
                        1, ans_tok.unsqueeze(1), 1)
                cur_inp = cur_inp.unsqueeze(1)

                t += 1

            cond_str_score = torch.stack(scores, 2)
            for b, num in enumerate(question_len):
                if num < max_input_len:
                    cond_str_score[b, :, :, num:].fill_(-100)

        return cond_str_score
