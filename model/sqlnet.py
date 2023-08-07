import torch
import torch.nn as nn
import numpy as np
from modules.word_embedding import WordEmbedding
from modules.aggregator_predict import AggregatePredictor
from modules.selection_predict import SelectPredictor
from modules.sqlnet_condition_predict import SQLNetConditionPredictor
from typing import List, Tuple, Dict, Union, Optional, Any

class SQLNet(nn.Module):
    def __init__(self, 
                 word_embeddings: torch.Tensor, 
                 word_dimensions: int, 
                 hidden_size: int = 100, 
                 num_layers: int = 2, 
                 use_gpu: bool = False,
                 use_column_attention: bool = False, 
                 trainable_embedding: bool = False) -> None:
        super().__init__()

        self.device = torch.device('cuda' if use_gpu else 'cpu')
        self.trainable_embedding = trainable_embedding
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.max_column_num = 45
        self.max_token_num = 200
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        if trainable_embedding:
            self.aggregate_embedding_layer = WordEmbedding(word_embeddings, word_dimensions, use_gpu,
                                                           self.SQL_TOK, True, trainable=trainable_embedding)
            self.select_embedding_layer = WordEmbedding(word_embeddings, word_dimensions, use_gpu,
                                                        self.SQL_TOK, True, trainable=trainable_embedding)
            self.condition_embedding_layer = WordEmbedding(word_embeddings, word_dimensions, use_gpu,
                                                           self.SQL_TOK, True, trainable=trainable_embedding)
        else:   
            self.embed_layer = WordEmbedding(word_embeddings, word_dimensions, use_gpu, self.SQL_TOK, True)

        self.AggregatePredictor = AggregatePredictor(word_dimensions, hidden_size, num_layers, use_gpu=use_gpu, 
                                                     use_column_attention=use_column_attention)
        self.SelectPredictor = SelectPredictor(word_dimensions, hidden_size, num_layers, 
                                               use_gpu=use_gpu, use_column_attention=use_column_attention)
        self.ConditionPredictor = SQLNetConditionPredictor(word_dimensions, hidden_size, num_layers, self.max_column_num,
                                                           self.max_token_num, use_gpu=use_gpu, use_column_attention=use_column_attention)

        self.crossentropyloss = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=-1)
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.bce_logit = nn.BCEWithLogitsLoss()

    def _find_operator_position(self, query_substring: str) -> int:
        operators = ['EQL', 'GT', 'LT']
        for operator in operators:
            if operator in query_substring:
                return query_substring.index(operator)
        raise RuntimeError("No operator in it!")

    def _get_sequence_indices(self, all_tokens: List[str], token_str: List[str]) -> List[int]:
        return [all_tokens.index(token) if token in all_tokens else 0 for token in token_str]

    def generate_gt_where_seq(self, q: List[List[str]], col: List[List[str]], query: List[str]) -> List[List[List[int]]]:
        return_sequences = []
        
        for question, column, cur_query in zip(q, col, query):
            current_values = []
            start_idx = cur_query.index('WHERE')+1 if 'WHERE' in cur_query else len(cur_query)
            all_tokens = ['<BEG>'] + question + ['<END>']
            
            while start_idx < len(cur_query):
                end_idx = cur_query[start_idx:].index('AND') + start_idx if 'AND' in cur_query[start_idx:] else len(cur_query)
                operator_idx = self._find_operator_position(cur_query[start_idx:end_idx]) + start_idx
                
                tokens_str = ['<BEG>'] + cur_query[operator_idx+1:end_idx] + ['<END>']
                current_sequence = self._get_sequence_indices(all_tokens, tokens_str)
                current_values.append(current_sequence)
                start_idx = end_idx+1

            return_sequences.append(current_values)

        return return_sequences

    def forward(self, 
            question: List[List[str]], 
            column_data: List[List[List[str]]], 
            column_numbers: List[int], 
            ground_truth_where: Optional[List[List[str]]] = None,
            ground_truth_condition = None,
            ground_truth_selection = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.trainable_embedding:
            question_embedding, question_length = self.aggregate_embedding_layer.generate_batch(question, column_data)
            column_input, column_name_length, column_length = self.aggregate_embedding_layer.generate_column_batch(column_data)
            aggregate_score = self.AggregatePredictor(question_embedding, question_length, column_input, column_name_length, 
                                                      column_length, ground_truth_selection)

            question_embedding, question_length = self.select_embedding_layer.generate_batch(question, column_data)
            column_input, column_name_length, column_length = self.select_embedding_layer.generate_column_batch(column_data)
            select_score = self.SelectPredictor(question_embedding, question_length, column_input, column_name_length, 
                                                column_length, column_numbers)

            question_embedding, question_length = self.condition_embedding_layer.generate_batch(question, column_data)
            column_input, column_name_length, column_length = self.condition_embedding_layer.generate_column_batch(column_data)
            condition_score = self.ConditionPredictor(question_embedding, question_length, column_input, column_name_length, 
                                                      column_length, column_numbers, ground_truth_where, ground_truth_condition)

        else:
            question_embedding, question_length = self.embed_layer.generate_batch(question, column_data)
            column_input, column_name_length, column_length = self.embed_layer.generate_column_batch(column_data)

            aggregate_score = self.AggregatePredictor(question_embedding, question_length, column_input, column_name_length, 
                                                      column_length, ground_truth_selection)
            select_score = self.SelectPredictor(question_embedding, question_length, column_input, column_name_length, 
                                                column_length, column_numbers)
            condition_score = self.ConditionPredictor(question_embedding, question_length, column_input, column_name_length, 
                                                      column_length, column_numbers, ground_truth_where, ground_truth_condition)

        return aggregate_score, select_score, condition_score

    def loss(self, 
            score: Union[torch.Tensor, List[torch.Tensor]], 
            truth_values: List[List[Union[int, List[int]]]], 
            ground_truth_conditions: List[List[List[Union[int, str]]]]) -> torch.Tensor:
        
        aggregate_score, select_score, condition_scores = score

        num_conditions_score, column_condition_score, operator_condition_score, string_condition_score = condition_scores

        total_loss = 0
        batch_size = len(truth_values)
        num_conditions = len(column_condition_score[0])

        for score, index in zip([aggregate_score, select_score, num_conditions_score], [0, 1, 2]):
            ground_truth = torch.tensor([x[index] for x in truth_values], device=self.device)
            total_loss += self.crossentropyloss(score, ground_truth)

        truth_probability = np.zeros((batch_size, num_conditions), dtype=np.float32)
        for batch_index, truth_value in enumerate(truth_values):
            if truth_value[3]:
                truth_probability[batch_index][list(truth_value[3])] = 1

        column_condition_truth = torch.from_numpy(truth_probability).to(self.device)
        sigmoid = nn.Sigmoid()
        column_condition_probability = sigmoid(column_condition_score)
        bce_loss = -torch.mean(3 * (column_condition_truth * torch.log(column_condition_probability + 1e-10)) +
                                (1 - column_condition_truth) * torch.log(1 - column_condition_probability + 1e-10))
        total_loss += bce_loss

        for batch_index, truth_value in enumerate(truth_values):
            if truth_value[4]:
                operator_condition_truth = torch.tensor(truth_value[4], device=self.device)
                operator_condition_prediction = operator_condition_score[batch_index, :len(truth_value[4])]
                total_loss += self.crossentropyloss(operator_condition_prediction, operator_condition_truth) / len(truth_values)

        for batch_index, conditions in enumerate(ground_truth_conditions):
            for condition_index, string_condition_truth in enumerate(conditions):
                if len(string_condition_truth) > 1:
                    string_condition_truth_variable = torch.tensor(string_condition_truth[1:], device=self.device)
                    string_condition_end = len(string_condition_truth) - 1
                    string_condition_prediction = string_condition_score[batch_index, condition_index, :string_condition_end]
                    total_loss += self.crossentropyloss(string_condition_prediction, string_condition_truth_variable) / (len(ground_truth_conditions) * len(conditions))

        return total_loss

    def check_accuracy(self, predicted_queries, ground_truth_queries):
        total_errors = aggregation_errors = selection_errors = condition_errors = 0.0

        for predicted_query, ground_truth_query in zip(predicted_queries, ground_truth_queries):
            is_query_correct = True

            if predicted_query['agg'] != ground_truth_query['agg']:
                aggregation_errors += 1
                is_query_correct = False

            if predicted_query['sel'] != ground_truth_query['sel']:
                selection_errors += 1
                is_query_correct = False

            predicted_conditions, ground_truth_conditions = predicted_query['conds'], ground_truth_query['conds']
            is_conditions_correct = self.check_conditions(predicted_conditions, ground_truth_conditions)

            if not is_conditions_correct:
                condition_errors += 1
                is_query_correct = False

            if not is_query_correct:
                total_errors += 1

        return np.array((aggregation_errors, selection_errors, condition_errors)), total_errors

    def check_conditions(self, predicted_conditions, ground_truth_conditions):
        if len(predicted_conditions) != len(ground_truth_conditions):
            return False

        if set(x[0] for x in predicted_conditions) != set(x[0] for x in ground_truth_conditions):
            return False

        for predicted_col, predicted_op, predicted_val in predicted_conditions:
            ground_truth_index = tuple(x[0] for x in ground_truth_conditions).index(predicted_col)

            if ground_truth_conditions[ground_truth_index][1] != predicted_op:
                return False

            if str(ground_truth_conditions[ground_truth_index][2]).lower() != str(predicted_val).lower():
                return False

        return True

    def generate_query(self, 
                score: Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]], 
                question: List[List[str]], 
                column_tokens, 
                raw_question: List[str], 
                raw_column) -> List[Dict[str, Any]]:

        aggregate_scores, select_scores, condition_scores = score

        query_results = []
        num_batches = len(aggregate_scores)
        for batch_idx in range(num_batches):
            query_info = {}
            query_info['agg'] = np.argmax(aggregate_scores[batch_idx].cpu().detach().numpy())
            query_info['sel'] = np.argmax(select_scores[batch_idx].cpu().detach().numpy())
            query_info['conds'] = []

            condition_numbers,condition_columns,condition_operators,condition_values =\
                    [x.cpu().detach().numpy() for x in condition_scores]

            num_conditions = np.argmax(condition_numbers[batch_idx])

            all_tokens = ['<BEG>'] + question[batch_idx] + ['<END>']
            highest_indices = np.argsort(-condition_columns[batch_idx])[:num_conditions]

            for idx in range(num_conditions):
                condition = []
                condition.append(highest_indices[idx])
                condition.append(np.argmax(condition_operators[batch_idx][idx]))

                condition_value_tokens = []
                for value_score in condition_values[batch_idx][idx]:
                    value_token = np.argmax(value_score[:len(all_tokens)])
                    value_string = all_tokens[value_token]
                    if value_string == '<END>':
                        break
                    condition_value_tokens.append(value_string)

                condition.append(self.merge_tokens(condition_value_tokens, raw_question[batch_idx]))
                query_info['conds'].append(condition)

            query_results.append(query_info)

        return query_results
    
    def merge_tokens(self, token_list: List[str], raw_token_string: str) -> str:
        token_string = raw_token_string.lower()
        valid_chars = 'abcdefghijklmnopqrstuvwxyz0123456789$('
        special_chars = {'-LRB-':'(', '-RRB-':')', '-LSB-':'[', '-RSB-':']', '``':'"', '\'\'':'"', '--':u'\u2013'}
        merged_string = ''
        double_quote_flag = 0
        for raw_token in token_list:
            if not raw_token:
                continue
            token = special_chars.get(raw_token, raw_token)
            if token == '"':
                double_quote_flag = 1 - double_quote_flag
            if len(merged_string) == 0:
                pass
            elif len(merged_string) > 0 and merged_string + ' ' + token in token_string:
                merged_string = merged_string + ' '
            elif len(merged_string) > 0 and merged_string + token in token_string:
                pass
            elif token == '"':
                if double_quote_flag:
                    merged_string = merged_string + ' '
            elif token[0] not in valid_chars:
                pass
            elif (merged_string[-1] not in ['(', '/', u'\u2013', '#', '$', '&']) and (merged_string[-1] != '"' or not double_quote_flag):
                merged_string = merged_string + ' '
            merged_string = merged_string + token
        return merged_string.strip()