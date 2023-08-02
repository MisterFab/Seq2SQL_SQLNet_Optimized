import torch
import torch.nn as nn
import numpy as np
from modules.word_embedding import WordEmbedding
from modules.aggregator_predict import AggregatePredictor
from modules.selection_predict import SelectPredictor
from modules.seq2sql_condition_predict import ConditionPredictor
from typing import List, Tuple, Dict, Union, Optional

# This is a re-implementation based on the following paper:

# Victor Zhong, Caiming Xiong, and Richard Socher. 2017.
# Seq2SQL: Generating Structured Queries from Natural Language using
# Reinforcement Learning. arXiv:1709.00103

class Seq2SQL(nn.Module):
    def __init__(self, word_embeddings: torch.Tensor, word_dimensions: int, hidden_size: int = 100,
                 num_layers: int = 2, use_gpu: bool = False) -> None:
        super().__init__()

        self.device = torch.device('cuda' if use_gpu else 'cpu')

        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

        self.WordEmbedding = WordEmbedding(word_embeddings, word_dimensions, use_gpu, self.SQL_TOK, False)
        self.AggregatePredictor = AggregatePredictor(word_dimensions, hidden_size, num_layers, use_gpu=use_gpu)
        self.SelectPredictor = SelectPredictor(word_dimensions, hidden_size, num_layers, use_gpu=use_gpu)
        self.ConditionPredictor = ConditionPredictor(word_dimensions, hidden_size, num_layers, 200, use_gpu=use_gpu)

        self.crossentropyloss = nn.CrossEntropyLoss()

    def generate_gt_where_seq(self, queries: List[str], columns: List[List[str]], 
                              full_queries: List[str]) -> List[List[int]]:
        gt_where_sequences = []

        for query_tokens, column_tokens, full_query in zip(queries, columns, full_queries):
            connected_columns = [token for column in column_tokens for token in column + [',']]
            all_tokens = self.SQL_TOK + connected_columns + [None] + query_tokens + [None]

            sequence = [all_tokens.index('<BEG>')]

            if 'WHERE' in full_query:
                where_clause = full_query[full_query.index('WHERE'):]
                sequence += [all_tokens.index(token) if token in all_tokens else 0 for token in where_clause]

            sequence.append(all_tokens.index('<END>'))
            gt_where_sequences.append(sequence)

        return gt_where_sequences

    def forward(self, questions: List[List[str]], columns: List[List[List[str]]], column_numbers: List[int],
                ground_truth_where: Optional[List[List[str]]] = None, ground_truth_condition = None,
                ground_truth_selection = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        question_embedding, question_lengths = self.WordEmbedding.generate_batch(questions, columns)
        column_input_var, column_name_lengths, column_lengths = self.WordEmbedding.generate_column_batch(columns)

        aggregate_scores = self.AggregatePredictor(question_embedding, question_lengths, column_input_var, 
                                                   column_name_lengths, column_lengths, ground_truth_selection)
        selection_scores = self.SelectPredictor(question_embedding, question_lengths, column_input_var, 
                                                         column_name_lengths, column_lengths, column_numbers)
        condition_scores = self.ConditionPredictor(question_embedding, question_lengths, ground_truth_where)

        return aggregate_scores, selection_scores, condition_scores

    def loss(self, score: torch.Tensor, answer_sequence: List[List[int]],
             ground_truth_where: List[List[Union[int, str]]]) -> torch.Tensor:
        
        aggregation_score, selection_score, condition_score = score
        total_loss = 0

        aggregation_truth = torch.tensor([x[0] for x in answer_sequence], device=self.device)
        total_loss += self.crossentropyloss(aggregation_score, aggregation_truth)

        selection_truth = torch.tensor([x[1] for x in answer_sequence], device=self.device)
        total_loss += self.crossentropyloss(selection_score, selection_truth)

        for i in range(len(ground_truth_where)):
            cond_truth = torch.tensor(ground_truth_where[i][1:], device=self.device)
            cond_pred_score = condition_score[i, :len(ground_truth_where[i])-1]
            total_loss += self.crossentropyloss(cond_pred_score, cond_truth) / len(ground_truth_where)

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

    def generate_query(self, scores: Tuple[np.ndarray], questions: List[List[str]], columns: List[List[str]],
                       raw_questions: List[str], raw_columns: List[str]) -> List[Dict]:

        aggregate_scores, selection_scores, condition_scores = scores

        queries = []
        for batch in range(len(aggregate_scores)):
            query = {
                'agg': np.argmax(aggregate_scores[batch].data.cpu().numpy()),
                'sel': np.argmax(selection_scores[batch].data.cpu().numpy()),
                'conds': []
            }

            all_tokens = self.SQL_TOK + [token for tokens in columns[batch] for token in tokens + [',']] + [''] + questions[batch] + ['']
            
            condition_tokens = []
            for score in condition_scores[batch].data.cpu().numpy():
                token = all_tokens[np.argmax(score)]
                if token == '<END>':
                    break
                condition_tokens.append(token)

            if condition_tokens:
                condition_tokens = condition_tokens[1:]

            condition_start = 0
            while condition_start < len(condition_tokens):
                condition = [None, None, None]
                condition_end = condition_tokens[condition_start:].index('AND') + condition_start if 'AND' in condition_tokens[condition_start:] else len(condition_tokens)

                operators = {'EQL': 0, 'GT': 1, 'LT': 2}

                for op, symbol in operators.items():
                    if op in condition_tokens[condition_start:condition_end]:
                        operator_idx = condition_tokens[condition_start:condition_end].index(op) + condition_start
                        condition[1] = symbol
                        break
                else:
                    operator_idx = condition_start
                    condition[1] = operators['EQL']

                selected_column = condition_tokens[condition_start:operator_idx]
                columns_lower = [x.lower() for x in raw_columns[batch]]
                predicted_column = self.merge_tokens(selected_column, raw_questions[batch] + ' || ' + ' || '.join(raw_columns[batch]))
                condition[0] = columns_lower.index(predicted_column) if predicted_column in columns_lower else 0
                condition[2] = self.merge_tokens(condition_tokens[operator_idx+1:condition_end], raw_questions[batch])
                
                query['conds'].append(condition)
                condition_start = condition_end + 1

            queries.append(query)

        return queries

    def merge_tokens(self, token_list, raw_token_string):
        lowercase_token_string = raw_token_string.lower()
        valid_characters = 'abcdefghijklmnopqrstuvwxyz0123456789$('
        special_characters_map = {'-LRB-': '(', '-RRB-': ')', '-LSB-': '[', '-RSB-': ']', '``': '"', '\'\'': '"', '--': u'\u2013'}
        merged_string = ''
        double_quote_flag = False
        for raw_token in token_list:
            if not raw_token:
                continue
            token = special_characters_map.get(raw_token, raw_token)
            if token == '"':
                double_quote_flag = not double_quote_flag
            if merged_string and merged_string + ' ' + token in lowercase_token_string:
                merged_string += ' '
            elif merged_string and merged_string + token in lowercase_token_string:
                continue
            elif token == '"':
                if double_quote_flag:
                    merged_string += ' '
            elif token[0] not in valid_characters:
                continue
            elif merged_string and merged_string[-1] not in ['(', '/', u'\u2013', '#', '$', '&'] and (merged_string[-1] != '"' or not double_quote_flag):
                merged_string += ' '
            merged_string += token
        return merged_string.strip()