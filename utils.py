import json
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple

def load_data(sql_path: str, table_path: str) -> Tuple[List[Dict], Dict[str, Dict]]:
    with open(sql_path) as file:
        sql_data = [json.loads(line.strip()) for line in file]

    with open(table_path) as file:
        table_data = {entry['id']: entry for entry in (json.loads(line.strip()) for line in file)}

    return sql_data, table_data

def load_dataset(dataset_id: int) -> Tuple[List[Dict], Dict[str, Dict]]:
    if dataset_id == 0:
        print("Loading from original dataset")
        sql_data, table_data = load_data('data/train_tok.jsonl',
                'data/train_tok.tables.jsonl')
        val_sql_data, val_table_data = load_data('data/dev_tok.jsonl',
                'data/dev_tok.tables.jsonl')
        test_sql_data, test_table_data = load_data('data/test_tok.jsonl',
                'data/test_tok.tables.jsonl')
    elif dataset_id == 1:
        print("Loading from re-split dataset")
        sql_data, table_data = load_data('data_resplit/train.jsonl',
                'data_resplit/tables.jsonl')
        val_sql_data, val_table_data = load_data('data_resplit/dev.jsonl',
                'data_resplit/tables.jsonl')
        test_sql_data, test_table_data = load_data('data_resplit/test.jsonl',
                'data_resplit/tables.jsonl')
    elif dataset_id == 2:
        print("Loading test data from altered questions dataset 1")
        test_sql_data, test_table_data = load_data('data/altered_questions_1.jsonl',
                'data/test_tok.tables.jsonl')
        return None, None, None, None, test_sql_data, test_table_data
    elif dataset_id == 3:
        print("Loading test data from altered questions dataset 2")
        test_sql_data, test_table_data = load_data('data/altered_questions_2.jsonl',
                'data/test_tok.tables.jsonl')
        return None, None, None, None, test_sql_data, test_table_data
    elif dataset_id == 4:
        print("Loading test data from altered questions dataset 3")
        test_sql_data, test_table_data = load_data('data/altered_questions_3.jsonl',
                'data/test_tok.tables.jsonl')
        return None, None, None, None, test_sql_data, test_table_data
        
    return sql_data, table_data, val_sql_data, val_table_data, test_sql_data, test_table_data

def load_word_embeddings(filepath: str, load_used: bool) -> Dict[str, np.ndarray]:
    if not load_used:
        print("Loading word embedding")
        with open(filepath, encoding='utf-8') as file:
            word_vectors = {line.split()[0].lower(): np.array([float(x) for x in line.split()[1:]]) 
                            for line in file}
        return word_vectors
    else:
        print('Loading used word embedding')
        with open('glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('glove/usedwordemb.npy', 'rb') as inf:
            word_emb_val = np.load(inf, allow_pickle=True)
        return w2i, word_emb_val
    
def create_batch_sequence(sql_data: List[Dict], 
    table_data: Dict[str, Dict],
    indexes: list,
    start_index: int,
    end_index: int, 
    return_visualization_data: bool = False):

    question_sequences = []
    column_sequences = []
    column_counts = []
    answer_sequences = []
    query_sequences = []
    ground_truth_conditions = []
    visualization_sequences = []

    for i in range(start_index, end_index):
        sql_entry = sql_data[indexes[i]]
        table_id = sql_entry['table_id']
        table_entry = table_data[table_id]

        question_tokens = sql_entry['question_tok']
        header_token = table_entry['header_tok']
        header = table_entry['header']
        sql_conditions = sql_entry['sql']['conds']

        agg = sql_entry['sql']['agg']
        sel = sql_entry['sql']['sel']
        query_tokens = sql_entry['query_tok']

        question_sequences.append(question_tokens)
        column_sequences.append(header_token)
        column_counts.append(len(header))
        
        condition_columns = tuple(x[0] for x in sql_conditions)
        condition_operations = tuple(x[1] for x in sql_conditions)
        
        answer_sequences.append((agg, sel, len(sql_conditions), condition_columns, condition_operations))
        query_sequences.append(query_tokens)
        ground_truth_conditions.append(sql_conditions)

        if return_visualization_data:
            question = sql_entry['question']
            sql_query = sql_entry['query']
            visualization_sequences.append((question, header, sql_query))
    
    results = (question_sequences, column_sequences, column_counts, answer_sequences, query_sequences, ground_truth_conditions)
    if return_visualization_data:
        results += (visualization_sequences,)

    return results

def create_batch_query(sql_data: List[Dict], indexes: list, start_index: int, end_index: int
                       ) -> Tuple[List[Dict], List[str]]:

    query_ground_truths = [sql_data[indexes[i]]['sql'] for i in range(start_index, end_index)]
    table_ids = [sql_data[indexes[i]]['table_id'] for i in range(start_index, end_index)]

    return query_ground_truths, table_ids

def epoch_accuracy(model: nn.Module, batch_size: int, sql_data: np.ndarray, table_data: np.ndarray) -> Tuple[float, float]:
    model.eval()

    total_samples = len(sql_data)
    indexes = list(range(len(sql_data)))
    total_correct = 0.0
    one_correct = 0.0

    for start_index in range(0, total_samples, batch_size):
        end_index = min(start_index + batch_size, total_samples)

        batch = create_batch_sequence(sql_data, table_data, indexes, start_index, end_index, return_visualization_data=True)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = batch

        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        
        query_gt, _ = create_batch_query(sql_data, indexes, start_index, end_index)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model(q_seq, col_seq, col_num, ground_truth_selection = gt_sel_seq)
        pred_queries = model.generate_query(score, q_seq, col_seq, raw_q_seq, raw_col_seq)
        one_err, total_err = model.check_accuracy(pred_queries, query_gt)

        total_correct += (end_index - start_index - total_err)
        one_correct += (end_index - start_index - one_err)

    total_accuracy = total_correct / total_samples
    one_accuracy = one_correct / total_samples

    return total_accuracy, one_accuracy

def epoch_train(
    model: nn.Module, 
    optimizer: optim.Optimizer, 
    batch_size: int, 
    sql_data: np.ndarray, 
    table_data: np.ndarray
) -> float:
    model.train()
    indexes = np.random.permutation(len(sql_data))
    total_samples = len(sql_data)
    cumulative_loss = 0.0

    for start_index in range(0, total_samples, batch_size):
        end_index = min(start_index + batch_size, len(indexes))

        batch = create_batch_sequence(sql_data, table_data, indexes, start_index, end_index)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = batch
        
        gt_sel_seq = [x[1] for x in ans_seq]
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        score = model(q_seq, col_seq, col_num, gt_where_seq, gt_cond_seq, gt_sel_seq)

        loss = model.loss(score, ans_seq, gt_where_seq)
        cumulative_loss += loss.item() * (end_index - start_index)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return cumulative_loss / len(sql_data)