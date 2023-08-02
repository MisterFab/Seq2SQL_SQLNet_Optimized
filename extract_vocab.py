import json
import numpy as np
from typing import Dict, List, Tuple, Union
from utils import load_dataset, load_word_embeddings

NUM_VECTORS = 300
WORD_EMBEDDING_FILE = 'glove/glove.42B.300d.txt'
WORD2IDX_FILE = 'glove/word2idx.json'
USED_WORD_EMB_FILE = 'glove/usedwordemb.npy'

def build_vocabulary_and_embeddings(datasets: List[Union[Dict, List]], 
                                    word_embeddings: Dict[str, np.ndarray]) -> Tuple[Dict[str, int], List[np.ndarray]]:
    word_to_index = {'<UNK>': 0, '<BEG>': 1, '<END>': 2}
    used_embeddings = [np.zeros(NUM_VECTORS) for _ in range(len(word_to_index))]

    def check_and_add_token(token: str):
        if token not in word_to_index and token in word_embeddings:
            word_to_index[token] = len(word_to_index)
            used_embeddings.append(word_embeddings[token])

    for dataset in datasets:
        if isinstance(dataset, dict):
            for data in dataset.values():
                for col in data['header_tok']:
                    for token in col:
                        check_and_add_token(token)
        else:
            for entry in dataset:
                for token in entry['question_tok']:
                    check_and_add_token(token)
    
    return word_to_index, used_embeddings

def main():
    datasets = load_dataset(0)
    word_embeddings = load_word_embeddings(WORD_EMBEDDING_FILE)
    word_to_index, used_embeddings = build_vocabulary_and_embeddings(datasets, word_embeddings)
    
    emb_array = np.stack(used_embeddings, axis=0)
    
    with open(WORD2IDX_FILE, 'w') as file:
        json.dump(word_to_index, file)
        
    with open(USED_WORD_EMB_FILE, 'wb') as file:
        np.save(file, emb_array)

if __name__ == '__main__':
    main()
