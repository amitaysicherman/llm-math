import numpy as np
import pickle
from functools import reduce


class PileNumbersDataset:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.data_file = f'{base_dir}/pile_numbers.bin'
        self.pointers_file = f'{base_dir}/pointers.npy'
        self.inverse_index_file = f'{base_dir}/inverse_mapping.pickle'
        self.pointers = None
        self.inverse_mapping = None
        self.load()

    def load(self):
        self.pointers = np.load(self.pointers_file)
        with open(self.inverse_index_file, 'rb') as handle:
            self.inverse_mapping = pickle.load(handle)

    def query(self, numbers: np.ndarray):
        relevant_indexes = reduce(np.intersect1d,
                                  ([self.inverse_mapping[x] for x in numbers]))
        relevant_pointers = self.pointers[relevant_indexes]
        relevant_lenghts = self.pointers[
                               relevant_indexes + 1] - relevant_pointers
        sentences = []
        with open(self.data_file, "rb") as pile_numbers_file:
            for pointer, length in zip(relevant_pointers, relevant_lenghts):
                pile_numbers_file.seek(pointer)
                sentence = pile_numbers_file.read(length).decode('utf-8')
                sentences.append(sentence.strip())
        return sentences


if __name__ == "__main__":
    pile_number_dataset = PileNumbersDataset('assets/db_bu')
    pile_number_dataset.load()
    sentence=pile_number_dataset.query(np.array([78, 56,34 ]))
    sentence=[x for x in sentence if "56%" in x]
    print("\n".join(sentence))
