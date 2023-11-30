from tqdm import tqdm
import jsonlines
import nltk
from npy_append_array import NpyAppendArray
from multiprocessing import Pool
import os
import numpy as np
import pickle

from nltk.tokenize import sent_tokenize
from utils import find_numbers_in_text, MAX_N, MIN_N
import pandas as pd

nltk.download('punkt')


def get_triples_from_results_file(file_path='assets/gen_results.txt'):
    with open(file_path) as f:
        results = [sorted([int(y) for y in x.split(",") if y]) for x in
                   f.read().split('\n')]
    results_df = pd.DataFrame(results[:-1], columns=['x', 'y', 'z', 'a', 'c'])
    triples = set(
        results_df[['x', 'y', 'z']].itertuples(index=False, name=None))
    return triples


def build_full_num_arrays(num_arrays, num_array_count):
    full_num_arrays = np.zeros((len(num_array_count), MAX_N), dtype=np.bool_)
    x = 0
    for i, count_ in tqdm(enumerate(num_array_count),
                          total=len(num_array_count)):
        full_num_arrays[i, num_arrays[x:x + count_]] = True
        x += count_
    return full_num_arrays


def build_inverse_mapping(full_num_arrays):
    inverse_mapping = {}
    for i in tqdm(range(MAX_N)):
        inverse_mapping[i] = np.where(full_num_arrays[:, i])[0].astype(
            np.uint32)
    return inverse_mapping


def save_sentence(pile_numbers_file, single_line):
    sentence_bytes = single_line.encode('utf-8')
    pile_numbers_file.write(sentence_bytes)
    return sentence_bytes


class PileNumbersDataset:
    def __init__(self, base_dir, remove_existing=True):
        self.base_dir = base_dir
        self.remove_existing = remove_existing
        self.data_file = f'{base_dir}/pile_numbers.bin'
        self.pointers_file = f'{base_dir}/pointers.npy'
        self.num_arrays_file = f'{base_dir}/num_arrays.npy'
        self.num_array_count_file = f'{base_dir}/num_array_count.npy'
        self.inverse_index_file = f'{base_dir}/inverse_mapping.pickle'
        self.triples = get_triples_from_results_file()
        self.pointers = None
        self.inverse_mapping = None

    def build_inverse_index(self):
        num_arrays, num_array_count = self.load_num_arrays_and_counts()
        full_num_arrays = build_full_num_arrays(num_arrays, num_array_count)
        inverse_mapping = build_inverse_mapping(full_num_arrays)
        self.save_inverse_mapping(inverse_mapping)

    def load_num_arrays_and_counts(self):
        num_arrays = np.load(self.num_arrays_file)
        num_array_count = np.load(self.num_array_count_file)
        return num_arrays, num_array_count

    def save_inverse_mapping(self, inverse_mapping):
        with open(self.inverse_index_file, 'wb') as handle:
            pickle.dump(inverse_mapping, handle,
                        protocol=pickle.HIGHEST_PROTOCOL)

    def check_skipped(self, line_numbers):
        if any([(x != int(x)) or (x >= MAX_N) or (x < MIN_N) for x in
                line_numbers]):
            return True
        line_numbers = [int(x) for x in line_numbers]
        sorted__numbers = tuple(sorted(line_numbers))
        if len(set(line_numbers)) != 3 or sorted__numbers not in self.triples:
            return True
        return False

    def remove_or_create_existing_files(self):
        if self.remove_existing:
            for file_name in os.listdir(self.base_dir):
                full_path = os.path.join(self.base_dir, file_name)
                if os.path.isdir(full_path):
                    for file_name_ in os.listdir(full_path):
                        os.remove(os.path.join(full_path, file_name_))
                else:
                    os.remove(full_path)
            else:
                os.makedirs(self.base_dir, exist_ok=True)

    def save(self, input_file):
        self.remove_or_create_existing_files()
        with jsonlines.open(input_file) as reader, open(self.data_file,
                                                        "wb") as pile_numbers_file:
            a, b = 0, 0
            pbar = tqdm(reader)
            for i, line in enumerate(pbar):
                for single_line_ in sent_tokenize(line.get('text', '').lower()):
                    for single_line in single_line_.split('\n'):
                        b += 1
                        line_numbers = find_numbers_in_text(single_line)
                        if self.check_skipped(line_numbers):
                            continue

                        self.save_line_numbers(line_numbers)
                        sentence_bytes = save_sentence(pile_numbers_file,
                                                       single_line)
                        a += 1
                        self.save_pointer(pile_numbers_file, sentence_bytes)

                        if b % 1_000 == 0:
                            msg = f"{self.base_dir}[{a:,}/{b:,} ({a / b :.2%})])"
                            pbar.set_description(msg)
                if a > 100:  # TODO remove this
                    break

            self.save_final_pointer(pile_numbers_file)

    def save_line_numbers(self, line_numbers):
        with NpyAppendArray(self.num_arrays_file) as npaa:
            line_numbers = sorted(list(set(line_numbers)))
            npaa.append(np.array(line_numbers, dtype=np.uint16))

        with NpyAppendArray(self.num_array_count_file) as npaa:
            npaa.append(np.array([len(line_numbers)], dtype=np.uint16))

    def save_pointer(self, pile_numbers_file, sentence_bytes):
        with NpyAppendArray(self.pointers_file) as npaa:
            npaa.append(
                np.array([pile_numbers_file.tell() - len(sentence_bytes)],
                         dtype=np.uint64))

    def save_final_pointer(self, pile_numbers_file):
        with NpyAppendArray(self.pointers_file) as npaa:
            npaa.append(np.array([pile_numbers_file.tell()], dtype=np.uint64))

    def load(self):
        self.pointers = np.load(self.pointers_file)
        with open(self.inverse_index_file, 'rb') as handle:
            self.inverse_mapping = pickle.load(handle)


def make_dataset(split_num):
    os.makedirs(f'DB{split_num}', exist_ok=True)
    pile_number_dataset = PileNumbersDataset(f'DB{split_num}')
    pile_number_dataset.save(f'{PILE_BASE_DIR}/{split_num}.jsonl')
    return pile_number_dataset


def combine_datasets(output_dir, part_numbers):
    os.makedirs(output_dir, exist_ok=True)
    output_data_bin = f'{output_dir}/pile_numbers.bin'
    if os.path.exists(output_data_bin):
        os.remove(output_data_bin)

    output_pointers_file = f'{output_dir}/pointers.npy'
    output_num_arrays_file = f'{output_dir}/inverse_mapping.pickle'

    all_pointers, prev_last, prev_pointers_index, inverse_mapping = [], 0, 0, None

    for i in part_numbers:
        pointers = np.load(f'DB{i}/pointers.npy')
        all_pointers.append(pointers[:-1] + prev_last)
        prev_last += pointers[-1]

        data_bin = f'DB{i}/pile_numbers.bin'
        with open(data_bin, 'rb') as f:
            with open(output_data_bin, 'ab') as g:
                g.write(f.read())

        with open(f'DB{i}/inverse_mapping.pickle', 'rb') as handle:
            new_inverse_mapping = pickle.load(handle)

        prev_pointers_index, inverse_mapping = update_inverse_mapping(
            prev_pointers_index,
            inverse_mapping,
            new_inverse_mapping,
            pointers)

    save_final_inverse_mapping(output_num_arrays_file, inverse_mapping)

    all_pointers = concatenate_pointers(all_pointers, prev_last)
    np.save(output_pointers_file, all_pointers)


def update_inverse_mapping(prev_pointers_index, inverse_mapping,
    new_inverse_mapping, pointers):
    for key in new_inverse_mapping:
        new_inverse_mapping[key] += prev_pointers_index
        new_inverse_mapping[key] = new_inverse_mapping[key].astype(np.uint32)

    prev_pointers_index += len(pointers) - 1

    if inverse_mapping is None:
        inverse_mapping = new_inverse_mapping
    else:
        update_existing_inverse_mapping(inverse_mapping, new_inverse_mapping)

    return prev_pointers_index, inverse_mapping


def update_existing_inverse_mapping(inverse_mapping, new_inverse_mapping):
    for key in new_inverse_mapping:
        if key in inverse_mapping:
            inverse_mapping[key] = np.concatenate(
                [inverse_mapping[key], new_inverse_mapping[key]])
        else:
            inverse_mapping[key] = new_inverse_mapping[key]


def save_final_inverse_mapping(output_num_arrays_file, inverse_mapping):
    for key in inverse_mapping:
        print(key, inverse_mapping[key].shape)
    with open(output_num_arrays_file, 'wb') as handle:
        pickle.dump(inverse_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)


def concatenate_pointers(all_pointers, prev_last):
    all_pointers.append(np.array([prev_last]))
    all_pointers = np.concatenate(all_pointers).astype(np.uint64)
    print(all_pointers.shape)
    return all_pointers


if __name__ == "__main__":
    PILE_BASE_DIR = "../train"
    part_numbers = [f'{x:02}' for x in range(30)]

    with Pool() as pool:
        datasets = pool.map(make_dataset, part_numbers)

    for dataset in datasets:
        dataset.build_inverse_index()

    output_dir = "assets/db"
    combine_datasets(output_dir, part_numbers)

    for dataset in datasets:
        dataset.remove_existing = True
        dataset.remove_or_create_existing_files()
        os.rmdir(dataset.base_dir)
