import pandas as pd


def get_most_common_results(data):
    return data['z'].value_counts().index[0]


def get_most_common_wrong(data):
    x, y = data['x'].values[0], data['y'].values[0]
    z = x + y
    most_common_sorted = list(data['z'].value_counts().index)
    if most_common_sorted[0] == z:
        most_common_sorted.pop(0)

    if not most_common_sorted:
        return None

    return most_common_sorted[0]


def get_most_common_wrong_count(data):
    most_common = get_most_common_wrong(data)
    if most_common is None:
        return None
    return data['z'].value_counts().get(most_common, 0)


def get_correct_count(data):
    x, y = data['x'].values[0], data['y'].values[0]
    z = x + y
    return data['z'].value_counts().get(z, 0)


def get_most_common_count(data):
    return data['z'].value_counts().iloc[0]


def get_all_counts(data):
    return str(data['z'].value_counts().to_dict())


def get_full_string(data):
    results = []
    for _, row in data.iterrows():
        x, y, z, a, b = row['x'], row['y'], row['z'], row['a'], row['b']
        results.append(
            f"Q:{a}+{b}=<br/>A:{a + b}<br/>Q:{x}+{y}=<br/>A:[{z}]<br/>")
    return "<br/>".join(results)


class ResultsAnalyzer:
    def __init__(self):
        self.data = pd.read_csv('assets/gen_results.txt',
                                names=['x', 'y', 'z', 'a', 'b'])
        self.data.dropna(inplace=True)
        self.data = self.data.astype(int)
        self.groups = self.data.groupby(['x', 'y'])
        self.most_common_results = self.groups.apply(get_most_common_results)
        self.most_common_results_mesh = self.to_mesh(self.most_common_results)

        self.most_common_wrong = self.groups.apply(get_most_common_wrong)
        self.most_common_wrong_mesh = self.to_mesh(self.most_common_wrong)

        self.most_common_wrong_count = self.groups.apply(
            get_most_common_wrong_count)
        self.most_common_wrong_count_mesh = self.to_mesh(
            self.most_common_wrong_count)

        self.correct_count = self.groups.apply(get_correct_count)
        self.correct_count_mesh = self.to_mesh(self.correct_count)

        self.most_common_count = self.groups.apply(get_most_common_count)
        self.most_common_count_mesh = self.to_mesh(self.most_common_count)

        self.all_counts = self.groups.apply(get_all_counts)
        self.all_counts_mesh = self.to_mesh(self.all_counts)

        self.full_string = self.groups.apply(get_full_string)
        self.full_string_mesh = self.to_mesh(self.full_string)

    def to_mesh(self, data):
        data = data.to_frame(name="z").reset_index()
        data['x'] = data['x'].astype(int)
        data['y'] = data['y'].astype(int)
        data['real_z'] = data['x'] + data['y']
        data.loc[data['x'] < 10, 'z'] = None
        data.loc[data['y'] < 10, 'z'] = None
        data.loc[data['x'] > 99, 'z'] = None
        data.loc[data['y'] > 99, 'z'] = None
        data.loc[data['real_z'] > 100, 'z'] = None
        data_pivot = data.pivot_table(index='x', columns='y', values='z',
                                      aggfunc='first')
        data_pivot.index = data_pivot.index.astype(int)
        data_pivot.columns = data_pivot.columns.astype(int)

        # data_pivot.drop(columns=list(range(10)), inplace=True)
        # data_pivot.drop(index=list(range(10)), inplace=True)
        return data_pivot


if __name__ == "__main__":
    import time

    stsrt = time.time()
    results_analyzer = ResultsAnalyzer()
    print(time.time() - stsrt)
    print(results_analyzer.most_common_results)
