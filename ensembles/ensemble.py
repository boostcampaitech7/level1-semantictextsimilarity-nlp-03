import pandas as pd
import numpy as np
import glob
import torch    
import torchmetrics
from itertools import combinations

def average_last_column(input_files, output_file):
    dataframes = []

    for file in input_files:
        df = pd.read_csv(file, header=None)
        df = df.drop(index=0).reset_index(drop=True)
        dataframes.append(df)

    shapes = [df.shape for df in dataframes]
    if not all(shape == shapes[0] for shape in shapes):
        raise ValueError("All input CSV files must have the same number of rows and columns.")

    concatenated = pd.concat(dataframes, axis=0, keys=range(len(dataframes)))
    concatenated.iloc[:, -1] = concatenated.iloc[:, -1].astype(np.float64)

    grouped = concatenated.groupby(level=1)

    averaged_rows = []
    for idx, group in grouped:
        last_column_mean = group.iloc[:, -1].mean()
        averaged_rows.append(last_column_mean)

    averaged_df = pd.DataFrame(averaged_rows)

    ids = [f"boostcamp-sts-v1-test-{i:03d}" for i in range(len(averaged_rows))]
    result_df = pd.DataFrame({'id': ids, 'target': averaged_df[0]})

    result_df.to_csv(output_file, header=True, index=False)
    print(f"Averaged CSV file has been saved to {output_file}")

    return averaged_df.values
def pearson(output, target):
    with torch.no_grad():
        return torchmetrics.functional.pearson_corrcoef(output, target)
def inference(outputs, targets):

    result = {}
    result["val_pearson"] = pearson(outputs, targets)

    return result

if __name__ == "__main__":
    input_files = [
        '/data/ephemeral/home/nlp_sts/ensembles/dev_output_llama.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/dev_output_lighthouse.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/kakao_no_aug_dev_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/kr_electra_no_aug_dev_output.csv',
    ]

    def find_best_combinations(input_files, target_file, top_n=5):
        dev_df = pd.read_csv(target_file)
        targets = torch.Tensor(dev_df.iloc[:, -2].values)

        best_combinations = []
        for r in range(1, len(input_files) + 1):
            for combo in combinations(input_files, r):
                output_file = "temp_averaged_output.csv"
                averaged_df = average_last_column(combo, output_file)
                outputs = torch.Tensor(averaged_df).squeeze()
                result = inference(outputs, targets)
                val_pearson = float(result["val_pearson"])
                best_combinations.append((combo, val_pearson))

        best_combinations.sort(key=lambda x: x[1], reverse=True)
        return best_combinations[:top_n]

    target_file = '/data/ephemeral/home/nlp_sts/data/dev.csv'
    best_combinations = find_best_combinations(input_files, target_file)

    for combo, val_pearson in best_combinations:
        print(f"Combination: {combo}, val_pearson: {val_pearson}")

    # output_file = "averaged_output.csv"

    # averaged_df = average_last_column(input_files, output_file)
    # outputs = torch.Tensor(averaged_df).squeeze()

    # dev_df = pd.read_csv('/data/ephemeral/home/nlp_sts/data/dev.csv') # target file
    # targets = torch.Tensor(dev_df.iloc[:, -2].values)

    # result = inference(outputs, targets)

    # print("val_pearson: ", float(result["val_pearson"]))

