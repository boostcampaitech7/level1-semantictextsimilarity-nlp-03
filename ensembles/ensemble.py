import pandas as pd
import numpy as np
import torch    
import torchmetrics
from itertools import combinations
from sklearn.linear_model import LinearRegression

class Ensemble:
    def __init__(self, input_files, target_file):
        self.input_files = input_files
        self.target_file = target_file

    def data_loader(self, input_files):
        dataframes = []
        for file in input_files:
            df = pd.read_csv(file, header=0)
            dataframes.append(df["target"])
        dev_df = pd.read_csv(self.target_file)
        dataframes.append(dev_df["label"])
        features_df = pd.concat(dataframes, axis=1)
        return features_df

    def average_label(self, input_files, output_file, weights=None):
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

        for _, group in grouped:
            if weights is not None:
                values = group.iloc[:, -1].tolist()
                last_column_mean = self.weighted_mean(values, weights)
            else:
                last_column_mean = group.iloc[:, -1].mean()
            averaged_rows.append(last_column_mean)

        averaged_df = pd.DataFrame(averaged_rows)

        return averaged_df

    def weighted_mean(self, values, weights):
        values = np.array(values)
        weights = np.array(weights)

        if len(values) != len(weights):
            raise ValueError("The number of values must match the number of weights.")

        # Weighted mean calculation
        weighted_sum = np.sum(values * weights)
        weights_sum = np.sum(weights)

        return weighted_sum / weights_sum

    def pearson(self, output, target):
        with torch.no_grad():
            return torchmetrics.functional.pearson_corrcoef(output, target)
        
    def inference(self, outputs, targets):
        result = self.pearson(outputs, targets)
        return float(result)

    def find_best_combinations_rand(self, top_n=5, num_weights_search=None):
        dev_df = pd.read_csv(self.target_file)
        targets = torch.Tensor(dev_df.iloc[:, -2].values)

        best_combinations = []
        output_file = "./output/temp_averaged_output.csv"
        if num_weights_search is not None:
            for r in range(1, len(self.input_files) + 1):
                for combo in combinations(self.input_files, r):
                    for _ in range(num_weights_search):
                        weights = np.random.dirichlet(np.ones(len(combo)), size=1)[0]
                        averaged_df = self.average_label(combo, output_file, weights=weights)
                        outputs = torch.Tensor(averaged_df.values).squeeze()
                        result = self.inference(outputs, targets)
                        val_pearson = result
                        best_combinations.append((combo, val_pearson, weights))
            best_combinations.sort(key=lambda x: x[1], reverse=True)
            for combo, val_pearson, weights in best_combinations[:top_n]:
                print(f"Combination: {combo}, val_pearson: {val_pearson}, weights: {weights}")
        else:
            for r in range(1, len(self.input_files) + 1):
                for combo in combinations(self.input_files, r):
                    averaged_df = self.average_label(combo, output_file)
                    outputs = torch.Tensor(averaged_df.values).squeeze()
                    result = self.inference(outputs, targets)
                    val_pearson = result
                    best_combinations.append((combo, val_pearson))
            best_combinations.sort(key=lambda x: x[1], reverse=True)
            for combo, val_pearson in best_combinations[:top_n]:
                print(f"Combination: {combo}, val_pearson: {val_pearson}")
        print(f"num of combinations: ", len(best_combinations))
        return best_combinations
    
    def find_best_combinations(self, top_n=5):

        best_combinations = []
        output_file = "./output/temp_averaged_output.csv"

        for r in range(1, len(self.input_files) + 1):
            for combo in combinations(self.input_files, r):
                featured_df = self.data_loader(combo)
                targets = torch.Tensor(featured_df['label'].values)

                X = featured_df.iloc[:, :-1]
                y = featured_df['label']
                model = LinearRegression(fit_intercept=False)  # intercept를 0으로 고정
                model.fit(X, y)
                weights = model.coef_.tolist()
                # print("coefficients: ", weights)
                
                averaged_df = self.average_label(combo, output_file, weights=weights)
                outputs = torch.Tensor(averaged_df.values).squeeze()
                result = self.inference(outputs, targets)
                val_pearson = result
                best_combinations.append((combo, val_pearson, weights))

        best_combinations.sort(key=lambda x: x[1], reverse=True)
        for combo, val_pearson, weights in best_combinations[:top_n]:
            print(f"Combination: {combo}, val_pearson: {val_pearson}, weights: {weights}")
    
    def run(self, weights=None):
        output_file = "./output/averaged_output.csv"
        averaged_df = ensemble.average_label(input_files, output_file, weights)
        outputs = torch.Tensor(averaged_df.values).squeeze()

        target_df = pd.read_csv(self.target_file) 
        targets = torch.Tensor(target_df.iloc[:, -2].values)

        result = self.inference(outputs, targets)
        print("val_pearson: ", result)
        return result
    
    def extract(self, weights=None):
        output_file = "./output/averaged_output.csv"
        averaged_df = ensemble.average_label(input_files, output_file, weights)

        ids = [f"boostcamp-sts-v1-test-{i:03d}" for i in range(len(averaged_df))]
        result_df = pd.DataFrame({'id': ids, 'target': averaged_df[0]})
        result_df.to_csv(output_file, header=True, index=False)
        print(f"Averaged CSV file has been saved to {output_file}")

if __name__ == "__main__":
    target_file = '/data/ephemeral/home/nlp_sts/data/dev.csv' # target file
    input_files = [
        './model_csv/dev_output_llama.csv',
        # './model_csv/dev_output_lighthouse.csv',
        './model_csv/kakao_no_aug_dev_output.csv',
        './model_csv/kr_electra_no_aug_dev_output.csv',
    ]
   
    ensemble = Ensemble(input_files, target_file)

    """
        ensemble.run(weights=[0.7, 0.1, 0.1, 0.1]) 
        각각의 비율로 평균을 내고, 평가 지표를 계산한다.
        weights가 비어있는 경우, 각 모델의 결과를 단순 평균내어 평가 지표를 계산한다.

        ensemble.find_best_combinations_rand(num_weights_search=20) 
        모든 조합을 만들어, 랜덤한 가중치를 부여하여 평가 지표를 계산한다.
        num_weights_search는 랜덤 가중치를 부여하는 횟수를 의미한다.
        가장 높은 평가 지표를 가진 조합을 출력한다.

        ensemble.find_best_combinations()
        모든 조합을 만들어, Linear Regression을 사용하여 가중치를 부여하여 평가 지표를 계산한다.
        가장 높은 평가 지표를 가진 조합을 출력한다.

        ensemble.extract(weights=[0.7, 0.1, 0.1, 0.1]) 
        각각의 비율로 평균을 내고, 평가 지표를 계산한다.
        weights가 비어있는 경우, 각 모델의 결과를 단순 평균내어 평가 지표를 계산한다.
        평균 결과를 CSV 파일로 저장한다.
    """

    ensemble.find_best_combinations()
    # ensemble.run(weights=[0.3, 0.3, 0.5]) 
    # ensemble.find_best_combinations(num_weights_search=20)
    # ensemble.extract(weights=[0.7, 0.1, 0.1, 0.1])
