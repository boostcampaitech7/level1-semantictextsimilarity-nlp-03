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
        features_df = pd.concat(dataframes, axis=1)
        target_df = pd.read_csv(self.target_file)
        return features_df, target_df["label"]

    def average_label(self, input_files, weights=None):

        data, label = self.data_loader(input_files)
        
        if weights is None:
            df_mean = data.mean(axis=1)
        else:
            df_mean = data.apply(lambda row: self.weighted_mean(row, weights=weights), axis=1)

        return df_mean

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
        outputs = torch.Tensor(outputs.values).squeeze()
        targets = torch.Tensor(targets.values)
        result = self.pearson(outputs, targets)
        return float(result)

    def find_best_combinations(self, top_n=5):

        best_combinations = []

        for r in range(1, len(self.input_files) + 1):
            for combo in combinations(self.input_files, r):
                data, label = self.data_loader(combo)

                model = LinearRegression(fit_intercept=False)  # intercept를 0으로 고정
                model.fit(data, label)
                weights = model.coef_.tolist()
                
                outputs = self.average_label(combo, weights=weights)
                val_pearson = self.inference(outputs, label)
                best_combinations.append((combo, val_pearson, weights))

        best_combinations.sort(key=lambda x: x[1], reverse=True)
        for combo, val_pearson, weights in best_combinations[:top_n]:
            print(f"Combination: {combo}, val_pearson: {val_pearson}, weights: {weights}")
            print()
        print(f"num of combinations: ", len(best_combinations))
    
    def run(self, weights=None):
        outputs = ensemble.average_label(self.input_files, weights)

        _, label = self.data_loader(self.input_files)

        result = self.inference(outputs, label)
        print("val_pearson: ", result)
        return result
    
    def extract(self, weights=None):
        output_file = "./output_csv/averaged_output.csv"
        averaged_df = ensemble.average_label(self.input_files, weights)
        print("len: ",len(averaged_df))

        ids = [f"boostcamp-sts-v1-test-{i:03d}" for i in range(len(averaged_df))]
        result_df = pd.DataFrame({'id': ids, 'target': averaged_df.values})
        result_df.to_csv(output_file, header=True, index=False)
        print(f"Averaged CSV file has been saved to {output_file}")

if __name__ == "__main__":
    target_file = '/data/ephemeral/home/nlp_sts/data/dev.csv' # target file
    input_files = [
        './model_csv/dev_output_llama.csv',
        './model_csv/dev_output_lighthouse.csv',
        './model_csv/kakao_no_aug_dev_output.csv',
        './model_csv/kr_electra_no_aug_dev_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/STSModel_kakaobank-kf-deberta-base_val_pearson=0.9138057231903076/dev_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/preprocess_l2_electra/dev_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/preprocess_l1_electra_val_p=0.9263.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/WithDropout_kakaobank-kf-deberta-base_val_pearson=0.9299044609069824-20240926T015555Z-001.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/원래데이터_monologg-koelectra-base-v3-discriminator_0.9288387298583984.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/llama_dev_output=0.93.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/원래데이터_jhgan-ko-sroberta-multitask_0.9175683259963989.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/only_preprocessin_kakao/dev_output.csv',
    ]
    test_input_files = [
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/라마 0.92/test_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/lighthouse/test_output_lighthouse.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/카뱅 최고성능/Dberta_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/electra 최고성능/electra_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/STSModel_kakaobank-kf-deberta-base_val_pearson=0.9138057231903076/test_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/preprocess_l2_electra/test_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/preprocess_l1_electra_val_p=0.9263/test_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/kakao_0자른데이터_0.929_output/WithDropout_kakaobank-kf-deberta-base_val_pearson=0.9299044609069824/test_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/원래데이터_monologg-koelectra-base-v3-discriminator_0.9288387298583984/test_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/SLMModel_beomi-Llama-3-Open-Ko-8B_val_pearson=0.9314584732055664/test_output.csv',
        '/data/ephemeral/home/nlp_sts/ensembles/model_csv/csvs/원래데이터_jhgan-ko-sroberta-multitask_0.9175683259963989/test_output.csv',
    ]
   
    ensemble = Ensemble(input_files, target_file)

    """
        ensemble.run(weights=[0.7, 0.1, 0.1, 0.1]) 
        각각의 비율로 평균을 내고, 평가 지표를 계산한다.
        weights가 비어있는 경우, 각 모델의 결과를 단순 평균내어 평가 지표를 계산한다.

        ensemble.find_best_combinations_rand(num_weights_search=20) -> Not Available now
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
    # ensemble.run([0.15985997782000064, -0.02712903419977486, 0.13194138929042556, 0.35350721304784233, 0.025836505388175692, 0.31567315973627313, -0.10067403003035807, 0.14697840454286856, 0.10128404782691164, 0.24550223393821832, -0.06979780249565032]) # weights=[0.3, 0.3, 0.5]
    # ensemble.extract(weights=[0.15985997782000064, -0.02712903419977486, 0.13194138929042556, 0.35350721304784233, 0.025836505388175692, 0.31567315973627313, -0.10067403003035807, 0.14697840454286856, 0.10128404782691164, 0.24550223393821832, -0.06979780249565032])
    






    # ----------------- Not Available now -----------------

    # def find_best_combinations_rand(self, top_n=5, num_weights_search=None):
    #     dev_df = pd.read_csv(self.target_file)
    #     targets = torch.Tensor(dev_df.iloc[:, -2].values)

    #     best_combinations = []
    #     output_file = "./output/temp_averaged_output.csv"
    #     if num_weights_search is not None:
    #         for r in range(1, len(self.input_files) + 1):
    #             for combo in combinations(self.input_files, r):
    #                 for _ in range(num_weights_search):
    #                     weights = np.random.dirichlet(np.ones(len(combo)), size=1)[0]
    #                     outputs = self.average_label(combo, weights=weights)
    #                     result = self.inference(outputs, targets)
    #                     val_pearson = result
    #                     best_combinations.append((combo, val_pearson, weights))
    #         best_combinations.sort(key=lambda x: x[1], reverse=True)
    #         for combo, val_pearson, weights in best_combinations[:top_n]:
    #             print(f"Combination: {combo}, val_pearson: {val_pearson}, weights: {weights}")
    #     else:
    #         for r in range(1, len(self.input_files) + 1):
    #             for combo in combinations(self.input_files, r):
    #                 outputs = self.average_label(combo)
    #                 result = self.inference(outputs, targets)
    #                 val_pearson = result
    #                 best_combinations.append((combo, val_pearson))
    #         best_combinations.sort(key=lambda x: x[1], reverse=True)
    #         for combo, val_pearson in best_combinations[:top_n]:
    #             print(f"Combination: {combo}, val_pearson: {val_pearson}")
    #     print(f"num of combinations: ", len(best_combinations))
    #     return best_combinations
    