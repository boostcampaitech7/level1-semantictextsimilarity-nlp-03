import torch
import pandas as pd
from abc import *
from base.base_dataset import BaseDataset
from tqdm.auto import tqdm


class STSDataset(BaseDataset):
    def __init__(self, data_path, tokenizer, col_info):
        """
        Args:
            data_path (str): csv data file path
            tokenizer : tokenizer
            col_info (dict): csv column information
        """
        super().__init__(data_path, tokenizer, col_info)

    def preprocessing(self, data):
        """
        Args:
            data (pandas.core.frame.DataFrame): data
        Returns:
            inputs (List[List[int]]): input list (each input is int list)
            targets (List[float] or List[int]): target list (each target is float(regression))
        """

        try:
            targets = data[self.col_info["label"]].values.tolist()
        except:
            targets = []
        # 텍스트 데이터를 전처리합니다.
        inputs = self.tokenizing(data)

        return inputs, targets

    # def tokenizing(self, dataframe):
    #     data = []
    #     for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
    #         # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
    #         text = '[SEP]'.join([item[text_column] for text_column in self.col_info['input']])
    #         outputs = self.tokenizer(text, add_special_tokens=True, padding='max_length', truncation=True)
    #         data.append(outputs['input_ids'])
    #     return data
    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(
            dataframe.iterrows(), desc="tokenizing", total=len(dataframe)
        ):
            # 두 입력 문장을 [SEP] 토큰으로 이어붙여서 전처리합니다.
            text = "[SEP]".join(
                [str(item[text_column]) for text_column in self.col_info["input"]]
            )
            outputs = self.tokenizer(
                text,
                add_special_tokens=True,
                truncation=True,
                padding=False,  # 패딩은 collator에서 처리
                return_tensors="pt",  # 텐서 형태로 반환
            )
            # 텐서를 스칼라로 변환하여 딕셔너리로 저장
            encoding = {key: val.squeeze(0) for key, val in outputs.items()}
            data.append(encoding)
        return data
