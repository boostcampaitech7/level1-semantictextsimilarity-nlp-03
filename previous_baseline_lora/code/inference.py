import re
import argparse
import pandas as pd
from tqdm.auto import tqdm
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import torch
import pytorch_lightning as pl
from peft import PeftModel
from transformers import DataCollatorWithPadding
import bitsandbytes as bnb  # bitsandbytes 라이브러리 추가

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inputs.items()}
        return item

    def __len__(self):
        return len(next(iter(self.inputs.values())))


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size

        self.predict_path = predict_path
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.delete_columns = ['id']
        self.text_columns = ['sentence_1', 'sentence_2']

        self.data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)

    def tokenizing(self, dataframe):
        data = []
        for idx, item in tqdm(dataframe.iterrows(), desc='tokenizing', total=len(dataframe)):
            text = '[SEP]'.join([str(item[text_column]) for text_column in self.text_columns])
            outputs = self.tokenizer(
                text,
                add_special_tokens=True,
                padding='max_length',
                truncation=True,
                max_length=160
            )
            data.append(outputs)
        # 데이터 키별로 리스트를 묶어서 반환
        inputs = {key: [d[key] for d in data] for key in data[0]}
        return inputs

    def preprocess_text(self, text: str):
        # 전처리 내용 그대로 사용
        text = re.sub('<.*?>', '', text)
        text = re.sub(r'[^ㄱ-ㅎㅏ-ㅣ가-힣0-9\s.,?!:;^]', '', text)
        text = re.sub('[ㅋㅎㅠ]+', '', text)
        text = re.sub(r'([.?!,:]){2,}', r'\1', text)
        text = text.strip()
        return text

    def preprocessing(self, data):
        data = data.drop(columns=self.delete_columns)

        for column in self.text_columns:
            data[column] = data[column].astype(str).apply(self.preprocess_text)

        inputs = self.tokenizing(data)

        return inputs

    def setup(self, stage=None):
        predict_data = pd.read_csv(self.predict_path)
        predict_inputs = self.preprocessing(predict_data)
        self.predict_dataset = Dataset(predict_inputs)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator
        )


class Model(pl.LightningModule):
    def __init__(self, model_path):
        super().__init__()

        # 4-bit 양자화를 위한 모델 준비
        # self.base_model = AutoModelForSequenceClassification.from_pretrained(
        #     model_name,
        #     device_map="auto",
        #     quantization_config=BitsAndBytesConfig(
        #         load_in_4bit=True,
        #         bnb_4bit_use_double_quant=True,
        #         bnb_4bit_quant_type="nf4",
        #         bnb_4bit_compute_dtype=torch.float16
        #     ),
        #     num_labels = 1
        # )

        # # LoRA 어댑터 로드
        # self.plm = PeftModel.from_pretrained(self.base_model, 'qlora_model')

        self.plm = torch.load(model_path)
        

    def forward(self, x):
        outputs = self.plm(**x)
        return outputs.logits.squeeze(-1)

    def predict_step(self, batch, batch_idx):
        logits = self(batch)
        return logits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='BAAI/bge-multilingual-gemma2', type=str)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--predict_path', default='../data/dev.csv')
    args = parser.parse_args()

    dataloader = Dataloader(
        args.model_name,
        args.batch_size,
        predict_path=args.predict_path
    )

    dataloader.setup()

    model_path = '/data/ephemeral/home/level1-semantictextsimilarity-nlp-03/code/roberta.pt'  ## 실제 모델 경로로 수정

    model = Model(model_path)
    model.eval()  
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16  # Mixed Precision Training 적용
    )

    predictions = trainer.predict(model=model, datamodule=dataloader)

    # 예측된 결과를 형식에 맞게 반올림하여 준비합니다.
    predictions = list(round(float(i), 1) for i in torch.cat(predictions))

    output_df = pd.DataFrame(predictions, columns=["prediction"])

    model_name = model_path.split('/')[-1].replace('.pt', '')
    output_filename = f"{model_name}_dev_pr.csv"    ## 실제 저장 경로로 수정
    output_path = f"../data/{output_filename}"

    output_df.to_csv(output_path, index=False)

    print(f"Predictions saved to {output_path}")
