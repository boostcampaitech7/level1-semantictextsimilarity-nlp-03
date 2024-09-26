import re
import argparse
import random
import pandas as pd
from tqdm.auto import tqdm
import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
import torch
import torchmetrics
import pytorch_lightning as pl
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import DataCollatorWithPadding
import bitsandbytes as bnb
import wandb
from pytorch_lightning.loggers import WandbLogger

class Dataset(torch.utils.data.Dataset):
    def __init__(self, inputs, targets=[]):
        self.inputs = inputs
        self.targets = targets

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.inputs.items()}
        if self.targets:
            item['labels'] = torch.tensor(self.targets[idx]).float()
        return item

    def __len__(self):
        return len(self.targets) if self.targets else len(next(iter(self.inputs.values())))


class Dataloader(pl.LightningDataModule):
    def __init__(self, model_name, batch_size, shuffle, train_path, dev_path, test_path, predict_path):
        super().__init__()
        self.model_name = model_name
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.predict_path = predict_path

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.target_columns = ['label']
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

        try:
            targets = data[self.target_columns].values.tolist()
        except:
            targets = []
        inputs = self.tokenizing(data)

        return inputs, targets

    def setup(self, stage='fit'):
        if stage == 'fit':
            train_data = pd.read_csv(self.train_path)
            val_data = pd.read_csv(self.dev_path)

            train_inputs, train_targets = self.preprocessing(train_data)
            val_inputs, val_targets = self.preprocessing(val_data)

            self.train_dataset = Dataset(train_inputs, train_targets)
            self.val_dataset = Dataset(val_inputs, val_targets)
        else:
            test_data = pd.read_csv(self.test_path)
            test_inputs, test_targets = self.preprocessing(test_data)
            self.test_dataset = Dataset(test_inputs, test_targets)

            predict_data = pd.read_csv(self.predict_path)
            predict_inputs, _ = self.preprocessing(predict_data)
            self.predict_dataset = Dataset(predict_inputs)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            collate_fn=self.data_collator
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator
        )

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(
            self.predict_dataset,
            batch_size=self.batch_size,
            collate_fn=self.data_collator
        )



class Model(pl.LightningModule):
    def __init__(self, model_name, lr):
        super().__init__()
        self.save_hyperparameters()
    
        self.model_name = model_name
        self.lr = lr
    
        # BitsAndBytesConfig 설정
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
    
        # LoRA 설정에서 wandb.config 사용
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=wandb.config.lora_r,
            lora_alpha=wandb.config.lora_alpha,
            lora_dropout=wandb.config.lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
    
        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = 1
    
        # 모델 로드
        self.plm = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=model_name,
            config=config,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # k-bit 훈련 준비
        self.plm = prepare_model_for_kbit_training(self.plm)
    
        # 분류 레이어의 requires_grad를 True로 설정
        for param in self.plm.score.parameters():
            param.requires_grad = True
    
        # LoRA 적용
        self.plm = get_peft_model(self.plm, lora_config)
    
        # 학습 가능한 파라미터 확인
        self.plm.print_trainable_parameters()
    
        self.loss_func = torch.nn.MSELoss()

    def forward(self, x):
        outputs = self.plm(**x)
        return outputs.logits.squeeze(-1)

    def training_step(self, batch, batch_idx):
        labels = batch.pop('labels').squeeze(-1)
        logits = self(batch)
        loss = self.loss_func(logits, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch.pop('labels').squeeze(-1)
        logits = self(batch)
        loss = self.loss_func(logits, labels)
        self.log("val_loss", loss)
        pearson_corr = torchmetrics.functional.pearson_corrcoef(logits, labels)
        self.log("val_pearson", pearson_corr)
        return loss

    def test_step(self, batch, batch_idx):
        labels = batch.pop('labels').squeeze(-1)
        logits = self(batch)
        pearson_corr = torchmetrics.functional.pearson_corrcoef(logits, labels)
        self.log("test_pearson", pearson_corr)

    def predict_step(self, batch, batch_idx):
        logits = self(batch)
        return logits

    def configure_optimizers(self):
        # 학습 가능한 파라미터 수집
        trainable_params = [p for p in self.plm.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)
        return optimizer

def train():
    with wandb.init():

        dataloader = Dataloader(
            wandb.config.model_name,
            wandb.config.batch_size,
            wandb.config.shuffle,
            wandb.config.train_path,
            wandb.config.dev_path,
            wandb.config.test_path,
            wandb.config.predict_path
        )
        model = Model(wandb.config.model_name, wandb.config.learning_rate)
        model.plm.config.pad_token_id = dataloader.tokenizer.pad_token_id
        wandb_logger = WandbLogger(project='LLM_STS')
        print(model)
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            max_epochs=wandb.config.max_epoch,
            log_every_n_steps=1,
            precision=16,
            logger=wandb_logger
        )

        trainer.fit(model=model, datamodule=dataloader)
        trainer.test(model=model, datamodule=dataloader)

        # 모델 저장
        model.plm.save_pretrained('qlora_model')
    

if __name__ == '__main__':
    # 하이퍼파라미터 기본값 설정
    default_config = {
        'model_name': 'beomi/Llama-3-Open-Ko-8B',
        'batch_size': 32,
        'max_epoch': 16,
        'shuffle': True,
        'learning_rate': 2e-4,
        'train_path': '../data/train.csv',
        'dev_path': '../data/dev.csv',
        'test_path': '../data/dev.csv',
        'predict_path': '../data/test.csv',
        'lora_r': 64,
        'lora_alpha': 16,
        'lora_dropout': 0.05
    }
    
    # 스윕 설정 정의
    sweep_config = {
        'method': 'bayes',
        'metric': {
            'name': 'val_pearson',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'values': [1e-5, 2e-5, 5e-5, 1e-4]
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'lora_r': {
                'values': [8, 16, 32, 64]
            },
            'lora_alpha': {
                'values': [16, 32, 64]
            },
            'lora_dropout': {
                'values': [0.0, 0.1, 0.2]
            },
            'model_name': {
                'values': ['beomi/Llama-3-Open-Ko-8B']
            },
            'max_epoch': {
                'value': 5  # 빠른 탐색을 위해 에포크 수를 줄임
            },
            'shuffle': {
                'value': True
            },
            'train_path': {
                'value': '../data/train.csv'
            },
            'dev_path': {
                'value': '../data/dev.csv'
            },
            'test_path': {
                'value': '../data/dev.csv'
            },
            'predict_path': {
                'value': '../data/test.csv'
            }
        }
    }
    
    # 스윕 생성
    sweep_id = wandb.sweep(sweep_config, project="LLM_STS")
    
    # 스윕 에이전트 실행
    wandb.agent(sweep_id, function=train)