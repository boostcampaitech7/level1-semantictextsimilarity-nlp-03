import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

class STSModel(nn.Module):
    def __init__(self, plm_name):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.plm_name, num_labels=1, use_auth_token=False
        )

    def forward(self, x):
        x = self.plm(x)["logits"]
        return x


class WithDropout(nn.Module):
    def __init__(self, plm_name, dropout_rate, lora_r, lora_alpha, lora_dropout):
        super().__init__()
        self.plm_name = plm_name
        self.dropout = nn.Dropout(dropout_rate) 
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=self.plm_name, num_labels=1, use_auth_token=False
        )
        
    def forward(self, x):
        x = self.plm(x)["logits"]
        x = self.dropout(x)
        return x

class SLMModel(nn.Module):
    def __init__(self, plm_name, dropout_rate, lora_r, lora_alpha, lora_dropout):
        super().__init__()
    
        # BitsAndBytesConfig 
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        # LoraConfig
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["q_proj", "v_proj"]
        )
    
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name, 
            num_labels=1, 
            use_auth_token=False,
            quantization_config=bnb_config,
            device_map="auto"
        )

        # k-bit 훈련 준비
        self.plm = prepare_model_for_kbit_training(self.plm)
    
        for param in self.plm.score.parameters():
            param.requires_grad = True
    
        # LoRA 적용
        self.plm = get_peft_model(self.plm, lora_config)
    

    def forward(self, x):
        outputs = self.plm(x)
        return outputs.logits.squeeze(-1)


class SevenElevenWithBiLSTM(nn.Module):
    def __init__(self, plm_name, hidden_size=64):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name,
            num_labels=hidden_size,
            use_auth_token=True,
        )
        self.bilstm = nn.LSTM(
            hidden_size, hidden_size, num_layers=1, bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(
            hidden_size * 2, 1
        )  # 양방향 LSTM이기 때문에 hidden_size * 2

    def forward(self, x):
        x = self.plm(x)["logits"]
        x, _ = self.bilstm(x.unsqueeze(1))
        x = self.fc(x.squeeze(1))
        return x


class STSModelWithAttention(nn.Module):
    def __init__(self, plm_name, hidden_size=128):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name,
            num_labels=hidden_size,
            use_auth_token=True,
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = self.plm(x)["logits"]
        x, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = self.fc(x.squeeze(0))
        return x


class STSModelWithResidualConnection(nn.Module):
    def __init__(self, plm_name):
        super().__init__()
        self.plm_name = plm_name
        self.plm = transformers.AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=plm_name,
            num_labels=1,
            use_auth_token=True,
        )
        self.residual_fc = nn.Linear(1, 1)

    def forward(self, x):
        plm_output = self.plm(x)["logits"]
        residual = self.residual_fc(plm_output)  # Residual Connection 적용
        x = plm_output + residual  # Residual 연결
        return x
