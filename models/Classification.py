from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
import transformers
from layers.StandardNorm import Normalize
from utils import my_read_data
from peft import LoraConfig, get_peft_model

transformers.logging.set_verbosity_error()


class RNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, head_dropout=0.1):
        super(RNNModel, self).__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(head_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, n, input_dim]
        _, (hidden, _) = self.rnn(x)  # hidden: [1, batch_size, hidden_dim]
        hidden = hidden.squeeze(0)  # [batch_size, hidden_dim]
        out = self.fc(hidden)  # [batch_size, num_classes]
        out = self.dropout(out)
        out = self.softmax(out)
        return out


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class ClassificationHead(nn.Module):
    def __init__(self, input_dim, num_classes, head_dropout=0):
        super(ClassificationHead, self).__init__()
        # print(input_dim)
        self.linear = nn.Linear(input_dim, num_classes)
        self.dropout = nn.Dropout(head_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        features = x  # 全连接层前的特征 [batch_size, input_dim]
        # x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.linear(x)
        x = self.dropout(x)
        x = self.softmax(x)
        return x


class DimensionalityReductionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DimensionalityReductionNet, self).__init__()
        # 特征提取部分
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=2, padding=1),  # 输入通道为3（RGB图像）
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()  # 展平
        )
        self.reducer = nn.Sequential(
            nn.Linear(64 * 16 * 16, 128),  # 降到中间维度
            nn.ReLU(),
            nn.Linear(128, output_dim)  # 最终降到低维度
        )
        # 解码器部分
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * 16 * 16),
            nn.ReLU(),
            nn.Unflatten(1, (64, 16, 16)),  # 恢复到卷积层输入的格式
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # 将值归一化到[0, 1]
        )

    def forward(self, x):
        encoded = self.encoder(x.to(torch.bfloat16))
        reduced = self.reducer(encoded)
        decoded = self.decoder(reduced)
        return reduced, decoded  # 返回降维后的特征和重建的图片


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.num_classes = configs.num_class
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.sampling_rate = configs.sampling_rate
        self.llm_model_root = configs.llm_model_root
        self.llm_lora = configs.llm_lora
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads

        if configs.llm_model == 'LLAMA':
            # self.llama_config = LlamaConfig.from_pretrained('/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/')
            self.llama_config = LlamaConfig.from_pretrained(self.llm_model_root)
            self.llama_config.num_hidden_layers = configs.llm_layers
            self.llama_config.output_attentions = True
            self.llama_config.output_hidden_states = True
            try:
                # self.llm_model = AutoModelForCausalLM.from_pretrained(
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.llama_config,
                    ignore_mismatched_sizes=True,
                    # torch_dtype=torch.float32,
                    # device_map="auto"
                    # load_in_4bit=True
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = LlamaModel.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/",
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.llama_config,
                    # load_in_8bit=True
                    # load_in_4bit=True
                )
            try:
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=True,
                    # torch_dtype=torch.float32,
                    # device_map="auto"
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = LlamaTokenizer.from_pretrained(
                    # "/mnt/alps/modelhub/pretrained_model/LLaMA/7B_hf/tokenizer.model",
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'GPT2':
            self.gpt2_config = GPT2Config.from_pretrained(self.llm_model_root)

            self.gpt2_config.num_hidden_layers = configs.llm_layers
            self.gpt2_config.output_attentions = True
            self.gpt2_config.output_hidden_states = True
            try:
                self.llm_model = GPT2Model.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.gpt2_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = GPT2Model.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.gpt2_config,
                )

            try:
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = GPT2Tokenizer.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'BERT':
            self.bert_config = BertConfig.from_pretrained(self.llm_model_root)

            self.bert_config.num_hidden_layers = configs.llm_layers
            self.bert_config.output_attentions = True
            self.bert_config.output_hidden_states = True
            try:
                self.llm_model = BertModel.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=True,
                    config=self.bert_config,
                )
            except EnvironmentError:  # downloads model from HF is not already done
                print("Local model files not found. Attempting to download...")
                self.llm_model = BertModel.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=False,
                    config=self.bert_config,
                )
            try:
                self.tokenizer = BertTokenizer.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=True
                )
            except EnvironmentError:  # downloads the tokenizer from HF if not already done
                print("Local tokenizer files not found. Atempting to download them..")
                self.tokenizer = BertTokenizer.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'deepseek':
            # 尝试加载本地模型
            self.deepseek_config = AutoConfig.from_pretrained(self.llm_model_root)
            self.deepseek_config.num_hidden_layers = configs.llm_layers
            self.deepseek_config.output_attentions = True
            self.deepseek_config.output_hidden_states = True
            self.deepseek_config.hidden_size = self.d_llm
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_root,
                    config=self.deepseek_config,
                    trust_remote_code=True,
                    local_files_only=True,
                    # device_map="auto",
                    ignore_mismatched_sizes=True
                )
            except EnvironmentError:  # 本地文件不存在，则从 Hugging Face 下载
                print("Local model files not found. Attempting to download...")
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=False
                )

            # 尝试加载本地 tokenizer
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=True,
                    # device_map="auto",
                )
            except EnvironmentError:  # 本地文件不存在，则从 Hugging Face 下载
                print("Local tokenizer files not found. Attempting to download...")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.llm_model_root,
                    trust_remote_code=True,
                    local_files_only=False
                )
        else:
            raise Exception('LLM model is not defined')

        if self.tokenizer.eos_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            pad_token = '[PAD]'
            self.tokenizer.add_special_tokens({'pad_token': pad_token})
            self.tokenizer.pad_token = pad_token

        # 配置 LoRA 参数（例如对模型中的查询和投影层进行微调，具体 target_modules 根据模型架构调整）
        try:
            if self.llm_lora:
                lora_config = LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=["q_proj", "v_proj", "k_proj"],  # 扩展适配模块
                    lora_dropout=0.1,
                    bias="lora_only",
                    modules_to_save=["classifier"],  # 微调分类头
                    task_type="SEQ_CLS",
                )
                self.llm_model = get_peft_model(self.llm_model, lora_config)
                self.llm_model.print_trainable_parameters()  # 查看可训练参数比例【&#8203;:contentReference[oaicite:0]{index=0}】
            else:
                for param in self.llm_model.parameters():
                    param.requires_grad = False
        except:
            for param in self.llm_model.parameters():
                param.requires_grad = False

        self.description = 'This dataset consists of vibration data collected from bearings. The data is captured over time and each record corresponds to a time series of vibration signals. These signals are recorded in the time domain and are commonly used for condition classification tasks. The data is sampled at a frequency of 12000Hz, ensuring high-resolution measurements for fault detection and classification tasks.'

        self.dropout = nn.Dropout(configs.dropout)

        # 第一步：图片降维层 600
        self.DimensionalityReductionNet = DimensionalityReductionNet(4 * 128 * 128, configs.enc_out)
        # my step
        self.pic_enc = DimensionalityReductionNet(input_dim=None, output_dim=configs.enc_out)
        # 第4步
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        # 第3步
        self.reprogramming_layer = ReprogrammingLayer(self.d_model, self.n_heads, d_llm=configs.llm_dim)

        # 第1步
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

        self.output_projection = ClassificationHead(configs.llm_dim, self.num_classes, head_dropout=configs.dropout)

    def forward(self, batch_csv, batch_iamges, batch_y):

        return self.classify(batch_csv, batch_iamges)

    def classify(self, batch_csv, batch_iamges):
        # 第1步 对csv进行处理
        # x_enc = self.normalize_layers(batch_csv, 'norm')  # 归一化，维度不变
        x_enc = batch_csv

        mean_value, variance, std_deviation, max_value, min_value, peak_value, kurt, skewness, rms_value, crest_factor = my_read_data.time_feature(
            x_enc, batch_csv.device)
        psd, total_power, peak_frequency, rms_frequency, center_frequency = my_read_data.fft_feature(x_enc,
                                                                                                     self.sampling_rate,
                                                                                                     batch_csv.device)
        prompt = []
        for b in range(x_enc.shape[0]):
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                "Task description: classify the sequence into predefined categories; "
                "Input statistics: "
                f"mean value {str(mean_value[b].tolist())}, "
                f"variance value {str(variance[b].tolist())}, "
                f"standard deviation value {str(std_deviation[b].tolist())}, "
                f"max value is {str(max_value[b].tolist())}, "
                f"min value is {str(min_value[b].tolist())}, "
                f"peak value is {str(peak_value[b].tolist())}, "
                f"kurtosis value is {str(kurt[b].tolist())}, "
                f"skewness value is {str(skewness[b].tolist())}, "
                f"root mean square (RMS) value is {str(rms_value[b].tolist())}, "
                f"crest factor value is : {str(crest_factor[b].tolist())}, "
                f"peak frequency value is {str(peak_frequency[b].tolist())}, "
                f"root mean square (RMS) frequency value is {str(rms_frequency[b].tolist())}, "
                f"center frequency value is {str(center_frequency[b].tolist())}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

        # 第4步
        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                max_length=4096).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # 第2步
        enc_out = x_enc.to(torch.bfloat16).unfold(1, self.patch_len, self.stride)
        pic_enc, pic_decoded = self.pic_enc(batch_iamges)
        pic_enc = pic_enc.unfold(1, self.patch_len, self.stride)
        enc_out = torch.cat([enc_out, pic_enc], dim=-1)
        # 第3步
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)  # 多头注意力
        # word-csv-images
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # 前向传播
        outputs = self.llm_model(inputs_embeds=llama_enc_out, output_hidden_states=True)

        # 获取隐藏状态

        hidden_states = outputs.hidden_states  # 这是一个元组，每一层的隐藏状态
        dec_out = hidden_states[-1]  # 取最后一层隐藏状态

        dec_out = dec_out.mean(dim=1)  # 16 * 4096  池化特征，降低维度

        # 第5步 输出层
        dec_out = self.output_projection(dec_out)  # 正常输出

        return dec_out


class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_keys=None, d_llm=None, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        # d_keys默认为d_model // n_heads
        d_keys = d_keys or (d_model // n_heads)

        # 将目标、源和值嵌入分别投影到指定的维度
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.value_projection = nn.Linear(d_llm, d_keys * n_heads)
        self.out_projection = nn.Linear(d_keys * n_heads, d_llm)
        self.n_heads = n_heads
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, target_embedding, source_embedding, value_embedding):
        B, L, _ = target_embedding.shape  # 目标数据的batch size和长度
        S, _ = source_embedding.shape  # 源数据的长度
        H = self.n_heads  # 头数

        # 将目标数据投影到查询、源数据投影到键和值数据投影到值
        target_embedding = self.query_projection(target_embedding).view(B, L, H, -1)
        source_embedding = self.key_projection(source_embedding).view(S, H, -1)
        value_embedding = self.value_projection(value_embedding).view(S, H, -1)

        # 进行重编程操作
        out = self.reprogramming(target_embedding, source_embedding, value_embedding)

        # 重新整理输出的形状
        out = out.reshape(B, L, -1)

        return self.out_projection(out)

    def reprogramming(self, target_embedding, source_embedding, value_embedding):
        B, L, H, E = target_embedding.shape  # B：batch size，L：目标长度，H：头数，E：嵌入维度

        scale = 1. / sqrt(E)

        # 计算目标和源嵌入之间的注意力分数
        scores = torch.einsum("blhe,she->bhls", target_embedding, source_embedding)

        # 计算注意力权重A
        A = self.dropout(torch.softmax(scale * scores, dim=-1))

        # 计算重编程嵌入
        reprogramming_embedding = torch.einsum("bhls,she->blhe", A, value_embedding)

        return reprogramming_embedding
