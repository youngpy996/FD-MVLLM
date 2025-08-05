from math import sqrt

import torch
import torch.nn as nn

from transformers import LlamaConfig, LlamaModel, LlamaTokenizer, GPT2Config, GPT2Model, GPT2Tokenizer, BertConfig, \
    BertModel, BertTokenizer, AutoModelForCausalLM, AutoTokenizer, AutoConfig
from layers.Embed import PatchEmbedding
import transformers
from layers.StandardNorm import Normalize
from utils import my_read_data
from peft import LoraConfig, get_peft_model

transformers.logging.set_verbosity_error()


class CNNModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, head_dropout=0.1):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, 64, kernel_size=3, stride=1, padding=1)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(head_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x: [batch_size, n, input_dim]
        x = x.permute(0, 2, 1)  # 转换为 [batch_size, input_dim, n]
        x = torch.relu(self.conv1(x))  # [batch_size, 128, n]
        x = torch.relu(self.conv2(x))  # [batch_size, 64, n]
        x = self.global_pool(x)  # [batch_size, 64, 1]
        x = x.squeeze(-1)  # [batch_size, 64]
        out = self.fc(x)  # [batch_size, num_classes]
        out = self.dropout(out)
        out = self.softmax(out)
        return out


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
    def __init__(self, configs, patch_len=16, stride=8):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.num_classes = configs.num_class
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.d_ff = configs.d_ff
        self.top_k = 5
        self.d_llm = configs.llm_dim
        self.patch_len = configs.patch_len
        self.stride = configs.stride
        self.sampling_rate = configs.sampling_rate
        self.llm_model_root = configs.llm_model_root
        self.llm_lora = configs.llm_lora

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
                    r'E:\dayoung\Llama-3-8B',
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
                    r'E:\dayoung\Llama-3-8B/tokenizer.model',
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
                    r'E:\young\online\gpt2',
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
                    r'E:\young\online\gpt2',
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
                    r'E:\dayoung\bert-base-uncased',
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
                    r'E:\dayoung\bert-base-uncased',
                    trust_remote_code=True,
                    local_files_only=False
                )
        elif configs.llm_model == 'deepseek':
            # 尝试加载本地模型
            self.deepseek_config = AutoConfig.from_pretrained(self.llm_model_root)
            self.deepseek_config.num_hidden_layers = configs.llm_layers
            self.deepseek_config.output_attentions = True
            self.deepseek_config.output_hidden_states = True
            try:
                self.llm_model = AutoModelForCausalLM.from_pretrained(
                    self.llm_model_root,
                    config=self.deepseek_config,
                    trust_remote_code=True,
                    local_files_only=True,
                    # device_map="auto",
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
                    r'E:\young\online\DeepSeek-R1-Distill-Qwen-7B',
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

        # for param in self.llm_model.parameters():
        #     param.requires_grad = False
        # 配置 LoRA 参数（例如对模型中的查询和投影层进行微调，具体 target_modules 根据模型架构调整）
        # if self.llm_lora:
        try:
            lora_config = LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj"],  # 扩展适配模块
                lora_dropout=0.2,
                bias="lora_only",
                modules_to_save=["classifier"],  # 微调分类头
                task_type="SEQ_CLS",
            )

            # 将模型包装为可进行 LoRA 微调的模型
            self.llm_model = get_peft_model(self.llm_model, lora_config)
            self.llm_model.print_trainable_parameters()  # 查看可训练参数比例【&#8203;:contentReference[oaicite:0]{index=0}】
        # else:
        except:
            for param in self.llm_model.parameters():
                param.requires_grad = False

        if configs.prompt_domain:
            self.description = configs.content
        else:
            self.description = 'This dataset consists of vibration data collected from bearings. The data is captured over time and each record corresponds to a time series of vibration signals. These signals are recorded in the time domain and are commonly used for condition classification tasks. The data is sampled at a frequency of 10000Hz, ensuring high-resolution measurements for fault detection and classification tasks.'
            # self.description = 'The Electricity Transformer Temperature (ETT) is a crucial indicator in the electric power long-term deployment.'

        self.dropout = nn.Dropout(configs.dropout)

        # 第一步：图片降维层 600
        self.DimensionalityReductionNet = DimensionalityReductionNet(4 * 128 * 128, configs.enc_out)
        # 初始化模型和加载数据
        # input_dim = 128 * 128 * 3  # 原始图片尺寸
        # output_dim = 64  # 降维后的维度
        # model = DimensionalityReductionNet(input_dim=input_dim, output_dim=output_dim).to(device)
        # 第2步
        self.patch_embedding = PatchEmbedding(
            configs.d_model, self.patch_len, 0, configs.dropout)
        # my step
        self.pic_enc = DimensionalityReductionNet(input_dim=None, output_dim=configs.enc_out)
        # 第4步
        self.word_embeddings = self.llm_model.get_input_embeddings().weight
        self.vocab_size = self.word_embeddings.shape[0]
        self.num_tokens = 1000
        self.attention_dim = 1000

        self.mapping_layer = nn.Linear(self.vocab_size, self.num_tokens)
        self.reprogramming_layer = ReprogrammingLayer(
            d_model=configs.d_model,
            d_attention=self.attention_dim,
            d_llm=configs.llm_dim,
            n_heads=8
        )
        # 第3步
        # self.reprogramming_layer = ReprogrammingLayer(configs.llm_dim, configs.n_heads)
        self.reprogramming_layer = ReprogrammingLayer(128, 8, d_llm=configs.llm_dim)

        self.patch_nums = int((configs.seq_len - self.patch_len) / self.stride + 2)
        self.head_nf = self.d_ff * self.patch_nums

        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.output_projection = FlattenHead(configs.enc_in, self.head_nf, self.pred_len,
                                                 head_dropout=configs.dropout)

        elif self.task_name == 'classification':
            # input_dim, num_classes, head_dropout=0
            self.output_projection = ClassificationHead(configs.llm_dim, self.num_classes,
                                                        head_dropout=configs.dropout)
            # self.output_projection = RNNModel(configs.llm_dim, 1024, self.num_classes,
            #                                      head_dropout=configs.dropout)
            # self.output_projection = CNNModel(configs.llm_dim, 1024, self.num_classes,
            #                                   head_dropout=configs.dropout)
        else:
            raise NotImplementedError
        # 第1步
        self.normalize_layers = Normalize(configs.enc_in, affine=False)

    def forward(self, batch_csv, batch_iamges, batch_y, mask=None):
        # def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(batch_csv, batch_iamges, batch_y, batch_y)  # 随便放置
            return dec_out[:, -self.pred_len:, :]
        elif self.task_name == 'classification':
            return self.classify(batch_csv, batch_iamges, batch_y)
        return None

    def classify(self, batch_csv, batch_iamges, batch_y):
        # def classify(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 第1步 对csv进行处理
        x_enc = self.normalize_layers(batch_csv, 'norm')  # 归一化，维度不变

        # B, T, N = x_enc.size()
        # x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        # lags = self.calcute_lags(x_enc)
        # trends = x_enc.diff(dim=1).sum(dim=1)

        mean_value, variance, std_deviation, max_value, min_value, peak_value, kurt, skewness, rms_value, crest_factor = my_read_data.time_feature(
            x_enc, batch_csv.device)
        psd, total_power, peak_frequency, rms_frequency, center_frequency = my_read_data.fft_feature(x_enc,
                                                                                                     self.sampling_rate,
                                                                                                     batch_csv.device)

        prompt = []
        for b in range(x_enc.shape[0]):
            # psd, total_power, peak_frequency, rms_frequency, center_frequency = my_read_data.fft_feature(x_enc[b],
            #                                                                                              self.sampling_rate,
            #                                                                                              batch_csv.device,
            #                                                                                              self.dataset_type,
            #                                                                                              batch_y[b])

            # min_values_str = str(min_values[b].tolist()[0])
            # max_values_str = str(max_values[b].tolist()[0])
            # median_values_str = str(medians[b].tolist()[0])
            # # lags_values_str = str(lags[b].tolist())
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
                f"peak frequency value is {str(peak_frequency.tolist())}, "
                f"root mean square (RMS) frequency value is {str(rms_frequency.tolist())}, "
                f"center frequency value is {str(center_frequency.tolist())}<|<end_prompt>|>"
            )
            prompt.append(prompt_)

            # f"power spectral density value is {psd[b].tolist()[0]}, "
            # f"total Power value is {total_power[b].tolist()[0]}, "
        # 第4步
        # x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True,
                                max_length=2048).input_ids  # 16*477
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)
        # 第2步
        # x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))  # x_en 16 * 600
        enc_out = enc_out.unfold(1, 64, 16)
        # my step
        pic_enc, pic_decoded = self.pic_enc(batch_iamges)
        pic_enc = pic_enc.unfold(1, 64, 16)
        # enc_out = torch.stack((enc_out, pic_enc), dim=1).permute(0, 2, 1).contiguous()
        enc_out = torch.cat([enc_out, pic_enc], dim=-1)
        # 第3步
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)  # 多头注意力
        # 拼接word-csv-images
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        # 获取大模型输出前一层
        # dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        # 前向传播
        outputs = self.llm_model(inputs_embeds=llama_enc_out, output_hidden_states=True)

        # 获取隐藏状态
        hidden_states = outputs.hidden_states  # 这是一个元组，每一层的隐藏状态
        dec_out = hidden_states[-1]  # 取最后一层隐藏状态
        # print(dec_out.shape)
        # dec_out = dec_out.view(dec_out.size(0), -1)
        # print("原始输入形状:", dec_out.shape)
        # dec_out = dec_out.permute(0, -1).contiguous()
        dec_out = dec_out.mean(dim=1)  # 16 * 4096  池化特征，降低维度
        # dec_out = dec_out[:, :, 128:]
        # dec_out = dec_out[:, :, :self.d_ff]

        # dec_out = torch.reshape(
        #     dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        # dec_out = dec_out.permute(0, 1, 3, 2).contiguous()
        # 第5步 输出层
        # dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = self.output_projection(dec_out)  # 正常输出
        # dec_out = dec_out.permute(0, 2, 1).contiguous()
        # dec_out = torch.argmax(dec_out, dim=1)
        # dec_out = torch.tensor(dec_out)
        # dec_out = torch.tensor(dec_out, dtype=torch.long)
        # print(dec_out.shape)
        # features, logits = self.output_projection(dec_out)  # TNSE 获取特征和预测结果
        return dec_out

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # 时域数据归一化
        x_enc = self.normalize_layers(x_enc, 'norm')

        B, T, N = x_enc.size()
        x_enc = x_enc.permute(0, 2, 1).contiguous().reshape(B * N, T, 1)

        min_values = torch.min(x_enc, dim=1)[0]
        max_values = torch.max(x_enc, dim=1)[0]
        medians = torch.median(x_enc, dim=1).values
        lags = self.calcute_lags(x_enc)
        trends = x_enc.diff(dim=1).sum(dim=1)

        prompt = []
        for b in range(x_enc.shape[0]):
            min_values_str = str(min_values[b].tolist()[0])
            max_values_str = str(max_values[b].tolist()[0])
            median_values_str = str(medians[b].tolist()[0])
            lags_values_str = str(lags[b].tolist())
            prompt_ = (
                f"<|start_prompt|>Dataset description: {self.description}"
                f"Task description: forecast the next {str(self.pred_len)} steps given the previous {str(self.seq_len)} steps information; "
                "Input statistics: "
                f"min value {min_values_str}, "
                f"max value {max_values_str}, "
                f"median value {median_values_str}, "
                f"the trend of input is {'upward' if trends[b] > 0 else 'downward'}<|<end_prompt>|>"
            )

            prompt.append(prompt_)

        x_enc = x_enc.reshape(B, N, T).permute(0, 2, 1).contiguous()

        prompt = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=2048).input_ids
        prompt_embeddings = self.llm_model.get_input_embeddings()(prompt.to(x_enc.device))  # (batch, prompt_token, dim)

        source_embeddings = self.mapping_layer(self.word_embeddings.permute(1, 0)).permute(1, 0)

        x_enc = x_enc.permute(0, 2, 1).contiguous()
        enc_out, n_vars = self.patch_embedding(x_enc.to(torch.bfloat16))
        enc_out = self.reprogramming_layer(enc_out, source_embeddings, source_embeddings)

        # 进行拼接
        llama_enc_out = torch.cat([prompt_embeddings, enc_out], dim=1)
        dec_out = self.llm_model(inputs_embeds=llama_enc_out).last_hidden_state
        dec_out = dec_out[:, :, :self.d_ff]

        dec_out = torch.reshape(
            dec_out, (-1, n_vars, dec_out.shape[-2], dec_out.shape[-1]))
        dec_out = dec_out.permute(0, 1, 3, 2).contiguous()

        dec_out = self.output_projection(dec_out[:, :, :, -self.patch_nums:])
        dec_out = dec_out.permute(0, 2, 1).contiguous()

        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out

    def calcute_lags(self, x_enc):
        q_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(x_enc.permute(0, 2, 1).contiguous(), dim=-1)
        res = q_fft * torch.conj(k_fft)
        corr = torch.fft.irfft(res, dim=-1)
        mean_value = torch.mean(corr, dim=1)
        _, lags = torch.topk(mean_value, self.top_k, dim=-1)
        return lags


class MultiHeadAttentionReprogramming(nn.Module):
    # class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(MultiHeadAttentionReprogramming, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        # Define linear layers for query, key, value
        self.query_proj = nn.Linear(embed_dim, embed_dim)
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5  # Scaling factor for attention

    def forward(self, target_embedding, source_embedding, value_embedding):
        batch_size, target_len, embed_dim = target_embedding.size()
        source_len = source_embedding.size(1)

        # Linear projections
        query = self.query_proj(target_embedding)  # (batch_size, target_len, embed_dim)
        key = self.key_proj(source_embedding)  # (batch_size, source_len, embed_dim)
        value = self.value_proj(value_embedding)  # (batch_size, source_len, embed_dim)

        # Reshape for multi-head attention
        query = query.view(batch_size, target_len, self.num_heads, self.head_dim).transpose(1,
                                                                                            2)  # (batch_size, num_heads, target_len, head_dim)
        key = key.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1,
                                                                                        2)  # (batch_size, num_heads, source_len, head_dim)
        value = value.view(batch_size, source_len, self.num_heads, self.head_dim).transpose(1,
                                                                                            2)  # (batch_size, num_heads, source_len, head_dim)

        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2,
                                                             -1)) * self.scale  # (batch_size, num_heads, target_len, source_len)
        attention_probs = torch.softmax(attention_scores, dim=-1)  # Normalize scores
        attention_probs = self.dropout(attention_probs)

        # Weighted sum of value
        attention_output = torch.matmul(attention_probs, value)  # (batch_size, num_heads, target_len, head_dim)

        # Combine heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, target_len, embed_dim)

        # Final output projection
        output = self.out_proj(attention_output)  # (batch_size, target_len, embed_dim)
        return output


# # Example usage
# batch_size = 16
# target_len = 1208
# source_len = 1000
# embed_dim = 4096  # You can adjust this based on your task
# num_heads = 8
#
# target_embedding = torch.randn(batch_size, target_len, embed_dim).cuda()
# source_embedding = torch.randn(batch_size, source_len, embed_dim).cuda()
# value_embedding = source_embedding  # Typically the same as source_embedding
#
# mha = MultiHeadAttention(embed_dim, num_heads).cuda()
# output = mha(target_embedding, source_embedding, value_embedding)
# print(output.shape)  # Should be (batch_size, target_len, embed_dim)

class ReprogrammingLayer(nn.Module):
    def __init__(self, d_model, d_attention, d_llm, n_heads, attention_dropout=0.1):
        super(ReprogrammingLayer, self).__init__()

        self.n_heads = n_heads
        self.d_attention = d_attention
        self.head_dim = d_attention // n_heads
        assert self.head_dim * n_heads == self.d_attention, "d_attention must be divisible by n_heads"

        # Projection layers to map inputs into the common attention space (d_attention)
        self.query_projection = nn.Linear(d_model, self.d_attention)
        self.key_projection = nn.Linear(d_llm, self.d_attention)
        self.value_projection = nn.Linear(d_llm, self.d_attention)

        # Final projection layer to map the attention output back to the LLM's dimension
        self.out_projection = nn.Linear(self.d_attention, d_llm)

        self.dropout = nn.Dropout(attention_dropout)
        self.scale = self.head_dim ** -0.5

    def forward(self, target_embedding, source_embedding, value_embedding):
        # target_embedding (from time-series/image):
        # source_embedding (from LLM vocab):
        # value_embedding (from LLM vocab):

        B, L, _ = target_embedding.shape
        S, _ = source_embedding.shape
        H = self.n_heads

        # 1. Project Q, K, V into the attention space (d_attention)
        # Query from target (time-series/image)
        query = self.query_projection(target_embedding)  #
        # Key and Value from source (learnable tokens)
        key = self.key_projection(source_embedding)  #
        value = self.value_projection(value_embedding)  #

        # 2. Reshape for Multi-Head Attention
        query = query.view(B, L, H, self.head_dim)  #
        key = key.view(S, H, self.head_dim)  #
        value = value.view(S, H, self.head_dim)  #

        # 3. Perform Scaled Dot-Product Attention in d_attention space
        # einsum computes batched dot product: (B, L, H, E) x (S, H, E) -> (B, H, L, S)
        scores = torch.einsum("blhe,she->bhls", query, key) * self.scale

        A = self.dropout(torch.softmax(scores, dim=-1))  # Attention weights

        # 4. Compute weighted sum of values
        # (B, H, L, S) x (S, H, E) -> (B, L, H, E)
        reprogrammed_embedding = torch.einsum("bhls,she->blhe", A, value)

        # 5. Concatenate heads and project back to d_llm
        reprogrammed_embedding = reprogrammed_embedding.reshape(B, L, self.d_attention)
        out = self.out_projection(reprogrammed_embedding)  #

        return out
