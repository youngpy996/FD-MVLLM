import argparse

import pandas as pd
import torch
from accelerate import Accelerator, DeepSpeedPlugin
from accelerate import DistributedDataParallelKwargs
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Autoformer, DLinear, Classification
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, random_split
from data_provider.data_factory import data_provider
import time
import random
import numpy as np
import os
from utils import my_read_data
from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content, compute_f1_score, \
    extract_features, plot_tsne, get_features

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Time-LLM')

    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='classification',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--num_class', type=int, default=4, help='number of class')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model_comment', type=str, required=True, default='none',
                        help='prefix when saving test results')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',
                        help='model name, options: [Autoformer, DLinear]')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='ETTm1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, S, MS]; '
                             'M:multivariate predict multivariate, S: univariate predict univariate, '
                             'MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--loader', type=str, default='modal', help='dataset type')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, '
                             'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                             'you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--output_csv', default="./output_windowed_data.csv", help='iamge2data output')
    parser.add_argument('--sampling_rate', type=int, default=50000, help='sampling_rate')
    parser.add_argument('--overlap', type=int, default=512, help='滑动窗口重叠大小')
    parser.add_argument('--window_size', type=int, default=1024, help='滑动窗口大小')
    parser.add_argument('--csv_root', type=str, default=r'E:\young\JNU-Bearing-Dataset-main\input', help='csv路径')
    parser.add_argument('--image_root', type=str, default=r"E:\young\JNU-Bearing-Dataset-main\results600-200-200",
                        help='图片路径')
    # parser.add_argument('--csv_root', type=str, default=r'E:\dayoung\matlab output\input', help='csv路径')
    # parser.add_argument('--image_root', type=str, default=r"E:\dayoung\matlab output\results",
    #                     help='图片路径')
    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

    # model define
    parser.add_argument('--enc_in', type=int, default=512, help='encoder input size csv')
    parser.add_argument('--enc_out', type=int, default=512, help='encoder output size picture')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size')
    parser.add_argument('--d_model', type=int, default=8, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    parser.add_argument('--patch_len', type=int, default=8, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--prompt_domain', type=int, default=0, help='')
    parser.add_argument('--llm_model_root', type=str, default=r'E:\young\online\DeepSeek-R1-Distill-Llama-8B', help='LLM model root')  # LLAMA, GPT2, BERT, deepseek
    parser.add_argument('--llm_model', type=str, default='deepseek', help='LLM model')  # LLAMA, GPT2, BERT, deepseek
    parser.add_argument('--llm_dim', type=int, default='4096',
                        help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768; deepseek-llama:4096; deepseek-qwen:3584
    parser.add_argument('--llm_lora', type=bool, default=True, help='LLM lora')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=50, help='train epochs')
    parser.add_argument('--align_epochs', type=int, default=50, help='alignment epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test',help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', help='type1 adjust learning rate TST / COS')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--llm_layers', type=int, default=2)
    parser.add_argument('--percent', type=int, default=100)

    args = parser.parse_args()

    root_file = os.getcwd()
    save_file = f'results {os.path.basename(args.llm_model_root)} epo{args.train_epochs} tsne'
    save_root = os.path.join(root_file, save_file)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    deepspeed_plugin = DeepSpeedPlugin(hf_ds_config='./ds_config_zero2.json')
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
    # accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], deepspeed_plugin=deepspeed_plugin)

    for ii in range(args.itr):
        # setting record of experiments
        setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}_{}'.format(
            args.task_name,
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers,
            args.d_ff,
            args.factor,
            args.embed,
            args.des, ii)

        # train_data, train_loader = data_provider(args, 'train')
        # vali_data, vali_loader = data_provider(args, 'val')
        # test_data, test_loader = data_provider(args, 'test')
        # image_root ([112, 4, 128, 128])  data_csv (28, 600)
        data_csv, image_data, labels = my_read_data.read_data(args.csv_root, args.image_root, args.output_csv,
                                                              args.window_size, args.overlap)

        # data_csv = TensorDataset(data_csv)
        # image_data = TensorDataset(image_data)

        # dataloader_data_csv = DataLoader(data_csv, batch_size=16, shuffle=False)
        # dataloader_image_data = DataLoader(image_data, batch_size=16, shuffle=False)
        # data_len = len(data_csv)
        # combined_dataset = ConcatDataset([data_csv, image_data])
        # combined_dataset = my_read_data.CombinedDataset(data_csv, image_data)
        # combined_dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=False)

        # 生成随机索引
        train_loader, test_loader, val_loader = my_read_data.data_indices(data_csv, image_data, labels, args)

        # # 数据集划分比例
        # train_ratio = 0.7
        # val_ratio = 0.2
        # test_ratio = 0.1
        #
        # # 计算每个数据集的大小
        # train_size = int(len(combined_dataset) * 0.7)
        # test_size = int(len(combined_dataset) * 0.2)
        # val_size = len(combined_dataset) - train_size - test_size

        # 随机划分数据集
        # train_loader, vali_loader, test_loader = random_split(combined_dataset, [train_size, val_size, test_size])

        # 创建 DataLoader
        # combined_dataloader = DataLoader(combined_dataset, batch_size=16, shuffle=True)

        if args.model == 'Autoformer':
            model = Autoformer.Model(args).to(torch.bfloat16)
        elif args.model == 'DLinear':
            model = DLinear.Model(args).to(torch.bfloat16)
        else:
            model = Classification.Model(args).to(torch.bfloat16)

        path = os.path.join(args.checkpoints,
                            setting + '-' + args.model_comment)  # unique checkpoint saving path
        args.content = load_content(args)
        if not os.path.exists(path) and accelerator.is_local_main_process:
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(accelerator=accelerator, patience=args.patience)

        # 收集所有需要更新的模型参数
        trained_parameters = []
        for p in model.parameters():
            if p.requires_grad is True:
                trained_parameters.append(p)

        model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

        if args.lradj == 'COS':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
        else:
            scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                                steps_per_epoch=train_steps,
                                                pct_start=args.pct_start,
                                                epochs=args.train_epochs,
                                                max_lr=args.learning_rate)

        criterion = nn.CrossEntropyLoss()
        mae_metric = nn.L1Loss()

        train_loader, test_loader, val_loader, model, model_optim, scheduler = accelerator.prepare(
            train_loader, test_loader, val_loader, model,
            model_optim, scheduler)
        # # 提取 train_dataset 中属于第一个数据集的数据
        # first_dataset_indices = [i for i, (_, source) in enumerate(train_loader) if source == 0]
        # first_dataset = [train_loader[i][0] for i in first_dataset_indices]
        # # first_dataset = torch.stack(first_dataset)  # 转为 torch.Tensor
        # # 转换为 DataLoader（可选）
        # tensor1 = DataLoader(first_dataset, batch_size=16, shuffle=False)

        if args.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        accuracy_cha = []
        accuracy_val_cha = []
        accuracy_test_cha = []
        train_f1_score = []
        val_f1_score = []
        test_f1_score = []
        for epoch in range(args.train_epochs):
            iter_count = 0
            train_loss = []
            correct = 0
            model.train()
            epoch_time = time.time()
            all_preds = []
            all_labels = []
            for i, (batch_csv, batch_images, batch_y) in tqdm(enumerate(train_loader)):
                # print(type(batch_csv), type(batch_images), type(batch_y))
                # print(iter_count)
                iter_count += 1

                model_optim.zero_grad()

                batch_csv = batch_csv.float().to(accelerator.device)
                batch_images = batch_images.float().to(accelerator.device)
                batch_y = batch_y.long().to(accelerator.device)
                # batch_x_mark = batch_x_mark.float().to(accelerator.device)
                # batch_y_mark = batch_y_mark.float().to(accelerator.device)

                # decoder input
                # dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float().to(
                #     accelerator.device)
                # dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
                #     accelerator.device)

                # encoder - decoder
                if args.use_amp:
                    with torch.cuda.amp.autocast():
                        if args.output_attention:
                            outputs = model(batch_csv, batch_images, batch_y)[0]
                        else:
                            outputs = model(batch_csv, batch_images, batch_y)

                        f_dim = -1 if args.features == 'MS' else 0
                        outputs = outputs[:, -args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)
                        loss = criterion(outputs, batch_y)
                        train_loss.append(loss.item())
                else:
                    if args.output_attention:
                        outputs = model(batch_csv, batch_images, batch_y)[0]
                    else:
                        outputs = model(batch_csv, batch_images, batch_y)  # 将数据放入模型训练 16 * 600

                    f_dim = -1 if args.features == 'MS' else 0
                    # outputs = outputs[:, -args.pred_len:, f_dim:]
                    # batch_y = batch_y[:, -args.pred_len:, f_dim:]
                    # 计算模型损失
                    loss = criterion(outputs, batch_y.squeeze(-1))
                    train_loss.append(loss.item())
                    # 计算F1-score
                    predictions = torch.argmax(outputs, dim=1)  # 取最大概率类别
                    all_preds.append(predictions)
                    all_labels.append(batch_y)
                    # 计算acc
                    _, predictions = torch.max(outputs, 1)  # 获取每个样本的预测类别
                    correct += (predictions == batch_y.squeeze(-1)).sum().item()  # 计算正确预测的样本数量
                    # print('正确的个数', correct)

                if (i + 1) % 100 == 0:
                    accelerator.print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                    accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    accelerator.backward(loss)
                    model_optim.step()

                if args.lradj == 'TST':
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=False)
                    scheduler.step()

            accelerator.print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, vali_mae_loss, accuracy_val, f1_val = vali(args, accelerator, model, val_loader, val_loader,
                                                                  criterion, mae_metric)
            test_loss, test_mae_loss, accuracy_test, f1_test = vali(args, accelerator, model, test_loader, test_loader,
                                                                    criterion, mae_metric)
            print('data number:', args.batch_size * len(train_loader))
            accuracy_cha.append(correct / (args.batch_size * len(train_loader)))  # 准确率
            accuracy_val_cha.append(accuracy_val / (args.batch_size * len(val_loader)))  # 准确率
            accuracy_test_cha.append(accuracy_test / (args.batch_size * len(test_loader)))  # 准确率
            # F1-Score
            preds = torch.cat(all_preds, dim=0)
            labels = torch.cat(all_labels, dim=0)
            f1 = compute_f1_score(labels, preds)
            train_f1_score.append(f1)
            val_f1_score.append(f1_val)
            test_f1_score.append(f1_test)

            accelerator.print(
                "Epoch: {0} | Train Loss: {1:.7f} Vali Loss: {2:.7f} Test Loss: {3:.7f} Train accuracy: {4:.7f}".format(
                    epoch + 1, train_loss, vali_loss, test_loss, accuracy_cha[-1]))

            # early_stopping(vali_loss, model, path)
            # if early_stopping.early_stop:
            #     accelerator.print("Early stopping")
            #     break

            if args.lradj != 'TST':
                if args.lradj == 'COS':
                    scheduler.step()
                    accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                else:
                    if epoch == 0:
                        args.learning_rate = model_optim.param_groups[0]['lr']
                        accelerator.print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
                    adjust_learning_rate(accelerator, model_optim, scheduler, epoch + 1, args, printout=True)

            else:
                accelerator.print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))

        features, labels = get_features(model, test_loader, accelerator.device)

        # 可视化部分修改
        sample_size = 1000  # 预设采样量
        if len(features) == 0:
            raise ValueError("特征数组为空，请检查数据加载")

        actual_size = min(sample_size, len(features))
        if actual_size < sample_size:
            print(f"注意：实际采样量为{actual_size}（总样本数不足{sample_size}）")

        indices = np.random.choice(len(features), actual_size, replace=False)
        plot_tsne(features[indices], labels[indices], save_root)

        df = pd.DataFrame.from_dict({'f1': train_f1_score})
        df.to_csv(os.path.join(save_root, '/f1_train.csv'), index=False)
        df = pd.DataFrame.from_dict({'f1': val_f1_score})
        df.to_csv(os.path.join(save_root, '/f1_val.csv'), index=False)
        df = pd.DataFrame.from_dict({'f1': test_f1_score})
        df.to_csv(os.path.join(save_root, 'f1_test.csv'), index=False)

        df = pd.DataFrame.from_dict({'accuracy_train': accuracy_cha})
        df.to_csv(os.path.join(save_root, 'acc_train.csv'), index=False)
        df = pd.DataFrame.from_dict({'accuracy_val': accuracy_val_cha})
        df.to_csv(os.path.join(save_root, 'acc_val.csv'), index=False)
        df = pd.DataFrame.from_dict({'accuracy_test': accuracy_test_cha})
        df.to_csv(os.path.join(save_root, 'acc_test.csv'), index=False)

    accelerator.wait_for_everyone()
    if accelerator.is_local_main_process:
        path = './checkpoints'  # unique checkpoint saving path
        del_files(path)  # delete checkpoint files
        accelerator.print('success delete checkpoints')
