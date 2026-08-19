import argparse
from gc import collect
import pandas as pd
import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from torch import nn
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import Classification
import time
import random
import numpy as np
import os
from utils import my_read_data
from utils.tools import adjust_learning_rate, vali, compute_f1_score, plot_tsne, get_features

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"


if __name__ == '__main__':

    start_time = time.time()
    parser = argparse.ArgumentParser(description='FD-MVLLM')

    fix_seed = 2023
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)
    parser.add_argument('--seed', type=int, default=2025, help='random seed')
    # data loader
    parser.add_argument('--num_class', type=int, default=4, help='number of class')
    parser.add_argument('--sampling_rate', type=int, default=12000, help='sampling rate')
    parser.add_argument('--overlap', type=int, default=512, help='overlap of sliding window')
    parser.add_argument('--window_size', type=int, default=1024, help='size of sliding window')
    parser.add_argument('--csv_root', type=str, default=r'E:\dayoung\JNU-data\input1-10', help='csv file root')
    parser.add_argument('--image_root', type=str, default=r"E:\dayoung\JNU-data\img1_1024_128-3",
                        help='iamge file root')
    # model define
    parser.add_argument('--enc_in', type=int, default=1024, help='encoder input size csv')
    parser.add_argument('--enc_out', type=int, default=1024, help='encoder output size picture')
    parser.add_argument('--d_model', type=int, default=64, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')

    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--patch_len', type=int, default=32, help='patch length')
    parser.add_argument('--stride', type=int, default=32, help='stride')
    parser.add_argument('--llm_model_root', type=str, default=r'E:\dayoung\DeepSeek-R1-Distill-Llama-8B', help='LLM model root')  # LLAMA, GPT2, BERT, deepseek
    parser.add_argument('--llm_model', type=str, default='deepseek', help='LLM model')  # LLAMA, GPT2, BERT, deepseek
    parser.add_argument('--llm_dim', type=int, default='1024',
                        help='LLM model dimension')  # LLama7b:4096; GPT2-small:768; BERT-base:768; deepseek-llama:4096; deepseek-qwen:3584
    parser.add_argument('--llm_lora', type=bool, default=True, help='LLM lora')

    # optimization
    parser.add_argument('--train_epochs', type=int, default=2, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='TST', help='type1 adjust learning rate')
    parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
    parser.add_argument('--llm_layers', type=int, default=2)

    args = parser.parse_args()

    root_file = os.getcwd()
    save_file = f'results {os.path.basename(args.llm_model_root)} epo{args.train_epochs} without pic'
    save_root = os.path.join(root_file, save_file)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

    data_csv, image_data, labels = my_read_data.read_data(args.csv_root, args.image_root, args.window_size, args.overlap)

    train_loader, test_loader, val_loader = my_read_data.data_indices(data_csv, image_data, labels, args)

    model = Classification.Model(args).to(torch.bfloat16)

    time_now = time.time()

    train_steps = len(train_loader)

    trained_parameters = []
    for p in model.parameters():
        if p.requires_grad is True:
            trained_parameters.append(p)

    # model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
    model_optim = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01  # 避免过大的权重衰减
    )
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
            iter_count += 1

            model_optim.zero_grad()

            batch_csv = batch_csv.float().to(accelerator.device)
            batch_images = batch_images.float().to(accelerator.device)
            batch_y = batch_y.long().to(accelerator.device)

            outputs = model(batch_csv, batch_images, batch_y)

            # loss
            loss = criterion(outputs, batch_y.squeeze(-1))
            train_loss.append(loss.item())
            # F1-score
            predictions = torch.argmax(outputs, dim=1)  # 取最大概率类别
            all_preds.append(predictions)
            all_labels.append(batch_y)
            # acc
            _, predictions = torch.max(outputs, 1)  # 获取每个样本的预测类别
            correct += (predictions == batch_y.squeeze(-1)).sum().item()  # 计算正确预测的样本数量

            if (i + 1) % 100 == 0:
                accelerator.print(
                    "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
                accelerator.print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()

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

    sample_size = 1000  # 预设采样量
    if len(features) == 0:
        raise ValueError("特征数组为空，请检查数据加载")

    actual_size = min(sample_size, len(features))
    if actual_size < sample_size:
        print(f"注意：实际采样量为{actual_size}（总样本数不足{sample_size}）")

    indices = np.random.choice(len(features), actual_size, replace=False)
    plot_tsne(features[indices], labels[indices], save_path=save_root)

    df = pd.DataFrame.from_dict({'f1': train_f1_score})
    df.to_csv(os.path.join(save_root, 'f1_train.csv'), index=False)
    df = pd.DataFrame.from_dict({'f1': val_f1_score})
    df.to_csv(os.path.join(save_root, 'f1_val.csv'), index=False)
    df = pd.DataFrame.from_dict({'f1': test_f1_score})
    df.to_csv(os.path.join(save_root, 'f1_test.csv'), index=False)

    df = pd.DataFrame.from_dict({'accuracy_train': accuracy_cha})
    df.to_csv(os.path.join(save_root, 'acc_train.csv'), index=False)
    df = pd.DataFrame.from_dict({'accuracy_val': accuracy_val_cha})
    df.to_csv(os.path.join(save_root, 'acc_val.csv'), index=False)
    df = pd.DataFrame.from_dict({'accuracy_test': accuracy_test_cha})
    df.to_csv(os.path.join(save_root, 'acc_test.csv'), index=False)

    accelerator.wait_for_everyone()
    end_time = time.time()
    print(f'结束时间是：{time.ctime()}\n总用时：{(end_time - start_time) / 3600} 小时')

    # 训练结束后释放资源
    torch.cuda.empty_cache()
    collect()
