import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler as st
import pandas as pd

plt.switch_backend('agg')


def plot_tsne(features, labels, title="t-SNE Visualization", save_path=r'E:\young\online\Time-LLM-young\Time-LLM-PU\results/'):
    """执行t-SNE并绘制结果"""
    # 数据标准化
    scaler = st()
    features_scaled = scaler.fit_transform(features)

    # 执行t-SNE降维
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_tsne = tsne.fit_transform(features_scaled)

    assert features_tsne.shape[1] == 2, "t-SNE输出维度错误"
    if labels.ndim > 1:
        labels = np.squeeze(labels)

    # 创建DataFrame
    df = pd.DataFrame({
        't-SNE1': features_tsne[:, 0].ravel(),
        't-SNE2': features_tsne[:, 1].flatten(),
        'Label': labels.astype(int)
    })

    # 保存为CSV
    df.to_csv(os.path.join(save_path, 'tsne.csv'), index=False)
    # print(f"t-SNE结果已保存至：{os.path.join(save_path, r'tsne.csv')}")

    # 绘制结果
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        features_tsne[:, 0],
        features_tsne[:, 1],
        c=labels,
        cmap="viridis",
        alpha=0.6,
        s=15  # 点的大小
    )

    # 添加颜色条和图例
    plt.colorbar(scatter, label="Class Label")
    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.savefig(os.path.join(save_path, 'tsne.png'))
    # plt.show()


def get_features(model, dataloader, device):
    features_list = []
    labels_list = []

    def hook(module, input, output):
        # 方法1：显式转换
        converted = input[0].detach().cpu().float()  # BFloat16 -> Float32
        features_list.append(converted.numpy())

    hook_handle = model.output_projection.linear.register_forward_hook(hook)

    model.eval()
    with torch.no_grad():
        for batch_csv, batch_images, batch_y in dataloader:
            batch_csv = batch_csv.to(device).to(torch.float32)  # 确保输入类型
            batch_images = batch_images.to(device).to(torch.float32)
            labels_list.append(batch_y.cpu().numpy())

            _ = model.classify(batch_csv, batch_images, batch_y)

    hook_handle.remove()

    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    return features_array, labels_array


def extract_features(model, dataloader, device):
    """提取分类层前的特征"""
    model.eval()
    features_list = []
    labels_list = []

    with torch.no_grad():
        for batch_csv, batch_images, batch_y in dataloader:
            batch_csv = batch_csv.to(device)
            batch_images = batch_images.to(device)
            features = model.classify(batch_csv, batch_images, None)
            features_list.append(features.cpu().numpy())
            labels_list.append(batch_y.numpy())

    return np.concatenate(features_list), np.concatenate(labels_list)


def compute_f1_score(y_true, y_pred):
    return f1_score(y_true.cpu().numpy(), y_pred.cpu().numpy(), average='macro')


def adjust_learning_rate(accelerator, optimizer, scheduler, epoch, args, printout=True):
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        lr_adjust = {epoch: args.learning_rate if epoch < 3 else args.learning_rate * (0.9 ** ((epoch - 3) // 1))}
    elif args.lradj == 'PEMS':
        lr_adjust = {epoch: args.learning_rate * (0.95 ** (epoch // 1))}
    elif args.lradj == 'TST':
        lr_adjust = {epoch: scheduler.get_last_lr()[0]}
    elif args.lradj == 'constant':
        lr_adjust = {epoch: args.learning_rate}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if printout:
            if accelerator is not None:
                accelerator.print('Updating learning rate to {}'.format(lr))
            else:
                print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, accelerator=None, patience=7, verbose=False, delta=0, save_mode=True):
        self.accelerator = accelerator
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_mode = save_mode

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.accelerator is None:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            else:
                self.accelerator.print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save_mode:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            if self.accelerator is not None:
                self.accelerator.print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            else:
                print(
                    f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')

        if self.accelerator is not None:
            model = self.accelerator.unwrap_model(model)
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        else:
            torch.save(model.state_dict(), path + '/' + 'checkpoint')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def adjustment(gt, pred):
    anomaly_state = False
    for i in range(len(gt)):
        if gt[i] == 1 and pred[i] == 1 and not anomaly_state:
            anomaly_state = True
            for j in range(i, 0, -1):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
            for j in range(i, len(gt)):
                if gt[j] == 0:
                    break
                else:
                    if pred[j] == 0:
                        pred[j] = 1
        elif gt[i] == 0:
            anomaly_state = False
        if anomaly_state:
            pred[i] = 1
    return gt, pred


def cal_accuracy(y_pred, y_true):
    return np.mean(y_pred == y_true)


def del_files(dir_path):
    shutil.rmtree(dir_path)


def vali(args, accelerator, model, vali_data, vali_loader, criterion, mae_metric):
    total_loss = []
    total_mae_loss = []
    correct = 0
    all_preds = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for i, (batch_csv, batch_images, batch_y) in tqdm(enumerate(vali_loader)):
            # batch_x = batch_x.float().to(accelerator.device)
            # batch_y = batch_y.long()

            # batch_x_mark = batch_x_mark.float().to(accelerator.device)
            # batch_y_mark = batch_y_mark.float().to(accelerator.device)

            # decoder input
            # dec_inp = torch.zeros_like(batch_y[:, -args.pred_len:, :]).float()
            # dec_inp = torch.cat([batch_y[:, :args.label_len, :], dec_inp], dim=1).float().to(
            #     accelerator.device)
            # encoder - decoder
            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_csv, batch_images, batch_y)[0]
                    else:
                        outputs = model(batch_csv, batch_images, batch_y)
            else:
                if args.output_attention:
                    outputs = model(batch_csv, batch_images, batch_y)[0]
                else:
                    outputs = model(batch_csv, batch_images, batch_y)

            outputs, batch_y = accelerator.gather_for_metrics((outputs, batch_y))

            f_dim = -1 if args.features == 'MS' else 0
            # outputs = outputs[:, -args.pred_len:, f_dim:]
            # batch_y = batch_y[:, -args.pred_len:, f_dim:].to(accelerator.device)

            # pred = outputs.detach()
            # true = batch_y.detach()
            predictions = torch.argmax(outputs, dim=1)  # 取最大概率类别

            all_preds.append(predictions)
            all_labels.append(batch_y)

            # 计算 F1-Score

            pred = outputs
            true = batch_y.squeeze(-1)

            loss = criterion(pred, true)
            _, predictions = torch.max(outputs, 1)  # 获取每个样本的预测类别
            correct += (predictions == batch_y.squeeze(-1)).sum().item()  # 计算正确预测的样本数量
            # mae_loss = mae_metric(pred, true)

            total_loss.append(loss.item())
            # total_mae_loss.append(mae_loss.item())
    preds_ten = torch.cat(all_preds, dim=0)
    labels_ten = torch.cat(all_labels, dim=0)
    f1 = compute_f1_score(labels_ten, preds_ten)
    total_loss = np.average(total_loss)
    # total_mae_loss = np.average(total_mae_loss)

    model.train()
    return total_loss, total_mae_loss, correct, f1


def test(args, accelerator, model, train_loader, vali_loader, criterion):
    x, _ = train_loader.dataset.last_insample_window()
    y = vali_loader.dataset.timeseries
    x = torch.tensor(x, dtype=torch.float32).to(accelerator.device)
    x = x.unsqueeze(-1)

    model.eval()
    with torch.no_grad():
        B, _, C = x.shape
        dec_inp = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        dec_inp = torch.cat([x[:, -args.label_len:, :], dec_inp], dim=1)
        outputs = torch.zeros((B, args.pred_len, C)).float().to(accelerator.device)
        id_list = np.arange(0, B, args.eval_batch_size)
        id_list = np.append(id_list, B)
        for i in range(len(id_list) - 1):
            outputs[id_list[i]:id_list[i + 1], :, :] = model(
                x[id_list[i]:id_list[i + 1]],
                None,
                dec_inp[id_list[i]:id_list[i + 1]],
                None
            )
        accelerator.wait_for_everyone()
        outputs = accelerator.gather_for_metrics(outputs)
        f_dim = -1 if args.features == 'MS' else 0
        outputs = outputs[:, -args.pred_len:, f_dim:]
        pred = outputs
        true = torch.from_numpy(np.array(y)).to(accelerator.device)
        batch_y_mark = torch.ones(true.shape).to(accelerator.device)
        true = accelerator.gather_for_metrics(true)
        batch_y_mark = accelerator.gather_for_metrics(batch_y_mark)

        loss = criterion(x[:, :, 0], args.frequency_map, pred[:, :, 0], true, batch_y_mark)

    model.train()
    return loss


def load_content(args):
    if 'ETT' in args.data:
        file = 'ETT'
    else:
        file = args.data
    with open('./dataset/prompt_bank/{0}.txt'.format(file), 'r', encoding='utf-8') as f:
        content = f.read()
    return content
