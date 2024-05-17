import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from data_loader import Dataset_Custom
from model import master
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import numpy as np
import os
import time


class Pipeline(object):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.model = self._build_model()

    def _build_model(self):
        model = master.Model(self.args).float().to(self.device)

        return model

    def _acquire_device(self):
        if self.args.use_gpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.args.gpu)
            device = torch.device('cuda:{}'.format(self.args.gpu))
            print('Use GPU: cuda:{}'.format(self.args.gpu))
        else:
            device = torch.device('cpu')
            print('Use CPU')
        return device

    def _get_data(self, flag):
        args = self.args
        Data = Dataset_Custom
        if flag == 'test' or flag == 'val':
            shuffle_flag = False
            drop_last = True
            batch_size = args.batch_size
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args.batch_size
        data_set = Data(
            root_path=args.root_path,
            tweet_path=args.tweet_path,
            time_length=args.time_length,
            flag=flag,
            size=[args.seq_len, args.pred_len],
        )
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last
        )

        return data_set, data_loader

    # 选择优化器的函数
    def _select_optimizer(self):
        # 使用Adam优化器，优化模型的所有参数
        optimizer = optim.Adam(
            self.model.parameters(), lr=self.args.learning_rate, betas=(0.9, 0.95),
            weight_decay=self.args.weight_decay
        )
        return optimizer

    # 选择损失函数的函数
    def _select_criterion(self):
        # 使用均方误差损失函数
        criterion = nn.CrossEntropyLoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_tweet_data_x) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                for d in batch_tweet_data_x:
                    for k, v in d.items():
                        d[k] = v.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.target_size, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        tweet_pred, data_pred, cat_pred, loss_align = self.model(batch_x, batch_x_mark, dec_inp,
                                                                                 batch_y_mark, batch_tweet_data_x)
                else:
                    tweet_pred, data_pred, cat_pred, loss_align = self.model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark, batch_tweet_data_x)
                f_dim = -1 if self.args.features == 'MS' else 0
                y_out = batch_y.squeeze(dim=0)
                y_out = (y_out[:, -self.args.pred_len:, f_dim:].squeeze(dim=2) + 1).squeeze(dim=1).long().to(
                    self.device)
                outputs = self.args.alpha_tweet_loss * tweet_pred + self.args.alpha_data_loss * data_pred + self.args.alpha_cat_loss * cat_pred

                pred = outputs.detach().cpu()
                true = y_out.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        print("read data done.")

        if not os.path.exists(self.args.checkpoints):
            os.makedirs(self.args.checkpoints)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_tweet_data_x) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                for d in batch_tweet_data_x:
                    for k, v in d.items():
                        d[k] = v.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.target_size, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        tweet_pred, data_pred, cat_pred, loss_align = self.model(batch_x, batch_x_mark, dec_inp,
                                                                                 batch_y_mark, batch_tweet_data_x)

                        f_dim = -1 if self.args.features == 'MS' else 0
                        y_out = batch_y.squeeze(dim=0)
                        y_out = (y_out[:, -self.args.pred_len:, f_dim:].squeeze(dim=2) + 1).squeeze(dim=1).long().to(
                            self.device)
                        tweet_loss = criterion(tweet_pred, y_out)
                        data_loss = criterion(data_pred, y_out)
                        cat_loss = criterion(cat_pred, y_out)
                        loss = self.args.alpha_tweet_loss * tweet_loss + self.args.alpha_data_loss * data_loss + \
                               self.args.alpha_align_loss * loss_align + self.args.alpha_cat_loss * cat_loss
                        train_loss.append(loss.item())
                else:
                    tweet_pred, data_pred, cat_pred, align_loss = self.model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark, batch_tweet_data_x)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    y_out = batch_y.squeeze(dim=0)
                    y_out = (y_out[:, -self.args.pred_len:, f_dim:].squeeze(dim=2) + 1).squeeze(dim=1).long().to(
                        self.device)
                    tweet_loss = criterion(tweet_pred, y_out)
                    data_loss = criterion(data_pred, y_out)
                    cat_loss = criterion(cat_pred, y_out)
                    loss = self.args.alpha_tweet_loss * tweet_loss + self.args.alpha_data_loss * data_loss + \
                           self.args.alpha_align_loss * align_loss + self.args.alpha_cat_loss * cat_loss
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    print("\tdata_loss: {0:.7f}, tweet_loss: {1:.7f} | cat_loss: {2:.7f} | align_loss: {3:.7f}".format(
                        data_loss, tweet_loss, cat_loss, align_loss))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, self.args.checkpoints)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = self.args.checkpoints + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/', 'checkpoint.pth')))

        preds = []
        preds_logits = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark, batch_tweet_data_x) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                for d in batch_tweet_data_x:
                    for k, v in d.items():
                        d[k] = v.to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.target_size, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        tweet_pred, data_pred, cat_pred, align_loss = self.model(batch_x, batch_x_mark, dec_inp,
                                                                                 batch_y_mark, batch_tweet_data_x)
                else:
                    tweet_pred, data_pred, cat_pred, align_loss = self.model(batch_x, batch_x_mark, dec_inp,
                                                                             batch_y_mark, batch_tweet_data_x)

                f_dim = -1 if self.args.features == 'MS' else 0
                y_out = batch_y.squeeze(dim=0)
                y_out = (y_out[:, -self.args.pred_len:, f_dim:].squeeze(dim=2) + 1).squeeze(dim=1).long().to(
                    self.device)
                tweet_out = torch.softmax(tweet_pred, dim=-1)
                data_out = torch.softmax(data_pred, dim=-1)
                cat_out = torch.softmax(cat_pred, dim=-1)
                outputs = self.args.alpha_tweet_loss * tweet_out + self.args.alpha_data_loss * data_out + self.args.alpha_cat_loss * cat_out
                outputs_logits = torch.softmax(outputs, dim=-1)
                outputs = outputs.argmax(dim=-1)
                outputs_logits = outputs_logits.detach().cpu().numpy()
                outputs = outputs.detach().cpu().numpy()
                y_out = y_out.detach().cpu().numpy()

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = y_out  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)  # [stock_num]
                trues.append(true)
                preds_logits.append(outputs_logits)

        preds = np.array(preds)  # [time_step,stock_num]
        trues = np.array(trues)
        preds_logits = np.array(preds_logits)  # [time_step,stock_num,3]
        preds = preds.T
        trues = trues.T
        preds_logits = np.transpose(preds_logits, (1, 0, 2))
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        all_acc = []
        all_conf_matrix = []
        all_f1score = []
        all_auc = []
        f = open("result.txt", 'a')
        for i in range(len(preds)):
            pred = preds[i]
            true = trues[i]
            pred_logits = preds_logits[i]
            acc, conf_matrix, f1score, auc = metric(pred, true, pred_logits, folder_path, i)
            print('acc:{}, f1score:{}, auc:{}'.format(acc, f1score, auc))
            f.write('acc:{}, f1score:{}, auc:{}'.format(acc, f1score, auc))
            f.write('\n')
            f.write('\n')
            all_acc.append(acc)
            all_conf_matrix.append(conf_matrix)
            all_f1score.append(f1score)
            all_auc.append(auc)
        f.close()

        np.save(folder_path + 'acc.npy', all_acc)
        np.save(folder_path + 'conf_matrix.npy', all_conf_matrix)
        np.save(folder_path + 'f1score.npy', all_f1score)
        np.save(folder_path + 'auc.npy', all_auc)
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        preds = preds.flatten()
        trues = trues.flatten()
        preds_logits = np.reshape(preds_logits, (-1, preds_logits.shape[2]))  # 按第一维展平成二维
        overallacc, overallconf_matrix, overallf1score, overallauc = metric(preds, trues, preds_logits, folder_path)

        return overallacc, overallconf_matrix, overallf1score, overallauc