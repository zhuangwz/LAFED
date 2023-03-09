import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import json, argparse, random
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torchvision
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

import arguments, utils
from models.ensemble import Ensemble
from distillation import Linf_distillation, Linf_PGD


class Baseline_Trainer():
    def __init__(self, models, optimizers, schedulers,
                 trainloader, testloader,
                 writer, save_root=None, **kwargs):
        self.models = models
        self.epochs = kwargs['epochs']
        self.optimizers = optimizers
        self.schedulers = schedulers
        self.trainloader = trainloader
        self.testloader = testloader

        self.writer = writer
        self.save_root = save_root

        self.criterion = nn.CrossEntropyLoss()
        self.MSE = nn.MSELoss()
        self.depth = kwargs['depth']

        self.train_method = kwargs['train_method']
        self.distill_fixed_layer = kwargs['distill_fixed_layer']
        self.layer_list = kwargs['distill_layer']
        self.distill_cfg = {'eps': kwargs['distill_eps'],
                            'alpha': kwargs['distill_alpha'],
                            'steps': kwargs['distill_steps'],
                            # 'layer': kwargs['distill_layer'],
                            'before_relu': True,
                            'momentum': True
                            }

        self.plus_adv = kwargs['plus_adv']
        self.coeff = kwargs['dverge_coeff']
        if self.plus_adv:
            self.attack_cfg = {'eps': kwargs['eps'],
                               'alpha': kwargs['alpha'],
                               'steps': kwargs['steps'],
                               'is_targeted': False,
                               'rand_start': True
                               }

    def get_epoch_iterator(self):
        iterator = tqdm(list(range(1, self.epochs + 1)), total=self.epochs, desc='Epoch',
                        leave=False, position=1)
        return iterator

    def get_batch_iterator(self):
        iterator = tqdm(self.trainloader, desc='Batch', leave=False, position=2)
        return iterator

    def get_batch_iterator_distill(self):
        loader = utils.DistillationLoader(self.trainloader, self.trainloader)
        # loader = utils.DistillationLoader(self.testloader, self.testloader)
        iterator = tqdm(loader, desc='Batch', leave=False, position=2)
        return iterator

    def de_mean(self, x):
        x = x.view(-1)
        xmean = torch.mean(x)
        return torch.tensor([xi - xmean for xi in x])

    def cov(self, x, y):
        n = len(x)
        print(x.shape, torch.sum(x - y))
        return torch.dot(self.de_mean(x), self.de_mean(y)) / (n-1)

    def pcc(self, x, y):
        return self.cov(x, y) / (torch.std(x) * torch.std(y))

    def run(self):
        epoch_iter = self.get_epoch_iterator()
        for epoch in epoch_iter:
            self.train(epoch)
            # self.test(epoch)
            # self.save(epoch)
            break

    def train(self, epoch):
        if not self.distill_fixed_layer:
            tqdm.write('Randomly choosing a layer for distillation...')
            layers = random.randint(1, self.depth)
            self.layer_list = layers

        for i, m in enumerate(self.models):
            m.train()
            m.dropout_layer = 0
            # m.eval()

        losses = [0 for i in range(len(self.models))]

        img_num = 1000
        dis_mix_list = torch.zeros([img_num * 10, ])
        dis_mix_list2 = torch.zeros_like(dis_mix_list)
        dis_dverge_list = torch.zeros_like(dis_mix_list)
        dis_dverge_list2 = torch.zeros_like(dis_mix_list)
        dis_adp_list = torch.zeros_like(dis_mix_list)
        dis_adp_list2 = torch.zeros_like(dis_mix_list)

        dis_mix_list_m3 = torch.zeros([img_num * 10, ])
        dis_mix_list2_m3 = torch.zeros_like(dis_mix_list_m3)
        dis_dverge_list_m3 = torch.zeros_like(dis_mix_list_m3)
        dis_dverge_list2_m3 = torch.zeros_like(dis_mix_list_m3)
        dis_adp_list_m3 = torch.zeros_like(dis_mix_list_m3)
        dis_adp_list2_m3 = torch.zeros_like(dis_mix_list_m3)

        dis_mix_list_m5 = torch.zeros([img_num * 10, ])
        dis_mix_list2_m5 = torch.zeros_like(dis_mix_list_m5)
        dis_dverge_list_m5 = torch.zeros_like(dis_mix_list_m5)
        dis_dverge_list2_m5 = torch.zeros_like(dis_mix_list_m5)
        dis_adp_list_m5 = torch.zeros_like(dis_mix_list_m5)
        dis_adp_list2_m5 = torch.zeros_like(dis_mix_list_m5)

        batch_iter = self.get_batch_iterator() if self.train_method == 'none' else self.get_batch_iterator_distill()
        for batch_idx, in_data in enumerate(batch_iter):
            if batch_idx >= (img_num) // len(in_data[0]):
                break
            si, sl = in_data[0].cuda(), in_data[1].cuda()

            if self.plus_adv:
                adv_inputs_list = []

            if self.train_method != 'none':
                ti, tl = in_data[2].cuda(), in_data[3].cuda()

                feature_data_list = []
                for i, m in enumerate(self.models):
                    # init
                    m.dropout_layer = 0

                    # distill feature
                    temp = Linf_distillation(m, si, ti, layer=self.layer_list, **self.distill_cfg)
                    feature_data_list.append(temp)

                    # advt
                    if self.plus_adv:
                        temp = Linf_PGD(m, si, sl, **self.attack_cfg)
                        adv_inputs_list.append(temp)

            elif self.plus_adv:
                for i, m in enumerate(self.models):
                    temp = Linf_PGD(m, si, sl, **self.attack_cfg)
                    adv_inputs_list.append(temp)

            m1 = self.models[0]
            i = 0
            m1.dropout_layer = 0
            mix = 0

            # outputs_mix_i
            cur_feature_input_idx = random.randint(0, len(self.models) - 1)
            while cur_feature_input_idx == i:
                cur_feature_input_idx = random.randint(0, len(self.models) - 1)
            for j in range(len(self.models)):
                if i == j or cur_feature_input_idx == j:
                    continue
                temp_feature_output = m1.get_features(feature_data_list[j], self.layer_list, before_relu=False)
                mix += temp_feature_output
            m1.dropout_layer = self.layer_list
            m1.mixup_feature = mix / (len(self.models) - 2)
            outputs_mix_i = m1(feature_data_list[cur_feature_input_idx]).clone().detach()

            m1.dropout_layer = 0

            # outputs_dverge_i
            outputs_dverge_i = torch.zeros(size=(len(self.models) - 1,)+outputs_mix_i.shape)
            k = 0
            for j, distilled_data in enumerate(feature_data_list):
                if i == j:
                    continue
                outputs = m1(distilled_data)
                outputs_dverge_i[k] = outputs.clone().detach()
                k += 1

            # outputs_adp_i
            outputs_adp_i = m1(si).clone().detach()
            outputs_adp_i_m3 = m1(si).clone().detach()
            outputs_adp_i_m5 = m1(si).clone().detach()
            outputs_adp_j = m1(si).clone().detach()
            outputs_adp_j_m3 = m1(si).clone().detach()
            outputs_adp_j_m5 = m1(si).clone().detach()

            mix = 0

            # outputs_mix_i_m3
            cur_feature_input_idx = random.randint(0, 2)
            while cur_feature_input_idx == i:
                cur_feature_input_idx = random.randint(0, 2)
            for j in range(3):
                if i == j or cur_feature_input_idx == j:
                    continue
                temp_feature_output = m1.get_features(feature_data_list[j], self.layer_list, before_relu=False)
                mix += temp_feature_output
            m1.dropout_layer = self.layer_list
            m1.mixup_feature = mix / (3 - 2)
            outputs_mix_i_m3 = m1(feature_data_list[cur_feature_input_idx]).clone().detach()

            m1.dropout_layer = 0

            # outputs_dverge_i_m3
            outputs_dverge_i_m3 = torch.zeros(size=(2,) + outputs_mix_i_m3.shape)
            k = 0
            for j, distilled_data in enumerate(feature_data_list):
                if i == j:
                    continue
                if j >= 3:
                    break
                outputs = m1(distilled_data)
                outputs_dverge_i_m3[k] = outputs.clone().detach()
                k += 1

            mix = 0

            # outputs_mix_i_m5
            cur_feature_input_idx = random.randint(0, 4)
            while cur_feature_input_idx == i:
                cur_feature_input_idx = random.randint(0, 4)
            for j in range(5):
                if i == j or cur_feature_input_idx == j:
                    continue
                temp_feature_output = m1.get_features(feature_data_list[j], self.layer_list, before_relu=False)
                mix += temp_feature_output
            m1.dropout_layer = self.layer_list
            m1.mixup_feature = mix / (5 - 2)
            outputs_mix_i_m5 = m1(feature_data_list[cur_feature_input_idx]).clone().detach()

            m1.dropout_layer = 0

            # outputs_dverge_i_m5
            outputs_dverge_i_m5 = torch.zeros(size=(4,) + outputs_mix_i_m5.shape)
            k = 0
            for j, distilled_data in enumerate(feature_data_list):
                if i == j:
                    continue
                if j >= 5:
                    break
                outputs = m1(distilled_data)
                outputs_dverge_i_m5[k] = outputs.clone().detach()
                k += 1

            i = 1
            mix = 0

            # outputs_mix_j
            cur_feature_input_idx = random.randint(0, len(self.models) - 1)
            while cur_feature_input_idx == i:
                cur_feature_input_idx = random.randint(0, len(self.models) - 1)
            for j in range(len(self.models)):
                if i == j or cur_feature_input_idx == j:
                    continue
                temp_feature_output = m1.get_features(feature_data_list[j], self.layer_list, before_relu=False)
                mix += temp_feature_output
            m1.dropout_layer = self.layer_list
            m1.mixup_feature = mix / (len(self.models) - 2)
            outputs_mix_j = m1(feature_data_list[cur_feature_input_idx]).clone().detach()

            m1.dropout_layer = 0

            # outputs_dverge_j
            outputs_dverge_j = torch.zeros(size=(len(self.models) - 1,) + outputs_mix_j.shape)
            k = 0
            for j, distilled_data in enumerate(feature_data_list):
                if i == j:
                    continue
                outputs = m1(distilled_data)
                outputs_dverge_j[k] = outputs.clone().detach()
                k += 1

            mix = 0

            # outputs_mix_j_m3
            cur_feature_input_idx = random.randint(0, 2)
            while cur_feature_input_idx == i:
                cur_feature_input_idx = random.randint(0, 2)
            for j in range(3):
                if i == j or cur_feature_input_idx == j:
                    continue
                temp_feature_output = m1.get_features(feature_data_list[j], self.layer_list, before_relu=False)
                mix += temp_feature_output
            m1.dropout_layer = self.layer_list
            m1.mixup_feature = mix / (3 - 2)
            outputs_mix_j_m3 = m1(feature_data_list[cur_feature_input_idx]).clone().detach()

            m1.dropout_layer = 0

            # outputs_dverge_j_m3
            outputs_dverge_j_m3 = torch.zeros(size=(2,) + outputs_mix_j_m3.shape)
            k = 0
            for j, distilled_data in enumerate(feature_data_list):
                if i == j:
                    continue
                if j >= 3:
                    break
                outputs = m1(distilled_data)
                outputs_dverge_j_m3[k] = outputs.clone().detach()
                k += 1

            mix = 0

            # outputs_mix_j_m5
            cur_feature_input_idx = random.randint(0, 4)
            while cur_feature_input_idx == i:
                cur_feature_input_idx = random.randint(0, 4)
            for j in range(5):
                if i == j or cur_feature_input_idx == j:
                    continue
                temp_feature_output = m1.get_features(feature_data_list[j], self.layer_list, before_relu=False)
                mix += temp_feature_output
            m1.dropout_layer = self.layer_list
            m1.mixup_feature = mix / (5 - 2)
            outputs_mix_j_m5 = m1(feature_data_list[cur_feature_input_idx]).clone().detach()

            m1.dropout_layer = 0

            # outputs_dverge_j_m5
            outputs_dverge_j_m5 = torch.zeros(size=(4,) + outputs_mix_j_m5.shape)
            k = 0
            for j, distilled_data in enumerate(feature_data_list):
                if i == j:
                    continue
                if j >= 5:
                    break
                outputs = m1(distilled_data)
                outputs_dverge_j_m5[k] = outputs.clone().detach()
                k += 1

            # [model_num - 1, batch_size, 10] -> [batch_size, 10]
            outputs_dverge_i = torch.mean(outputs_dverge_i, 0)
            outputs_dverge_j = torch.mean(outputs_dverge_j, 0)

            outputs_dverge_i_m3 = torch.mean(outputs_dverge_i_m3, 0)
            outputs_dverge_j_m3 = torch.mean(outputs_dverge_j_m3, 0)

            outputs_dverge_i_m5 = torch.mean(outputs_dverge_i_m5, 0)
            outputs_dverge_j_m5 = torch.mean(outputs_dverge_j_m5, 0)

            for i in range(len(in_data[0])):
                basic = batch_idx * 10 * len(in_data[0]) + i * 10
                dis_mix_list[basic:basic + 10] = outputs_mix_i[i]
                dis_mix_list2[basic:basic + 10] = outputs_mix_j[i]
                dis_dverge_list[basic:basic + 10] = outputs_dverge_i[i]
                dis_dverge_list2[basic:basic + 10] = outputs_dverge_j[i]
                dis_adp_list[basic:basic + 10] = outputs_adp_i[i]
                dis_adp_list2[basic:basic + 10] = outputs_adp_j[i]

                dis_mix_list_m3[basic:basic + 10] = outputs_mix_i_m3[i]
                dis_mix_list2_m3[basic:basic + 10] = outputs_mix_j_m3[i]
                dis_dverge_list_m3[basic:basic + 10] = outputs_dverge_i_m3[i]
                dis_dverge_list2_m3[basic:basic + 10] = outputs_dverge_j_m3[i]
                dis_adp_list_m3[basic:basic + 10] = outputs_adp_i_m3[i]
                dis_adp_list2_m3[basic:basic + 10] = outputs_adp_j_m3[i]

                dis_mix_list_m5[basic:basic + 10] = outputs_mix_i_m5[i]
                dis_mix_list2_m5[basic:basic + 10] = outputs_mix_j_m5[i]
                dis_dverge_list_m5[basic:basic + 10] = outputs_dverge_i_m5[i]
                dis_dverge_list2_m5[basic:basic + 10] = outputs_dverge_j_m5[i]
                dis_adp_list_m5[basic:basic + 10] = outputs_adp_i_m5[i]
                dis_adp_list2_m5[basic:basic + 10] = outputs_adp_j_m5[i]

        # loss_dict = {}
        # for i in range(len(self.models)):
        #     loss_dict[str(i)] = losses[i] / len(self.trainloader)
        pcc_mix_m3 = self.pcc(dis_mix_list_m3, dis_mix_list2_m3)
        pcc_dverge_m3 = self.pcc(dis_dverge_list_m3, dis_dverge_list2_m3)
        pcc_adp_m3 = self.pcc(dis_adp_list_m3, dis_adp_list2_m3)
        print('m3: mixup/dverge/adp', pcc_mix_m3, pcc_dverge_m3, pcc_adp_m3)

        pcc_mix_m5 = self.pcc(dis_mix_list_m5, dis_mix_list2_m5)
        pcc_dverge_m5 = self.pcc(dis_dverge_list_m5, dis_dverge_list2_m5)
        pcc_adp_m5 = self.pcc(dis_adp_list_m5, dis_adp_list2_m5)
        print('m5: mixup/dverge/adp', pcc_mix_m5, pcc_dverge_m5, pcc_adp_m5)

        pcc_mix = self.pcc(dis_mix_list, dis_mix_list2)
        pcc_dverge = self.pcc(dis_dverge_list, dis_dverge_list2)
        pcc_adp = self.pcc(dis_adp_list, dis_adp_list2)
        print('m8: mixup/dverge/adp', pcc_mix, pcc_dverge, pcc_adp)

        # plt.figure(figsize=(10, 5))
        # plt.subplot(121)
        # plt.xlabel('L_1 of FME')
        # plt.ylabel('L_2 of FME')
        # plt.legend(title='PCC: {:.2f}'.format(pcc_mix.tolist()), loc='lower right')
        # plt.scatter(x=dis_mix_list.tolist(), y=dis_mix_list2.tolist(), s=10)
        # plt.subplot(122)
        # plt.xlabel('L_1 of DVERGE')
        # plt.ylabel('L_2 of DVERGE')
        # plt.legend(title='PCC: {:.2f}'.format(pcc_dverge.tolist()), loc='lower right')
        # plt.scatter(x=dis_dverge_list.tolist(), y=dis_dverge_list2.tolist(), s=10)
        # plt.show()
        plt.figure(figsize=(10, 10))
        plt.subplot(331)
        plt.legend(title='PCC: {:.2f}'.format(pcc_adp_m3.tolist()), loc='lower right')
        plt.scatter(x=dis_adp_list_m3.tolist(), y=dis_adp_list2_m3.tolist(), s=10)
        plt.subplot(334)
        # plt.xlabel('L_1 of DVERGE/3')
        # plt.ylabel('L_2 of DVERGE/3')
        plt.legend(title='PCC: {:.2f}'.format(pcc_dverge_m3.tolist()), loc='lower right')
        plt.scatter(x=dis_dverge_list_m3.tolist(), y=dis_dverge_list2_m3.tolist(), s=10)
        plt.subplot(337)
        # plt.xlabel('L_1 of FME/3')
        # plt.ylabel('L_2 of FME/3')
        plt.legend(title='PCC: {:.2f}'.format(pcc_mix_m3.tolist()), loc='lower right')
        plt.scatter(x=dis_mix_list_m3.tolist(), y=dis_mix_list2_m3.tolist(), s=10)
        plt.subplot(332)
        plt.legend(title='PCC: {:.2f}'.format(pcc_adp_m5.tolist()), loc='lower right')
        plt.scatter(x=dis_adp_list_m5.tolist(), y=dis_adp_list2_m5.tolist(), s=10)
        plt.subplot(335)
        # plt.xlabel('L_1 of DVERGE/5')
        # plt.ylabel('L_2 of DVERGE/5')
        plt.legend(title='PCC: {:.2f}'.format(pcc_dverge_m5.tolist()), loc='lower right')
        plt.scatter(x=dis_dverge_list_m5.tolist(), y=dis_dverge_list2_m5.tolist(), s=10)
        plt.subplot(338)
        # plt.xlabel('L_1 of FME/5')
        # plt.ylabel('L_2 of FME/5')
        plt.legend(title='PCC: {:.2f}'.format(pcc_mix_m5.tolist()), loc='lower right')
        plt.scatter(x=dis_mix_list_m5.tolist(), y=dis_mix_list2_m5.tolist(), s=10)
        plt.subplot(333)
        plt.legend(title='PCC: {:.2f}'.format(pcc_adp.tolist()), loc='lower right')
        plt.scatter(x=dis_adp_list.tolist(), y=dis_adp_list2.tolist(), s=10)
        plt.subplot(336)
        # plt.xlabel('L_1 of DVERGE/8')
        # plt.ylabel('L_2 of DVERGE/8')
        plt.legend(title='PCC: {:.2f}'.format(pcc_dverge.tolist()), loc='lower right')
        plt.scatter(x=dis_dverge_list.tolist(), y=dis_dverge_list2.tolist(), s=10)
        plt.subplot(339)
        # plt.xlabel('L_1 of FME/8')
        # plt.ylabel('L_2 of FME/8')
        plt.legend(title='PCC: {:.2f}'.format(pcc_mix.tolist()), loc='lower right')
        plt.scatter(x=dis_mix_list.tolist(), y=dis_mix_list2.tolist(), s=10)
        plt.show()

        plt.figure(figsize=(5, 5))
        plt.legend(title='PCC: {:.2f}'.format(pcc_adp_m3.tolist()), loc='lower right')
        plt.scatter(x=dis_adp_list_m3.tolist(), y=dis_adp_list2_m3.tolist(), s=10)
        plt.show()
        plt.figure(figsize=(5, 5))
        plt.legend(title='PCC: {:.2f}'.format(pcc_dverge_m3.tolist()), loc='lower right')
        plt.scatter(x=dis_dverge_list_m3.tolist(), y=dis_dverge_list2_m3.tolist(), s=10)
        plt.show()
        plt.figure(figsize=(5, 5))
        plt.legend(title='PCC: {:.2f}'.format(pcc_mix_m3.tolist()), loc='lower right')
        plt.scatter(x=dis_mix_list_m3.tolist(), y=dis_mix_list2_m3.tolist(), s=10)
        plt.show()
        plt.figure(figsize=(5, 5))
        plt.legend(title='PCC: {:.2f}'.format(pcc_dverge_m5.tolist()), loc='lower right')
        plt.scatter(x=dis_dverge_list_m5.tolist(), y=dis_dverge_list2_m5.tolist(), s=10)
        plt.show()
        plt.figure(figsize=(5, 5))
        plt.legend(title='PCC: {:.2f}'.format(pcc_mix_m5.tolist()), loc='lower right')
        plt.scatter(x=dis_mix_list_m5.tolist(), y=dis_mix_list2_m5.tolist(), s=10)
        plt.show()
        plt.figure(figsize=(5, 5))
        plt.legend(title='PCC: {:.2f}'.format(pcc_dverge.tolist()), loc='lower right')
        plt.scatter(x=dis_dverge_list.tolist(), y=dis_dverge_list2.tolist(), s=10)
        plt.show()
        plt.figure(figsize=(5, 5))
        plt.legend(title='PCC: {:.2f}'.format(pcc_mix.tolist()), loc='lower right')
        plt.scatter(x=dis_mix_list.tolist(), y=dis_mix_list2.tolist(), s=10)
        plt.show()



    def test(self, epoch):
        for m in self.models:
            m.dropout_layer = 0
            m.eval()

        ensemble = Ensemble(self.models)

        loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _, (inputs, targets) in enumerate(self.testloader):
                inputs, targets = inputs.cuda(), targets.cuda()

                outputs = ensemble(inputs)
                loss += self.criterion(outputs, targets).item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()

                total += inputs.size(0)

        self.writer.add_scalar('test/ensemble_loss', loss / len(self.testloader), epoch)
        self.writer.add_scalar('test/ensemble_acc', 100 * correct / total, epoch)

        print_message = 'Evaluation  | Ensemble Loss {loss:.4f} Acc {acc:.2%}'.format(
            loss=loss / len(self.testloader), acc=correct / total)
        tqdm.write(print_message)

    def save(self, epoch):
        return
        state_dict = {}
        for i, m in enumerate(self.models):
            state_dict['model_%d' % i] = m.state_dict()
        torch.save(state_dict, os.path.join(self.save_root, 'epoch_%d.pth' % epoch))


def get_args():
    parser = argparse.ArgumentParser(description='CIFAR10 Baseline Training of Ensemble', add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    arguments.base_train_args(parser)
    arguments.my_train_args(parser)
    args = parser.parse_args()
    return args


def main():
    # get args
    args = get_args()

    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    # set up writer, logger, and save directory for models
    save_root = os.path.join('../checkpoints', 'my', 'seed_{:d}'.format(args.seed),
                             '{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth),
                             # 'ablation3',
                             '{:s}_eps{:.2f}_norm_dis'.format(args.train_method, args.distill_eps))
    if args.distill_fixed_layer:
        save_root += '_fixed_layer_{:d}'.format(args.distill_layer)
    if args.plus_adv:
        save_root += '_adv_coeff{:.1f}'.format(args.dverge_coeff)

    if not os.path.exists(save_root):
        os.makedirs(save_root)
    else:
        print('*********************************')
        print('* The checkpoint already exists *')
        print('*********************************')

    writer = SummaryWriter(save_root.replace('checkpoints', 'runs'))

    # dump configurations for potential future references
    with open(os.path.join(save_root, 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)
    with open(os.path.join(save_root.replace('checkpoints', 'runs'), 'cfg.json'), 'w') as fp:
        json.dump(vars(args), fp, indent=4)

    # set up random seed
    torch.manual_seed(args.seed)

    # initialize models
    if args.start_from == 'baseline':
        args.model_file = os.path.join('checkpoints', 'baseline', 'seed_0', '{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth), 'epoch_200.pth')
        # args.model_file = os.path.join('checkpoints', 'my', 'seed_0', '{:d}_{:s}{:d}'.format(args.model_num, args.arch, args.depth), 'mixup1.00_dverge_eps0.07', 'epoch_200.pth')
    else:
        args.model_file = None
    models = utils.get_models(args, train=True, as_ensemble=False, model_file=args.model_file)

    # get data loaders
    trainloader, testloader = utils.get_loaders(args)

    # get optimizers and schedulers
    optimizers = utils.get_optimizers(args, models)
    schedulers = utils.get_schedulers(args, optimizers)

    # train the ensemble
    trainer = Baseline_Trainer(models, optimizers, schedulers,
                               trainloader, testloader, writer, save_root, **vars(args))
    trainer.run()


if __name__ == '__main__':
    main()
