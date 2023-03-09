import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
sys.path.append(os.getcwd())
import argparse
from tqdm import tqdm
import pandas as pd

import numpy as np
import random

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets
from torch.utils.data import TensorDataset, DataLoader

from advertorch.attacks import GradientSignAttack, LinfBasicIterativeAttack, LinfPGDAttack
from advertorch.attacks.utils import multiple_mini_batch_attack, attack_whole_dataset
from advertorch.utils import to_one_hot

import arguments, utils
from models.ensemble import Ensemble


class CarliniWagnerLoss(nn.Module):
    """
    Carlini-Wagner Loss: objective function #6.
    Paper: https://arxiv.org/pdf/1608.04644.pdf
    """

    def __init__(self, conf=50.):
        super(CarliniWagnerLoss, self).__init__()
        self.conf = conf

    def forward(self, input, target):
        """
        :param input: pre-softmax/logits.
        :param target: true labels.
        :return: CW loss value.
        """
        num_classes = input.size(1)
        label_mask = to_one_hot(target, num_classes=num_classes).float()
        correct_logit = torch.sum(label_mask * input, dim=1)
        wrong_logit = torch.max((1. - label_mask) * input, dim=1)[0]
        loss = -F.relu(correct_logit - wrong_logit + self.conf).sum()
        return loss


def test(model, datafolder, return_acc=False):
    inputs = torch.load(os.path.join(datafolder, 'inputs.pt')).cpu()
    labels = torch.load(os.path.join(datafolder, 'labels.pt')).cpu()

    testset = TensorDataset(inputs, labels)
    testloader = DataLoader(testset, batch_size=100, shuffle=False, num_workers=4)

    correct = []
    with torch.no_grad():
        for (x, y) in testloader:
            x, y = x.cuda(), y.cuda()
            outputs = model(x)
            _, preds = outputs.max(1)
            correct.append(preds.eq(y))
    correct = torch.cat(correct, dim=0)
    if return_acc:
        return 100. * correct.sum().item() / len(testset)
    else:
        return correct


def get_args():
    parser = argparse.ArgumentParser(description='Evaluation of White-box Transfer Robustness of Ensembles',
                                     add_help=True)
    arguments.model_args(parser)
    arguments.data_args(parser)
    # arguments.bbox_eval_args(parser)
    arguments.wbox_eval_args(parser)
    args = parser.parse_args()
    return args


def main(args, eps_list):
    # load models
    if 'gal' in args.model_file:
        leaky_relu = True
    else:
        leaky_relu = False
    ensemble = utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)

    train_seed = args.model_file.split('/')[-3]
    train_alg = args.model_file.split('/')[-4]

    # eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    loss_fn_list = ['xent', 'cw']
    surrogate_model_list = ['{:s}{:d}'.format(args.which_ensemble, i) for i in [3, 5, 8]]
    method_list = ['mdi2_0.5_steps_{:d}'.format(args.steps), 'sgm_0.2_steps_{:d}'.format(args.steps)]
    index_list = ['{:s}_{:s}_mpgd'.format(a, b) for a in surrogate_model_list for b in loss_fn_list]
    index_list += ['{:s}_{:s}_{:s}'.format(a, b, c) for a in surrogate_model_list for b in loss_fn_list for c in
                   method_list]
    index_list.append('all')

    random_start = 3
    input_list = ['from_{:s}_{:s}_mpgd_steps_{:d}'.format(a, b, args.steps) for a in surrogate_model_list for b in
                  loss_fn_list]
    input_list += ['from_{:s}_{:s}_{:s}'.format(a, b, c) for a in surrogate_model_list for b in loss_fn_list for c in
                   method_list]

    rob = {}
    rob['source'] = index_list
    acc_list = [[] for _ in range(len(eps_list))]

    data_root = os.path.join(args.data_dir, args.folder)

    # clean acc
    input_folder = os.path.join(data_root, 'clean')
    clean_acc = test(ensemble, input_folder, return_acc=True)
    clean_acc_list = [clean_acc for _ in range(len(input_list) + 1)]
    rob['clean'] = clean_acc_list

    r_l = [clean_acc_list[-1]]

    # transfer attacks    
    for i, eps in enumerate(tqdm(eps_list, desc='eps', leave=True, position=0)):
        input_folder = os.path.join(data_root, 'eps_{:.2f}'.format(eps))
        correct_over_input = []

        for input_adv in tqdm(input_list, desc='source', leave=False, position=1):
            if 'mpgd' in input_adv:
                correct_over_rs = []

                for rs in tqdm(range(random_start), desc='Random Start', leave=False, position=2):
                    datafolder = os.path.join(input_folder, '_'.join((input_adv, str(rs))))
                    correct_over_rs.append(test(ensemble, datafolder))

                correct_over_rs = torch.stack(correct_over_rs, dim=-1).all(dim=-1)
                acc_list[i].append(100. * correct_over_rs.sum().item() / len(correct_over_rs))
                correct_over_input.append(correct_over_rs)

                # tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer from {:s} acc: {:.2f}%'.format(
                #     clean_acc, eps, input_adv, 100.*correct_over_rs.sum().item()/len(correct_over_rs)
                # ))
            else:
                datafolder = os.path.join(input_folder, input_adv)
                correct = test(ensemble, datafolder)
                acc_list[i].append(100. * correct.sum().item() / len(correct))
                correct_over_input.append(correct)

                # tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer from {:s} acc: {:.2f}%'.format(
                #     clean_acc, eps, input_adv, 100.*correct.sum().item()/len(correct)
                # ))

        correct_over_input = torch.stack(correct_over_input, dim=-1).all(dim=-1)
        acc_list[i].append(100. * correct_over_input.sum().item() / len(correct_over_input))

        tqdm.write('Clean acc: {:.2f}%, eps: {:.2f}, transfer acc: {:.2f}%'.format(
            clean_acc, eps, 100. * correct_over_input.sum().item() / len(correct_over_input)
        ))

        rob[str(eps)] = acc_list[i]

        r_l.append(acc_list[i][-1])

    return r_l


def wbox(args, eps_list):
    # load models
    if 'gal' in args.model_file:
        leaky_relu = True
    else:
        leaky_relu = False
    ensemble = utils.get_models(args, train=False, as_ensemble=True, model_file=args.model_file, leaky_relu=leaky_relu)

    # get data loaders
    total_sample_num = 10000
    if args.subset_num:
        random.seed(0)
        subset_idx = random.sample(range(total_sample_num), args.subset_num)
        testloader = utils.get_testloader(args, batch_size=200, shuffle=False, subset_idx=subset_idx)
    else:
        testloader = utils.get_testloader(args, batch_size=200, shuffle=False)

    loss_fn = nn.CrossEntropyLoss() if args.loss_fn == 'xent' else CarliniWagnerLoss(conf=args.cw_conf)

    rob = {}
    rob['sample_num'] = args.subset_num if args.subset_num else total_sample_num
    rob['loss_fn'] = 'xent' if args.loss_fn == 'xent' else 'cw_{:.1f}'.format(args.cw_conf)

    train_seed = args.model_file.split('/')[-3]
    train_alg = args.model_file.split('/')[-4]

    if args.convergence_check:
        eps = 0.01
        steps_list = [50, 500, 1000]
        random_start = 1

        rob['random_start'] = random_start
        rob['eps'] = eps

        # FGSM
        test_iter = tqdm(testloader, desc='FGSM', leave=False, position=0)
        adversary = GradientSignAttack(
            ensemble, loss_fn=nn.CrossEntropyLoss(), eps=eps,
            clip_min=0., clip_max=1., targeted=False)

        _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")
        print("Accuracy: {:.2f}%, FGSM Accuracy: {:.2f}%".format(
            100. * (label == pred).sum().item() / len(label),
            100. * (label == advpred).sum().item() / len(label)))
        rob['clean'] = 100. * (label == pred).sum().item() / len(label)
        rob['fgsm'] = 100. * (label == advpred).sum().item() / len(label)

        for steps in tqdm(steps_list, desc='PGD steps', leave=False, position=0):
            correct_or_not = []

            for i in tqdm(range(random_start), desc='Random Start', leave=False, position=1):
                torch.manual_seed(i)
                test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

                adversary = LinfPGDAttack(
                    ensemble, loss_fn=loss_fn, eps=eps,
                    nb_iter=steps, eps_iter=eps / 5, rand_init=True, clip_min=0., clip_max=1.,
                    targeted=False)

                _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")
                correct_or_not.append(label == advpred)

            correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)

            tqdm.write("Accuracy: {:.2f}%, steps: {:d}, PGD Accuracy: {:.2f}%".format(
                100. * (label == pred).sum().item() / len(label),
                steps,
                100. * correct_or_not.sum().item() / len(label)))

            rob[str(steps)] = 100. * correct_or_not.sum().item() / len(label)

        # save to file
        if args.save_to_csv:
            output_root = os.path.join('results', 'wbox', train_alg, train_seed, 'convergence_check')
            if not os.path.exists(output_root):
                os.makedirs(output_root)
            output_filename = args.model_file.split('/')[-2]
            output = os.path.join(output_root, '.'.join((output_filename, 'csv')))

            df = pd.DataFrame(rob, index=[0])
            if args.append_out and os.path.isfile(output):
                with open(output, 'a') as f:
                    f.write('\n')
                df.to_csv(output, sep=',', mode='a', header=False, index=False, float_format='%.2f')
            else:
                df.to_csv(output, sep=',', index=False, float_format='%.2f')
    else:
        # eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
        # eps_list = eps

        rob['random_start'] = args.random_start
        rob['steps'] = args.steps

        for eps in tqdm(eps_list, desc='PGD eps', leave=True, position=0):
            correct_or_not = []

            for i in tqdm(range(args.random_start), desc='Random Start', leave=False, position=1):
                torch.manual_seed(i)
                test_iter = tqdm(testloader, desc='Batch', leave=False, position=2)

                adversary = LinfPGDAttack(
                    ensemble, loss_fn=loss_fn, eps=eps,
                    nb_iter=args.steps, eps_iter=eps / 5, rand_init=True, clip_min=0., clip_max=1.,
                    targeted=False)

                _, label, pred, advpred = attack_whole_dataset(adversary, test_iter, device="cuda")

                correct_or_not.append(label == advpred)

            correct_or_not = torch.stack(correct_or_not, dim=-1).all(dim=-1)

            tqdm.write("Accuracy: {:.2f}%, eps: {:.2f}, PGD Accuracy: {:.2f}%".format(
                100. * (label == pred).sum().item() / len(label),
                eps,
                100. * correct_or_not.sum().item() / len(label)))

            rob['clean'] = 100. * (label == pred).sum().item() / len(label)
            rob[str(eps)] = 100. * correct_or_not.sum().item() / len(label)

    r_l = [args.random_start, args.steps]
    r_l.append(rob['clean'])
    for i in eps_list:
        r_l.append(rob[str(i)])
    return r_l


if __name__ == '__main__':
    # get args
    args = get_args()
    # set up gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    assert torch.cuda.is_available()

    train_seed = args.model_file.split('/')[-2]
    train_alg = args.model_file.split('/')[-3]
    output_filename = args.model_file.split('/')[-1]

    mf = args.model_file
    rob = []
    epc = []
    eps_list = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    for i in range(190, 200):
        args.model_file = mf + '/epoch_' + str(i + 1) + '.pth'
        # rob.append(main(args, eps_list))
        rob.append(wbox(args, eps_list))
        epc.append(i + 1)
        # print(rob)
    rob = np.transpose(np.array(rob), (1, 0))

    # **** bbox **** #
    # res = {'epoch': epc, 'clean': rob[0]}
    # for i in range(len(eps_list)):
    #     res[str(eps_list[i])] = rob[i + 1]
    # **** wbox **** #
    res = {'epoch': epc, 'random_start': rob[0], 'steps': rob[1], 'clean': rob[2]}
    for i in range(len(eps_list)):
        res[str(eps_list[i])] = rob[i + 3]

    # save to file
    if args.save_to_csv:
        output_root = os.path.join('results', 'epoch', train_alg, train_seed)

        if not os.path.exists(output_root):
            os.makedirs(output_root)
        output = os.path.join(output_root, '.'.join((output_filename + '_wbox', 'csv')))

        print(output)

        df = pd.DataFrame(res)
        if args.append_out and os.path.isfile(output):
            with open(output, 'a') as f:
                f.write('\n')
            df.to_csv(output, sep=',', mode='a', header=False, index=False, float_format='%.2f')
        else:
            df.to_csv(output, sep=',', index=False, float_format='%.2f')
