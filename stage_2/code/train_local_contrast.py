import argparse
import logging
import os
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn

import torch.optim as optim
from tensorboardX import SummaryWriter

from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader

from tqdm import tqdm


from dataloaders.dataset import (BaseDataSets, RandomGenerator,
                                 TwoStreamBatchSampler)
from networks.net_factory import net_factory
from utils import ramps

import torch.nn.functional as F
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str,
                    default='', help='path of dataset')
parser.add_argument('--exp', type=str,
                    default='', help='experiment_name')
parser.add_argument('--cpt_path', type=str,
                    default='', help='path of pretrained model')
parser.add_argument('--model', type=str,
                    default='unet_contrast', help='model_name')
parser.add_argument('--max_iterations', type=int,
                    default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=16,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[512,512],
                    help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=14,
                    help='output channel of network')
parser.add_argument('--inner_planes', type=int,  default=128,
                    help='output channel of feature embedding')
parser.add_argument('--queue_len', type=int,  default=64,
                    help='length of each queue')
parser.add_argument('--tempr', type=int,  default=0.2,
                    help='temperature')

# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=8,
                    help='labeled_batch_size per gpu')
parser.add_argument('--labeled_num', type=int, default=10,
                    help='labeled data')
# costs
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str,
                    default="mse", help='consistency_type')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
args = parser.parse_args()


def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"5": 0, "10": 0,
                    "15": 0, "20": 0, "75": 00}

    elif "Cervical" in dataset:
        ref_dict = {"3": 0, "6": 0, "9": 0}
    elif "flare" in dataset:
        ref_dict = {"5": 0, "10": 0, "15": 0}
    elif "WORD" in dataset:
        ref_dict = {"5": 0, "10": 0, "15": 0}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]





def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(alpha).add_(1 - alpha, param.data)



def label_onehot(inputs, num_class):
    '''
    inputs is class label
    return one_hot label
    dim will be increasee
    '''
    batch_size, image_h, image_w = inputs.shape
    inputs = torch.relu(inputs)
    outputs = torch.zeros([batch_size, num_class, image_h, image_w]).to(inputs.device)
    return outputs.scatter_(1, inputs.unsqueeze(1), 1.0)



class ContrastLoss(nn.Module):
    def __init__(self, num_classes=5, inner_planes=256, temperature=0.2, queue_len=64, args=None):
        super(ContrastLoss, self).__init__()

        self.temperature = temperature
        self.queue_len = queue_len
        self.num_classes = num_classes
        self.inner_planes=inner_planes


        for i in range(num_classes):
            self.register_buffer("queue" + str(i), torch.randn(self.inner_planes, self.queue_len))  
            self.register_buffer("ptr" + str(i), torch.zeros(1, dtype=torch.long))  
            exec("self.queue" + str(i) + '=' + 'nn.functional.normalize(' + "self.queue" + str(i) + ',dim=0)')

    def _dequeue_and_enqueue(self, keys, vals, cat, bs):
        if cat not in vals:
            return
        keys = keys[list(vals).index(cat)]
        # keys = Variable(keys,reqires_grad=False)
        batch_size = bs
        ptr = int(eval("self.ptr" + str(cat)))
        eval("self.queue" + str(cat))[:, ptr] = keys  # 入队了一个
        ptr = (ptr + batch_size) % self.queue_len
        eval("self.ptr" + str(cat))[0] = ptr

    def construct_region(self, fea, pred, gt=None):
        bs = fea.shape[0]


        # pred[pres < 0.7] = -1
        if gt is not None:
            pred = gt
            assert pred.shape[-1] == fea.shape[-1], "INVALID GT"
            pred = pred.view(bs, -1)
        else:
            pred = pred.detach()
            pres = F.softmax(pred, dim=1)
            # a = pres.sum(dim=1)
            pred = pred.max(1)[1].squeeze()  # 得到伪标签
        

            pred = pred.view(bs, -1)
            pres = pres.max(1)[0].view(bs, -1)
   
        val = torch.unique(pred)  # unique是去掉重复的元素，得到当前里面包含了哪些类别
        # val= torch.tensor([i for i in val if i>=0])

        fea = fea.squeeze()
        fea = fea.view(bs, self.inner_planes, -1).permute(1, 0, 2)  # BCHW -> BC(HW) -> CB(HW)

        # 下面的一系列操作应该就是对各类别的特征取平均

 

        new_fea = fea[:, pred == val[0]].mean(1).unsqueeze(0)  
        for i in val[1:]:

            class_fea = fea[:, pred == i].mean(1).unsqueeze(0)
            new_fea = torch.cat((new_fea, class_fea), dim=0)
        val = torch.tensor([i for i in val[1:] ])
        """
        new_fea = [[1,2,3],[4,5,6],[7,8,9]]
        val =       [0,       1,       2]
        """
        return new_fea, val

    def _compute_contrast_loss(self, l_pos, l_neg):
        N = l_pos.size(0)  # 1024
        logits = torch.cat((l_pos, l_neg), dim=1)  
        logits /= self.temperature
        labels = torch.zeros((N,), dtype=torch.long)
        return self.criterion(logits, labels)

    def forward(self, res1, fea1,  label_bs,gt=None, ema=False):

        contrast_loss_total = torch.tensor(0.).cuda()
        bs = res1.shape[0]
        if True:

 

            for ii in range(bs):
                fea = fea1[ii].unsqueeze(0)
                res = res1[ii].unsqueeze(0)
                if ii<label_bs:

                    gtt = gt[ii].unsqueeze(0)
                else:
                    gtt=None
                    # print(gtt.max())

                # print("fea",fea.shape,"res",res.shape)
                #     gtt=None

                keys, vals = self.construct_region(fea, res,
                                                   gtt)  # vals: N,  N is the category number in this batch
                keys = nn.functional.normalize(keys, dim=1)  # norm的目的应该是为了计算余弦相似度

                if not ema:
                    contrast_loss = 0
                    for cls_ind in range(self.num_classes): 
                        if cls_ind in vals:  # 如果这个batch里面有这个类
                            query = keys[list(vals).index(cls_ind)]  # 256   
                            lpos, lpos2 = query.unsqueeze(0), eval(
                                "self.queue" + str(cls_ind)).clone().detach()  # 1x256 256x512
                            l_pos = torch.mm(lpos,
                                             lpos2) / self.temperature  

                            exp_pos = torch.exp(l_pos)
                            # l_pos = l_pos.sum(1, keepdim=True)

                           
                            all_ind = [m for m in range(self.num_classes)]
                            # 下面应该是求负对相似度之和
                            l_neg = exp_pos.clone()
                            tmp = all_ind.copy()
                            tmp.remove(cls_ind)
                            for cls_ind2 in tmp:
                          
                                sim = torch.mm(query.unsqueeze(0), eval("self.queue" + str(cls_ind2)).clone().detach())
                                neg = torch.exp(sim / self.temperature).sum(1, keepdim=True)  # sum q,ki  1x1
                                l_neg += neg
                         
                            log_prob = (l_pos - torch.log(l_neg)).mean()  # 1x512,1x1 1x512
                            contrast_loss += (-log_prob)
                            # contrast_loss += self._compute_contrast_loss(l_pos, l_neg)
                        else:
                            continue
                    try:
                        contrast_loss_total += (contrast_loss / len(vals))
                    except ZeroDivisionError:
                        contrast_loss_total+=0.
                else:
                    for i in range(self.num_classes):
                        self._dequeue_and_enqueue(keys, vals, i, 1)
        return contrast_loss_total / bs

class DiceLoss(nn.Module):
    '''
    we rewrite dice loss for stable training
    '''
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []

        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        class_wise_dice = []
        loss = 0.0
        real_cls=0
        for i in range(0, self.n_classes):
            if (target[:,i]).mean()==0 and (inputs[:,i].mean())<0.001:
                continue
            real_cls+=1
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / real_cls

def train(args, snapshot_path):

    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    def create_model(ema=False):
        # Network definition
        model = net_factory(net_type=args.model, in_chns=1,
                            class_num=num_classes,inner_planes=args.inner_planes)
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    model = create_model()
    ema_model = create_model(ema=True)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)





    '''
    here is to load your dataset
    '''
    db_train = None




    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path, args.labeled_num)
    print("Total silices is: {}, labeled slices is: {}".format(
        total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size-args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=0, pin_memory=True, worker_init_fn=worker_init_fn)

    ###### load pretrained encoder ######

    pretrained = torch.load(args.cpt_path)
    save_model = pretrained["net"]
    model_dict = model.state_dict()
    # we only need to load the parameters of the encoder
    state_dict = {k: v for k, v in save_model.items() if "encoder" in k}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)


    model.train()


    optimizer = optim.SGD(model.parameters(), lr=base_lr,
                          momentum=0.9, weight_decay=0.0001)

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(num_classes)
    con_loss = ContrastLoss(args.num_classes,inner_planes=args.inner_planes,temperature=args.tempr,queue_len=args.queue_len).cuda()

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for epoch_num in iterator:
        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = F.interpolate(sampled_batch['image'],size=args.patch_size,mode='bilinear'), \
                F.interpolate(sampled_batch['label'].unsqueeze(1),size=args.patch_size,mode='nearest').squeeze(1)
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[args.labeled_bs:]

            noise1 = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.05, -0.1, 0.1)
            ema_inputs = unlabeled_volume_batch + noise1.cuda().detach()
            noise2 = torch.clamp(torch.randn_like(
                volume_batch) * 0.05, -0.1, 0.1)
            volume_batch = volume_batch + noise2.cuda().detach()


            outputs_logits, outputs_repre = model(volume_batch)
            outputs_soft = torch.softmax(outputs_logits, dim=1)

            with torch.no_grad():
                ema_output_logits, ema_output_repre = ema_model(ema_inputs)
                _,ema_out_lab_rep = ema_model(volume_batch[:args.labeled_bs])
                ema_output_soft = torch.softmax(ema_output_logits, dim=1)


            loss_ce = ce_loss(outputs_logits[:args.labeled_bs],
                              label_batch[:][:args.labeled_bs].long())

            loss_dice = dice_loss(
                outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss = (loss_dice + loss_ce)

            if iter_num>=5000:

                contrast_loss= con_loss(res1=outputs_logits,fea1 = outputs_repre,label_bs=args.labeled_bs,gt =label_batch[:args.labeled_bs].unsqueeze(1),ema=False )

            else:
                contrast_loss=torch.tensor(0).cuda()
            consistency_weight = get_current_consistency_weight(iter_num//100)

            if iter_num>=1000:
                model_consistency_loss  = torch.mean(
                    (outputs_soft[args.labeled_bs:] - ema_output_soft) ** 2)
            else:
                model_consistency_loss=0
            loss = supervised_loss + consistency_weight * (model_consistency_loss+contrast_loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            update_ema_variables(model, ema_model, args.ema_decay, iter_num)
            with torch.no_grad():
                _ = con_loss(res1=ema_output_logits, fea1=ema_output_repre, label_bs=args.labeled_bs,
                                         gt=label_batch[:args.labeled_bs].unsqueeze(1), ema=True)

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr_



            iter_num = iter_num + 1
            writer.add_scalar('info/lr', lr_, iter_num)
            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/loss_ce', loss_ce, iter_num)
            writer.add_scalar('info/loss_dice', loss_dice, iter_num)
            writer.add_scalar('info/contrast_loss',
                              contrast_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f, loss_contrast: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item(), contrast_loss.item()))


            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, 'iter_' + str(iter_num) + '.pth')
                torch.save(model.state_dict(), save_mode_path)
                logging.info("save model to {}".format(save_mode_path))

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()
    return "Training Finished!"


if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "../model/{}_{}_labeled/{}".format(
        args.exp, args.labeled_num, args.model)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
