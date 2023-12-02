import imp
# from multiprocessing import reduction
import torch
import torch.nn.functional as F
import torch.nn as nn
from IPython import embed


class LearnableSigmoid(nn.Module):
    def __init__(self, ):
        super(LearnableSigmoid, self).__init__()
        self.alpha_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.beta_weight = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.alpha_weight.data.fill_(1.0)
        self.beta_weight.data.fill_(0)

    def forward(self, input):
        return 1 / (1 + torch.exp(-self.alpha_weight * (input + self.beta_weight)))


class PLM_DE_Loss(nn.Module):
    """Negative sampling loss with self-adversarial training.

    Attributes:
        args: Some pre-set parameters, such as self-adversarial temperature, etc.
        model: The KG model for training.
    """

    def __init__(self, args, model, model_tea=None):
        super(PLM_DE_Loss, self).__init__()
        self.args = args
        self.model = model
        self.model_tea = model_tea
        self.distance = nn.SmoothL1Loss(reduction='none')
        self.sigmoid_pos_stu = LearnableSigmoid()
        self.sigmoid_neg_stu = LearnableSigmoid()
        self.sigmoid_pos_tea = LearnableSigmoid()
        self.sigmoid_neg_tea = LearnableSigmoid()
        self.kldivloss = nn.KLDivLoss(reduction="batchmean")
        self.mseloss = nn.MSELoss()
        self.celoss = nn.CrossEntropyLoss(reduction="elementwise_mean")

    def forward(self, pos_score, neg_score, pos_score_tea=None, neg_score_tea=None, score_of_lm=None, subsampling_weight=None):
        """
        Args:
            pos_score: The score of positive samples.
            neg_score: The score of negative samples.
            subsampling_weight: The weight for correcting pos_score and neg_score.

        Returns:
            loss: The training loss for back propagation.
        """

        d_score_pos = self.distance(torch.sigmoid(pos_score), torch.sigmoid(pos_score_tea))  # 512, 1
        d_score_neg = self.distance(torch.sigmoid(neg_score), torch.sigmoid(neg_score_tea))  # 512, 1024

        d_soft_pos = d_score_pos
        d_soft_neg = d_score_neg

        p_possoft_stu = self.sigmoid_pos_stu(pos_score_tea).squeeze(1)
        p_negsoft_stu = 1 - self.sigmoid_pos_stu(neg_score_tea)

        weight_neg_sampl = F.softmax(neg_score * self.args.adv_temp, dim=1).detach()
        d_soft_neg = (weight_neg_sampl * d_soft_neg).sum(dim=1)  # 负样本软标签
        p_negsoft_stu = (weight_neg_sampl * p_negsoft_stu).sum(dim=1)  # stu负样本软标签权重

        d_soft_pos = d_soft_pos.squeeze(1)

        Lsoft_stu = (subsampling_weight * p_possoft_stu * (d_soft_pos)).sum() / subsampling_weight.sum() + (
                subsampling_weight * p_negsoft_stu * (d_soft_neg)).sum() / subsampling_weight.sum()

        pos_score_basic = pos_score
        neg_score_basic = neg_score

        if self.args.negative_adversarial_sampling:
            neg_score_basic = (F.softmax(neg_score_basic * self.args.adv_temp, dim=1).detach()
                               * F.logsigmoid(-neg_score_basic)).sum(dim=1)  # shape:[bs]
        else:
            neg_score_basic = F.logsigmoid(-neg_score_basic).mean(dim=1)

        pos_score_basic = F.logsigmoid(pos_score_basic).view(neg_score_basic.shape[0])  # shape:[bs]

        if self.args.use_weight:
            positive_sample_loss_basic = - (subsampling_weight * pos_score_basic).sum() / subsampling_weight.sum()
            negative_sample_loss_basic = - (subsampling_weight * neg_score_basic).sum() / subsampling_weight.sum()
        else:
            positive_sample_loss_basic = - pos_score_basic.mean()
            negative_sample_loss_basic = - neg_score_basic.mean()

        loss_basic = (positive_sample_loss_basic + negative_sample_loss_basic) / 2
        distillation_loss_lm = self.mseloss(pos_score, score_of_lm) * 0.05
        loss_total = loss_basic + Lsoft_stu + distillation_loss_lm

        if self.args.model_name == 'ComplEx':
            # Use L3 regularization for ComplEx and DistMult
            regularization = self.args.regularization * (
                    self.model.ent_emb.weight.norm(p=3) ** 3 + self.model.rel_emb.weight.norm(p=3) ** 3
            )
            loss_total = loss_total + regularization

        return loss_total

    def normalize(self):
        """calculating the regularization.
        """
        regularization = self.args.regularization * (
                self.model.ent_emb.weight.norm(p=3) ** 3 + self.model.rel_emb.weight.norm(p=3) ** 3
        )
        return regularization
