import torch.nn as nn
import torch.nn.functional as F
from toolbox.losses.lovasz_losses import lovasz_softmax
import torch



def PCL(logits1, logits2, target_mask, temperature=1.):


    ce1 = F.cross_entropy(
        logits1.detach(),
        target_mask,
        reduction='none'
    ).unsqueeze(1)  # [B, 1, H, W]

    ce2 = F.cross_entropy(
        logits2.detach(),
        target_mask,
        reduction='none'
    ).unsqueeze(1)  # [B, 1, H, W]
    # Step 3: 计算权重因子（公式6）
    # total_ce = ce1 + ce2 + 1e-8  # 防止除零
    # weight1 = 1 - (ce1 / total_ce)  # S^t_hw
    # weight2 = 1 - weight1  # 1 - S^t_hw
    w1 = (ce1>ce2).float()
    w2 = (ce2>ce1).float()
    # w2 = torch.ones_like(ce2)
    # w2[ce2<=ce1] = 0
    # # Step 4: 生成混合logits（公式7）
    # mixed_logits = (
    #         weight1 * F.softmax(logits1,dim=1) +
    #         weight2 * F.softmax(logits2,dim=1)
    # ).detach()
    logits1= logits1 / temperature
    logits2 = logits2 / temperature
    # dice1 = DiceLoss()
    # dice2 = DiceLoss()
    loss1 = -(F.softmax(logits2.detach(),dim=1)*F.log_softmax(logits1,dim=1)).sum(dim=1,keepdim=True)
    loss2 = -(F.softmax(logits1.detach(),dim=1)*F.log_softmax(logits2,dim=1)).sum(dim=1,keepdim=True)
    demon1 = w1.sum()+1e-8
    demon2 = w2.sum()+1e-8
    loss1 = (loss1*w1).sum() / demon1
    loss2 = (loss2*w2).sum() / demon2

    return loss1,loss2
class KLDLoss(nn.Module):
    def __init__(self, alpha=1, tau=1):
        super().__init__()
        self.alpha_0 = alpha
        self.alpha = alpha
        self.tau = tau



        self.KLD = torch.nn.KLDivLoss(reduction='none')


    def forward(self, x_student, x_teacher):
        x_student = F.log_softmax(x_student / self.tau, dim=1)
        x_teacher = F.softmax(x_teacher / self.tau, dim=1)
        # print(x_student.shape)
        b,c,h,w = x_student.shape
        # numpix = b*h*w
        # loss = self.KLD(x_student, x_teacher) / numpix
        # print("self.alpha", self.alpha)
        loss = self.KLD(x_student,x_teacher).sum(1).mean()
        loss = self.alpha * loss
        return loss*(self.tau**2)



if __name__ == '__main__':
    kld = KLDLoss()
    x = torch.randn(4, 4, 256, 256)
    y = torch.randn(4, 4, 256, 256)
    print(kld(x, y))