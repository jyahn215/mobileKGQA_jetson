import torch
from torch.nn import functional as F
from tqdm import tqdm
import torch.nn as nn
        

class InfoNCELoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.temperature = args.temp
    
    def forward(self, z_i, z_j):
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2) / self.temperature
        
        pos_sim = torch.diag(sim_matrix, batch_size)
        pos_sim = torch.cat([pos_sim, torch.diag(sim_matrix, -batch_size)])
        
        mask = torch.cat([torch.arange(batch_size) for _ in range(2)], dim=0)
        mask = (mask.unsqueeze(dim=0) == mask.unsqueeze(dim=1)).bool()
        mask = mask.to(z.device)

        neg_sim = sim_matrix[~mask].view(2 * batch_size, -1)
        
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        labels = torch.zeros(2 * batch_size).long().to(z.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
    
class TripletLoss(nn.Module): 
    def __init__(self, args):
        super().__init__()
        self.margin = args.margin
        self.p = args.p
    
    def forward(self, anchor, positive, negative):
        pos_score = torch.norm(anchor - positive, dim=2, p=self.p).mean(dim=1)
        neg_score = torch.norm(anchor - negative, dim=2, p=self.p).mean(dim=1)
        loss = F.relu(pos_score - neg_score + self.margin)
        return loss.mean(dim=0)
    
class NSSALoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.margin = args.margin
        self.p = args.p

    def forward(self, anchor, positive, negative):
        pos_score = torch.norm(anchor - positive, dim=2, p=self.p)
        pos_score = F.logsigmoid(self.margin - pos_score)
        pos_score = pos_score.mean(dim=1)
        pos_score = - pos_score.mean(dim=0)

        neg_score = torch.norm(anchor - negative, dim=2, p=self.p)
        neg_score = F.logsigmoid(neg_score - self.margin)
        neg_score = neg_score.mean(dim=1)
        neg_score = - neg_score.mean(dim=0)

        return pos_score + neg_score
    
class LogRatioLoss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.eps = args.eps
        # mask = torch.arange(batch_size)
        # self.upper_tri = (mask.unsqueeze(0) > mask.unsqueeze(1)).cuda()
        # mask = torch.arange(batch_size*(batch_size-1) // 2)
        # self.diag = (mask.unsqueeze(0) == mask.unsqueeze(1)).cuda()

    def forward(self, rep, true_cossim):
        mask = torch.arange(rep.size(0))
        upper_tri = (mask.unsqueeze(0) > mask.unsqueeze(1)).cuda()
        mask = torch.arange(rep.size(0)*(rep.size(0)-1) // 2)
        diag = (mask.unsqueeze(0) == mask.unsqueeze(1)).cuda()

        train_cossim = F.cosine_similarity(rep.unsqueeze(1), rep.unsqueeze(0), dim=2)
        train_cossim = train_cossim[upper_tri==True]
        train_phase_diff = torch.acos(train_cossim)
        train_phase_diff[train_phase_diff < self.eps] = self.eps
        train_phase_ratio = torch.divide(train_phase_diff.unsqueeze(dim=0),
                                         train_phase_diff.unsqueeze(dim=1))
        train_phase_ratio = train_phase_ratio[diag==False]
        train_phase_ratio[train_phase_ratio < self.eps] = self.eps
        train_phase_ratio = torch.log(train_phase_ratio)
        
        true_cossim = true_cossim[upper_tri==True]
        true_phase_diff = torch.acos(true_cossim)
        true_phase_diff[true_phase_diff < self.eps] = self.eps
        true_phase_ratio = torch.divide(true_phase_diff.unsqueeze(dim=0),
                                        true_phase_diff.unsqueeze(dim=1))
        true_phase_ratio = true_phase_ratio[diag==False]
        true_phase_ratio[true_phase_ratio < self.eps] = self.eps
        true_phase_ratio = torch.log(true_phase_ratio)

        return F.mse_loss(train_phase_ratio, true_phase_ratio)
    

class LogRatioLoss_lin(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.eps = 1e-6
        mask = torch.arange(args.batch_size)
        self.upper_tri = (mask.unsqueeze(0) > mask.unsqueeze(1)).cuda()
        mask = torch.arange(args.batch_size*(args.batch_size-1) // 2)
        self.diag = (mask.unsqueeze(0) == mask.unsqueeze(1)).cuda()

    def forward(self, rep, true_cossim):
        rep1, rep2 = torch.chunk(rep, chunks=2)
        train_cossim = F.cosine_similarity(rep.unsqueeze(1), rep.unsqueeze(0), dim=2)
        train_cossim = train_cossim[self.upper_tri==True]
        train_phase_diff = torch.acos(train_cossim)
        train_phase_diff[train_phase_diff==0] = self.eps
        train_phase_ratio = torch.divide(train_phase_diff.unsqueeze(dim=0),
                                         train_phase_diff.unsqueeze(dim=1))

        train_phase_ratio = train_phase_ratio[self.diag==False]
        train_phase_ratio[train_phase_ratio==0] = self.eps
        train_phase_ratio = torch.log(train_phase_ratio)
        
        true_cossim = true_cossim[self.upper_tri==True]
        true_phase_diff = torch.acos(true_cossim)
        true_phase_diff[true_phase_diff==0] = self.eps
        true_phase_ratio = torch.divide(true_phase_diff.unsqueeze(dim=0),
                                        true_phase_diff.unsqueeze(dim=1))
        true_phase_ratio = true_phase_ratio[self.diag==False]
        true_phase_ratio[true_phase_ratio==0] = self.eps
        true_phase_ratio = torch.log(true_phase_ratio)

        return F.mse_loss(train_phase_ratio, true_phase_ratio)
    
def QuantError(rep, p=2):
    return -1 * torch.pow(rep, exponent=p).mean()

def QuantError_(rep):
    quant_rep = torch.sign(rep)
    cossim = F.cosine_similarity(rep, quant_rep, dim=1)
    phase_diff = torch.acos(cossim)
    return phase_diff.norm(p=2)

load_loss_fn = {
    "InfoNCELoss": InfoNCELoss,
    "TripletLoss": TripletLoss,
    "NSSALoss": NSSALoss,
    "LogRatioLoss": LogRatioLoss
}



