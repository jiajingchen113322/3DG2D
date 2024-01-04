import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _l2norm(x, dim=1, keepdim=True):
    return x / (1e-16 + torch.norm(x, 2, dim, keepdim))


def l2distance(x, y):
    """
    Input:
        x [.., c, M_x]
        y [.., c, M_y]
    Return:
        ret [.., M_x, M_y]
    """
    
    assert x.shape[:-2] == y.shape[:-2]
    prefix_shape = x.shape[:-2]

    c, M_x = x.shape[-2:]
    M_y = y.shape[-1]
    
    x = x.view(-1, c, M_x)
    y = y.view(-1, c, M_y)

    x_t = x.transpose(1, 2)
    x_t2 = x_t.pow(2.0).sum(-1, keepdim=True)
    y2 = y.pow(2.0).sum(1, keepdim=True)

    ret = x_t2 + y2 - 2.0 * x_t@y
    ret = ret.view(prefix_shape + (M_x, M_y))
    return ret




class Similarity(nn.Module):
    def __init__(self, metric='cosine'):
        super().__init__()
        self.metric = metric

    def forward(self, support_xf, query_xf):
        # query_xf: [b, q, c, h, w]
        # support_xf: [b, n, c, hxw]

        if query_xf.dim() == 5:
            b, q, c, h, w = query_xf.shape
            query_xf = query_xf.view(b, q, c, h*w)
        else:
            b, q = query_xf.shape[:2]

        s = support_xf.shape[1]

        support_xf = support_xf.unsqueeze(1).expand(-1, q, -1, -1, -1)
        query_xf = query_xf.unsqueeze(2).expand(-1, -1, s, -1, -1)
        M_q = query_xf.shape[-1]
        M_s = support_xf.shape[-1]

        if self.metric == 'cosine':
            support_xf = _l2norm(support_xf, dim=-2)
            query_xf = _l2norm(query_xf, dim=-2)
            query_xf = torch.transpose(query_xf, 3, 4)
            return query_xf@support_xf # bxQxNxM_qxM_s
        elif self.metric == 'innerproduct':
            query_xf = torch.transpose(query_xf, 3, 4)
            return query_xf@support_xf # bxQxNxM_qxM_s
        elif self.metric == 'euclidean':
            return 1 - l2distance(query_xf, support_xf)
        elif self.metric == 'neg_ed':
            query_xf = query_xf.contiguous().view(-1, c, M_q).transpose(-2, -1).contiguous()
            support_xf = support_xf.contiguous().view(-1, c, M_s).transpose(-2, -1).contiguous()
            dist = torch.cdist(query_xf, support_xf)
            return -dist.view(b, q, s, M_q, M_s) / 2.
        else:
            raise NotImplementedError


class MEL(nn.Module):
    def __init__(self,n_way,k_shot,query):
        super().__init__()

        self.n_way=n_way
        self.k_shot=k_shot
        self.q=query


        self.inner_simi = Similarity(metric='cosine')
        # self.gamma = cfg.model.mel.gamma
        # self.gamma2 = cfg.model.mel.gamma2
        self.gamma=20
        self.gamma2=10

        self.criterion = nn.NLLLoss()
        # self.katz_factor = cfg.model.mel.katz_factor
        self.katz_factor = 0.5

    def _equilibrium_state(self, T_sq, T_qs):
        N, M_q, M_s = T_sq.shape
        P = T_sq@T_qs
        # row-wise stochastic matrix to column-wise
        P = P.transpose(-2, -1) # [N, M_q, M_q]
        P_aug = torch.nn.functional.pad(P - torch.eye(M_q, device=P.device).expand_as(P), (0, 0, 0, 1), 'constant', 1)
        Pb = torch.Tensor([0] * M_q + [1]).to(P.device).view(1, M_q + 1, 1).expand(N, -1, -1)
        # generalized inversion not full ranks for column -> Failure Cases in training
        # Q_t = torch.bmm((P_aug.transpose(-2, -1)@P_aug).inverse(), P_aug.transpose(-2, -1))@Pb # [N, M_q, 1]
        # Q_t = torch.pinverse(P_aug)@Pb
        # IMPORTANT use QR decomposition for numerical stable
        QQ, RR = torch.qr(P_aug)
        # Q_t = torch.inverse(RR)@(QQ.transpose(-2, -1)@Pb)
        Q_t = torch.triangular_solve(QQ.transpose(-2, -1)@Pb, RR).solution
        # svd based pinverse would generate nan gradient for close eigenvalue, use left-pseudo inv instead
        # should detect for ill-conditioned cases
        # wellcond_m = self.wellcond_detect(P_aug)

        Q_t = Q_t.squeeze(-1).unsqueeze(-2) # [N, 1, M_q]
        S_t = Q_t@T_sq # [N, 1, M_s]
        S_t = S_t.squeeze(-2)
        return S_t, Q_t

    def averaging_based_similarities(self, support_xf, support_y, query_xf, query_y):
        s, c, h, w = support_xf.shape
        q = query_xf.shape[0]
        support_xf = support_xf.view(self.n_way, self.k_shot, c, h, w).mean(1)
        support_xf = support_xf.view(self.n_way, c, h * w)
        S = self.inner_simi(support_xf, query_xf) # [b, q, N, M_q, M_s]
        M_q = S.shape[-2]
        M_s = S.shape[2] * S.shape[-1]
        S = S.permute(0, 1, 3, 2, 4).contiguous().view(b * q, M_q, M_s)
        return S

    def bipartite_katz_forward(self, support_xf, support_y, query_xf, query_y, similarity_f):
        katz_factor = self.katz_factor
        S = similarity_f(support_xf, support_y, query_xf, query_y)
        N_examples, M_q, M_s = S.shape
        St = S.transpose(-2, -1)
        device = S.device

        T_sq = torch.exp(self.gamma * (S - S.max(-1, keepdim=True)[0]))
        T_sq = T_sq / T_sq.sum(-1, keepdim=True) # row-wise stochastic
        T_qs = torch.exp(self.gamma2 * (St - St.max(-1, keepdim=True)[0])) # [b * q, M_s, M_q]
        T_qs = T_qs / T_qs.sum(-1, keepdim=True) # row-wise stochastic

        T = torch.cat([
            torch.cat([torch.zeros((N_examples, M_s, M_s), device=device), T_sq.transpose(-2, -1)], dim=-1),
            torch.cat([T_qs.transpose(-2, -1), torch.zeros((N_examples, M_q, M_q), device=device)], dim=-1),
        ], dim=-2)
        katz = (torch.inverse(torch.eye(M_s + M_q, device=device)[None].repeat(N_examples, 1, 1) - katz_factor * T) - \
                torch.eye(M_s + M_q, device=S.device)[None].repeat(N_examples, 1, 1))@torch.ones((N_examples, M_s + M_q, 1), device=device)
        partial_katz = katz.squeeze(-1)[:, :M_s] / katz.squeeze(-1)[:, :M_s].sum(-1, keepdim=True)
        predicts = partial_katz.view(N_examples, self.n_way, -1).sum(-1)

        query_y = query_y.view(N_examples)
        loss = self.criterion(torch.log(predicts), query_y)
        if self.training:
            return {"MEL_loss": loss}
        else:
            _, predict_labels = torch.max(predicts, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards, loss

    def overdetermined_forward(self, support_xf, support_y, query_xf, query_y, similarity_f):
        S = similarity_f(support_xf, support_y, query_xf, query_y)
        N_examples, M_q, M_s = S.shape
        St = S.transpose(-2, -1)

        T_sq = torch.exp(self.gamma * (S - S.max(-1, keepdim=True)[0]))
        T_sq = T_sq / T_sq.sum(-1, keepdim=True) # row-wise stochastic
        T_qs = torch.exp(self.gamma2 * (St - St.max(-1, keepdim=True)[0])) # [b * q, M_s, M_q]
        T_qs = T_qs / T_qs.sum(-1, keepdim=True) # row-wise stochastic

        S_t, _ = self._equilibrium_state(T_sq, T_qs)
        S_t = S_t.view(S_t.shape[0], self.n_way, -1).sum(-1)

        query_y = query_y.view(N_examples)
        loss = self.criterion(torch.log(S_t), query_y)
        if self.training:
            return {"MEL_loss": loss}
        else:
            _, predict_labels = torch.max(S_t, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards, loss

    def blockwise_katz_forward(self, support_xf, support_y, query_xf, query_y, similarity_f):
        alpha = self.katz_factor
        alpha_2 = alpha * alpha

        S = similarity_f(support_xf, support_y, query_xf, query_y)
        N_examples, M_q, M_s = S.shape
        St = S.transpose(-2, -1)
        device = S.device

        P_sq = torch.exp(self.gamma * (St - St.max(-2, keepdim=True)[0]))
        P_sq = P_sq / P_sq.sum(-2, keepdim=True)
        P_qs = torch.exp(self.gamma2 * (S - S.max(-2, keepdim=True)[0]))
        P_qs = P_qs / P_qs.sum(-2, keepdim=True)

        inverted_matrix = torch.inverse(torch.eye(M_q, device=device)[None].repeat(N_examples, 1, 1) - alpha_2 * P_qs@P_sq)
        katz = (alpha_2 * P_sq@inverted_matrix@P_qs).sum(-1) + (alpha * P_sq@inverted_matrix).sum(-1)
        katz = katz / katz.sum(-1, keepdim=True)
        predicts = katz.view(N_examples, self.n_way, -1).sum(-1)

        query_y = query_y.view(N_examples)
        loss = self.criterion(torch.log(predicts), query_y)
        if self.training:
            return {"MEL_loss": loss}
        else:
            _, predict_labels = torch.max(predicts, 1)
            rewards = [1 if predict_labels[j]==query_y[j].to(predict_labels.device) else 0 for j in range(len(query_y))]
            return rewards, loss

    # def forward(self, support_xf, support_y, query_xf, query_y):
    def forward(self, feat,label):
        support_xf=feat[:self.n_way*self.k_shot] # (support_img_num,channel,h,w)
        query_xf=feat[self.n_way*self.k_shot:] # (query_img_num,channel,h,w)

        support_xf=F.adaptive_max_pool2d(query_xf,query_xf.shape[-1])
        support_y=label[0]
        query_y=label[1]

        return self.bipartite_katz_forward(support_xf, support_y, query_xf, query_y, self.averaging_based_similarities)


if __name__=='__main__':
    n_way=5
    k_shot=1
    q=3

    support_xf=torch.randn((n_way*k_shot,64,7,7))
    query_xf=torch.randn((n_way*q,64,7,7))
    feat=torch.cat((support_xf,query_xf))

    s_label=torch.arange(n_way)
    q_label=torch.arange(n_way).repeat_interleave(q)
    label=[s_label,q_label]

    net=MEL(n_way,k_shot,q)
    net(feat,label)