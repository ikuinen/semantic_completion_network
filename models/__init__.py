import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer


class MainModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dropout = 0.1
        self.target_stride = config['target_stride']
        self.vocab_size = config['vocab_size']
        # if config['target_stride'] > 1:
        #     self.avg_pool = nn.AvgPool1d(config['target_stride'], config['target_stride'])
        # else:
        #     self.avg_pool = None
        self.frame_fc = nn.Linear(config['frames_input_size'], config['hidden_size'])
        self.word_fc = nn.Linear(config['words_input_size'], config['hidden_size'])
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.trans = DualTransformer(**config['DualTransformer'])
        self.fc_props = nn.Linear(config['hidden_size'], len(config['prop_width']))
        self.fc_comp = nn.Linear(config['hidden_size'], self.vocab_size)

        self.word_pos_encoder = SinusoidalPositionalEmbedding(config['hidden_size'], 0, 20)

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights,
                props, props_valid, num_proposals, random_p, tau=0.60, **kwargs):
        bsz = frames_feat.size(0)
        # props = props.squeeze(0)
        # props_valid = props_valid.squeeze(0)
        words_feat[:, 0] = self.start_vec.cuda()
        words_pos = self.word_pos_encoder(words_feat)

        frames_feat = F.dropout(frames_feat, self.dropout, self.training)
        frames_feat = self.frame_fc(frames_feat)
        ori_frames_feat = frames_feat

        # if self.avg_pool is not None:
        #     frames_feat = self.avg_pool(frames_feat.transpose(-1, -2)).transpose(-1, -2)
        frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        # proposals scoring
        enc_out, h = self.trans(frames_feat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        props_align = torch.sigmoid(self.fc_props(h))  # [nb, nc, np]
        # proposals selection
        props_chosen, props_idx = self._select_proposals(props, props_valid, props_align,
                                                         random_p=random_p, num_proposals=num_proposals,
                                                         tau=tau)
        props_align = props_align.contiguous().view(bsz, -1)
        props_align = props_align.gather(dim=1, index=props_idx)

        props_feat, props_len, props_mask = self._generate_proposals_feat(ori_frames_feat, props_chosen)
        # if self.avg_pool is not None:
        #     props_feat = self.avg_pool(props_feat.transpose(-1, -2)).transpose(-1, -2)
        words_feat = self._mask_words(words_feat, words_len, weights=weights) + words_pos
        words_feat1 = words_feat[:, :-1]
        words_id1 = words_id
        words_mask1 = words_mask[:, :-1]

        weights = weights.unsqueeze(1) \
            .expand(bsz, num_proposals, -1).contiguous().view(bsz * num_proposals, -1)
        words_mask1 = words_mask1.unsqueeze(1) \
            .expand(bsz, num_proposals, -1).contiguous().view(bsz * num_proposals, -1)
        words_id1 = words_id1.unsqueeze(1) \
            .expand(bsz, num_proposals, -1).contiguous().view(bsz * num_proposals, -1)
        words_feat1 = words_feat1.unsqueeze(1) \
            .expand(bsz, num_proposals, -1, -1).contiguous().view(bsz * num_proposals, words_mask1.size(1), -1)
        # semantic completion
        _, h = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2)
        words_logit = self.fc_comp(h)

        weights = None
        if self.training:
            neg_words_logit, neg = None, False
            if neg:
                frames_feat1 = frames_feat[list(reversed(range(bsz)))]
                _, h = self.trans(frames_feat1, None,
                                  words_feat, words_mask, decoding=1, enc_out=enc_out)
                neg_props_align = torch.sigmoid(self.fc_props(h))  # [nb, nc, np]
                # proposals selection
                neg_props_chosen, neg_props_idx = self._select_proposals(props, props_valid, neg_props_align,
                                                                         random_p=random_p, num_proposals=num_proposals,
                                                                         tau=0.60)

                neg_props_feat, neg_props_len, neg_props_mask = \
                    self._generate_proposals_feat(ori_frames_feat[list(reversed(range(bsz)))], neg_props_chosen)
                _, h = self.trans(neg_props_feat, neg_props_mask, words_feat1, words_mask1, decoding=2)
                neg_words_logit = self.fc_comp(h)
            return {
                'neg_words_logit': neg_words_logit,
                'props_align': props_align,  # [nb, np]
                'words_logit': words_logit,  # [nb * nw, seq_len, vocab]
                'words_id': words_id1,  # [nb * nw, seq_len]
                'weights': weights,
                'words_mask': words_mask1,
            }
        else:
            return {
                'props_chosen': props_chosen,  # [nb, np]
                'props_align': props_align,  # [nb, np]
                'words_logit': words_logit,  # [nb * nw, seq_len, vocab]
                'words_id': words_id1,  # [nb * nw, seq_len]
                'weights': weights,
                'words_mask': words_mask1,
            }

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.cuda().unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = l // 3
            masked_words.append(torch.zeros([words_feat.size(1)]).byte().cuda())
            p = weights[i, :l].cpu().numpy()
            # print(p)
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        # exit(0)
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1

    def _select_proposals(self, props, props_valid, props_align,
                          random_p=0.0, num_proposals=3, tau=0.6):
        bsz = props.size(0)
        props = props.view(bsz, -1, 2)
        props_valid = props_valid.view(bsz, -1)

        props_chosen = []
        props_idx = []

        def choose(size):
            if np.random.rand() < random_p:
                get_id = np.random.choice(np.arange(0, size), replace=False)
            else:
                get_id = 0
            return get_id

        for i, a in enumerate(props_align):
            a = a.contiguous().view(-1).masked_fill(props_valid[i] == 0, 0)

            # reorder
            idx = torch.argsort(a, descending=True)
            props1 = props[i].index_select(dim=0, index=idx)

            # remove illegal
            kidx = props1[:, 0] >= 0
            idx = idx[kidx]
            props1 = props1[kidx]

            pid = choose(props1.size(0))
            cp, cp_idx = [props1[pid]], [idx[pid]]
            for _ in range(1, num_proposals):
                tmp = cp[-1].unsqueeze(0).expand(props1.size(0), 2)
                iou = calculate_IoU_batch((tmp[:, 0].float(), tmp[:, 1].float()),
                                          (props1[:, 0].float(), props1[:, 1].float()))
                kidx = iou < tau
                if int(kidx.sum()) > 2:
                    idx = idx[kidx]
                    props1 = props1[kidx]
                pid = choose(props1.size(0))
                cp.append(props1[pid])
                cp_idx.append(idx[pid])

            cp, cp_idx = torch.stack(cp, 0), torch.stack(cp_idx, 0)
            # print(cp, cp_idx)
            props_chosen.append(cp)
            props_idx.append(cp_idx)
            # exit(0)
        props_chosen = torch.stack(props_chosen, 0)
        props_idx = torch.stack(props_idx, 0)
        # print(props_chosen)
        return props_chosen, props_idx

    def _generate_proposals_feat(self, frames_feat, props):
        props_feats = []
        props_len = []

        for f, p in zip(frames_feat, props):
            for s, e in p:
                s, e = int(s) * self.target_stride, int(e) * self.target_stride
                clip_len = e - s
                idx = np.linspace(start=0, stop=clip_len - 1, num=16).astype(np.int32)
                try:
                    props_feats.append(f[s:e+1][idx])
                except IndexError:
                    print(f.size(), (s, e))
                    exit(0)
                props_len.append(props_feats[-1].size(0))
        # print(props_len)
        # exit(0)
        # max_len = max(props_len)
        # for i in range(len(props_feats)):
        #     props_feats[i] = F.pad(props_feats[i], [0, 0, 0, max_len - props_len[i]])

        props_feats = torch.stack(props_feats, 0)
        # props_len = torch.from_numpy(np.asarray(props_len).astype(np.int64)).cuda()
        # props_mask = _generate_mask(props_feats, props_len)
        return props_feats, props_len, None


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)]).byte().cuda())
            mask[-1][:l] = 1
        mask = torch.stack(mask, 0)
    return mask


def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def make_positions(tensor, padding_idx):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    return (
                   torch.cumsum(mask, dim=1).type_as(mask) * mask
           ).long() + padding_idx


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.cuda(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
