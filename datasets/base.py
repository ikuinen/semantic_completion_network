import numpy as np
import torch
from gensim.utils import tokenize
from torch.utils.data import Dataset

from utils import load_json
import nltk


def calculate_IoU_batch(i0, i1):
    union = (torch.min(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.max(torch.stack([i0[1], i1[1]], 0), 0)[0])
    inter = (torch.max(torch.stack([i0[0], i1[0]], 0), 0)[0], torch.min(torch.stack([i0[1], i1[1]], 0), 0)[0])
    iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


class BaseDataset(Dataset):
    def __init__(self, data_path, vocab, args, **kwargs):
        self.vocab = vocab
        self.args = args
        self.data = load_json(data_path)
        self.ori_data = self.data
        self.max_num_frames = args['max_num_frames']
        self.max_num_words = args['max_num_words']
        self.target_stride = args['target_stride']

        num_props = len(args['prop_width'])
        # self.prop_width = np.asarray(args['prop_width'])
        self.prop_width = []
        for i in range(1, num_props + 1):
            self.prop_width.append(1.0 / num_props * i)
        self.prop_width = np.asarray(self.prop_width)
        print('prop width', self.prop_width)

        self.keep_vocab = dict()
        for w, _ in vocab['counter'].most_common(8000):
            self.keep_vocab[w] = self.vocab_size

    def _generate_props(self, frames_feat):
        num_clips = self.num_clips
        prop_width = (self.prop_width * len(frames_feat)).astype(np.int64)
        prop_width[prop_width == 0] = 1

        props = []
        valid = []
        end = np.arange(1, num_clips + 1).astype(np.int64)
        for w in prop_width:
            start = end - w
            props.append(np.stack([start, end], -1))  # [nc, 2]
            valid.append(np.logical_and(props[-1][:, 0] >= 0, props[-1][:, 1] <= len(frames_feat)))  # [nc]
        props = np.stack(props, 1).astype(np.int64)  # [nc, np, 2]
        # print(props[len(frames_feat)-1], len(frames_feat))
        valid = np.stack(valid, 1).astype(np.uint8)  # [nc, np]
        valid_torch = torch.from_numpy(valid)
        props_torch = torch.from_numpy(props)
        return props_torch, valid_torch

    def load_data(self, data):
        self.data = data

    def _load_frame_features(self, vid):
        raise NotImplementedError

    def _sample_frame_features(self, frames_feat):
        num_clips = self.num_clips
        if len(frames_feat) <= num_clips and False:
            return frames_feat
        else:
            keep_idx = np.arange(0, num_clips + 1) / num_clips * len(frames_feat)
            keep_idx = np.round(keep_idx).astype(np.int64)
            keep_idx[keep_idx >= len(frames_feat)] = len(frames_feat) - 1
            frames_feat1 = []
            for j in range(num_clips):
                s, e = keep_idx[j], keep_idx[j + 1]
                assert s <= e
                if s == e:
                    frames_feat1.append(frames_feat[s])
                else:
                    frames_feat1.append(frames_feat[s:e].mean(axis=0))
            return np.stack(frames_feat1, 0)

    @property
    def num_clips(self):
        return self.max_num_frames // self.target_stride

    @property
    def vocab_size(self):
        return len(self.keep_vocab) + 1

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid, duration, timestamps, sentence = self.data[index]
        duration = float(duration)

        # words = [w.lower() for w in tokenize(sentence)]
        weights = []
        words = []
        for word, tag in nltk.pos_tag(nltk.tokenize.word_tokenize(sentence)):
            word = word.lower()
            if word in self.keep_vocab:
                if 'NN' in tag:
                    weights.append(2)
                elif 'VB' in tag:
                    weights.append(2)
                elif 'JJ' in tag or 'RB' in tag:
                    weights.append(2)
                else:
                    weights.append(1)
                words.append(word)
        # weights = np.exp(weights)
        # weights /= np.sum(weights)
        # print(weights)
        # exit(0)
        # words = [w.lower() for w in nltk.tokenize.word_tokenize(sentence)]
        # words = [w for w in words if w in self.keep_vocab]
        frames_feat = self._sample_frame_features(self._load_frame_features(vid))
        props, valid = self._generate_props(frames_feat)
        words_id = [self.keep_vocab[w] for w in words]
        words_feat = [self.vocab['id2vec'][self.vocab['w2id'][words[0]]].astype(np.float32)]
        words_feat.extend([self.vocab['id2vec'][self.vocab['w2id'][w]].astype(np.float32) for w in words])

        props1 = props.view(-1, 2).float() * duration / self.num_clips
        valid1 = valid.view(-1)
        gts = torch.tensor([timestamps[0], timestamps[1]]).unsqueeze(0).expand(props1.size(0), -1).float()
        prop_align = calculate_IoU_batch((props1[:, 0], props1[:, 1]), (gts[:, 0], gts[:, 1]))
        prop_align = prop_align.masked_fill(valid1 == 0, 0)
        prop_gt = props.view(-1, 2)[torch.argmax(prop_align)]

        # print(prop_gt * duration / self.num_clips, timestamps)

        return {
            'frames_feat': frames_feat,
            'words_feat': words_feat,
            'words_id': words_id,
            'prop_gt': prop_gt,
            'weights': weights,
            'props': props,
            'props_valid': valid,
            'raw': [vid, duration, timestamps, sentence]
        }


def build_collate_data(max_num_frames, max_num_words, frame_dim, word_dim):
    def collate_data(samples):
        bsz = len(samples)
        batch = {
            'raw': [sample['raw'] for sample in samples],
        }

        frames_len = []
        words_len = []

        for i, sample in enumerate(samples):
            frames_len.append(min(len(sample['frames_feat']), max_num_frames))
            words_len.append(min(len(sample['words_id']), max_num_words))

        frames_feat = np.zeros([bsz, max_num_frames, frame_dim]).astype(np.float32)
        words_feat = np.zeros([bsz, max(words_len) + 1, word_dim]).astype(np.float32)
        words_id = np.zeros([bsz, max(words_len)]).astype(np.int64)
        weights = np.zeros([bsz, max(words_len)]).astype(np.float32)
        prop_gt = []
        props = []
        props_valid = []
        for i, sample in enumerate(samples):
            keep = min(len(sample['words_feat']), words_feat.shape[1])
            words_feat[i, :keep] = sample['words_feat'][:keep]
            keep = min(len(sample['words_id']), words_id.shape[1])
            words_id[i, :keep] = sample['words_id'][:keep]
            tmp = np.exp(sample['weights'][:keep])
            weights[i, :keep] = tmp / np.sum(tmp)

            frames_feat[i, :len(sample['frames_feat'])] = sample['frames_feat']
            prop_gt.append(sample['prop_gt'])
            props.append(sample['props'])
            props_valid.append(sample['props_valid'])

        batch.update({
            'net_input': {
                'frames_feat': torch.from_numpy(frames_feat),
                'frames_len': torch.from_numpy(np.asarray(frames_len)),
                'words_feat': torch.from_numpy(words_feat),
                'words_id': torch.from_numpy(words_id),
                'weights': torch.from_numpy(weights),
                'words_len': torch.from_numpy(np.asarray(words_len)),
                'prop_gt': torch.stack(prop_gt, 0),
                'props': torch.stack(props, 0),
                'props_valid': torch.stack(props_valid, 0)
            }
        })
        return batch

    return collate_data
