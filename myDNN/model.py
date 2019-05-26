import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import dataset
import numpy as np


# Define model architecture

class SetConv(nn.Module):
    def __init__(self, predicate_feats, join_feats, join_selectivity_feats, hid_units_predicate, hid_units_join_selectivity,
                 hid_units_join, hid_units_output):
        super(SetConv, self).__init__()

        self.selectivity_mlp1 = nn.Linear(join_selectivity_feats, hid_units_join_selectivity)
        self.selectivity_mlp2 = nn.Linear(hid_units_join_selectivity, hid_units_join_selectivity)
        self.predicate_mlp1 = nn.Linear(predicate_feats, hid_units_predicate)
        self.predicate_mlp2 = nn.Linear(hid_units_predicate, hid_units_predicate)
        self.join_mlp1 = nn.Linear(join_feats + join_selectivity_feats, hid_units_join)
        self.join_mlp2 = nn.Linear(hid_units_join, hid_units_join)
        self.out_mlp1 = nn.Linear(hid_units_join + hid_units_predicate, hid_units_output)
        self.out_mlp2 = nn.Linear(hid_units_output, 1)

    def forward(self, predicates, joins, joins_selectivity, predicate_mask, join_mask):
            # samples has shape [batch_size x num_joins+1 x sample_feats]
            # predicates has shape [batch_size x num_predicates x predicate_feats]
            # joins has shape [batch_size x num_joins x join_feats]

            # hid_join_selectivity = F.relu(self.selectivity_mlp1(joins_selectivity))
            # hid_join_selectivity = F.relu(self.selectivity_mlp2(hid_join_selectivity))

            full_join = torch.cat((joins_selectivity, joins), 2)
            hid_join = F.relu(self.join_mlp1(full_join))
            hid_join = F.relu(self.join_mlp2(hid_join))
            hid_join = hid_join * join_mask
            hid_join = torch.sum(hid_join, dim=1, keepdim=False)
            join_norm = join_mask.sum(1, keepdim=False)
            hid_join = hid_join / join_norm

            hid_predicate = F.relu(self.predicate_mlp1(predicates))
            hid_predicate = F.relu(self.predicate_mlp2(hid_predicate))
            hid_predicate = hid_predicate * predicate_mask
            hid_predicate = torch.sum(hid_predicate, dim=1, keepdim=False)
            predicate_norm = predicate_mask.sum(1, keepdim=False)
            hid_predicate = hid_predicate / predicate_norm

            hid = torch.cat((hid_predicate, hid_join), 1)
            hid = F.relu(self.out_mlp1(hid))
            out = torch.sigmoid(self.out_mlp2(hid))
            return out


def train():

    pass
