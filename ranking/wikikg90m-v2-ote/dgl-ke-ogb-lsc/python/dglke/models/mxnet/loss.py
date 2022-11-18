# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from ..base_loss import *
from .tensor_models import logsigmoid


class HingeLoss(BaseHingeLoss):
    def __init__(self, margin):
        assert False, 'HingeLoss is not implemented'

    def __call__(self, score, label):
        pass


class LogisticLoss(BaseLogisticLoss):
    def __init__(self):
        assert False, 'LogisticLoss is not implemented'

    def __call__(self, score, label):
        pass


class BCELoss(BaseBCELoss):
    def __init__(self):
        assert False, 'BCELoss is not implemented'

    def __call__(self, score, label):
        pass


class LogsigmoidLoss(BaseLogsigmoidLoss):
    def __init__(self):
        super(LogsigmoidLoss, self).__init__()

    def __call__(self, score, label):
        return -logsigmoid(label * score)


class LossGenerator(BaseLossGenerator):
    def __init__(self,
                 args,
                 loss_genre='Logsigmoid',
                 neg_adversarial_sampling=False,
                 adversarial_temperature=1.0,
                 pairwise=False):
        assert False, 'LossGenerator is not implemented'

    def _get_pos_loss(self, pos_score, edge_weight):
        pass

    def _get_neg_loss(self, neg_score, edge_weight):
        pass

    def get_total_loss(self, pos_score, neg_score, edge_weight):
        pass
