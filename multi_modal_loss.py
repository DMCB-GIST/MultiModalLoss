import maimport torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.utils import common_functions as c_f
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu
from pytorch_metric_learning.losses.base_metric_loss_function import BaseMetricLossFunction
from pytorch_metric_learning.losses.mixins import WeightRegularizerMixin

###### This code is based on the paper ICCV'19: "SoftTriple Loss: Deep Metric Learning Without Triplet Sampling" . ######
###### Modified from https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/losses/soft_triple_loss.py ######

class MultiModalLoss(WeightRegularizerMixin, BaseMetricLossFunction,nn.Module):
    def __init__(
        self,
        num_classes,
        num_modalities,
        proxies_per_class=20,
        gamma=0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        assert self.distance.is_inverted
        self.gamma = 1.0 / gamma
        self.num_classes = num_classes
        self.num_modalities = num_modalities
        self.proxies_per_class = proxies_per_class
        self.fc = torch.nn.Parameter(
            torch.Tensor(num_classes * num_modalities, num_classes * proxies_per_class)
        )
        self.weight_init_func(self.fc)
        self.add_to_recordable_attributes(
            list_of_names=[
                "gamma",
                "proxies_per_class",
                "num_classes",
                "num_modalities",
            ],
            is_stat=False,
        )

    def cast_types(self, dtype, device):
        self.fc.data = c_f.to_device(self.fc.data, device=device, dtype=dtype)

    def compute_loss(self, embeddings, labels, indices_tuple):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        miner_weights = lmu.convert_to_weights(indices_tuple, labels, dtype=dtype)
        proxy = F.normalize(self.fc.clamp(min=0), p=2, dim=0)

        sim_to_proxies = self.distance(embeddings, proxy.t())         
        



        sim_to_proxies_2d = sim_to_proxies.view(
            -1, self.num_classes* self.proxies_per_class
        )

        att = F.softmax(sim_to_proxies_2d * self.gamma, dim=1)
        att = torch.mm(att , proxy.t())  
        embeddings_attention = torch.mul(embeddings,att) 
        embeddings_attention= embeddings_attention.view(-1,self.num_modalities,self.num_classes)   
        attended_output = torch.sum(embeddings_attention,dim=1)
        attended_output = attended_output.view(-1,self.num_classes)
        attended_output =  attended_output+torch.sum(embeddings.view(-1,self.num_modalities,self.num_classes),dim=1).view(-1,self.num_classes)


        sim_to_proxies_3d = sim_to_proxies.view(  
            -1, self.num_classes, self.proxies_per_class)
        normalized = F.softmax(sim_to_proxies_3d * self.gamma, dim=1)
        normalized_sim = torch.sum(normalized * sim_to_proxies_3d, dim=2)

        output = normalized_sim + attended_output

        loss = F.cross_entropy(
            output, labels, reduction="none"
        )
        loss = loss * miner_weights

        loss_dict = {
            "loss": {
                "losses": loss,
                "indices": c_f.torch_arange_from_size(embeddings),
                "reduction_type": "element"
            }
        }
        self.add_weight_regularization_to_loss_dict(loss_dict, proxy.t())

        return loss_dict
        
    def predict(self, embeddings):
        dtype, device = embeddings.dtype, embeddings.device
        self.cast_types(dtype, device)
        proxy = F.normalize(self.fc.clamp(min=0), p=2, dim=0)
        sim_to_proxies = self.distance(embeddings, proxy.t())         

        sim_to_proxies_2d = sim_to_proxies.view(
            -1, self.num_classes* self.proxies_per_class
        )
        att = F.softmax(sim_to_proxies_2d * self.gamma, dim=1)
        att = torch.mm(att , proxy.t())  
        embeddings_attention = torch.mul(embeddings,att) 
        embeddings_attention= embeddings_attention.view(-1,self.num_modalities,self.num_classes)   
        attended_output = torch.sum(embeddings_attention,dim=1)
        attended_output = attended_output.view(-1,self.num_classes)
        attended_output =  attended_output+torch.sum(embeddings.view(-1,self.num_modalities,self.num_classes),dim=1).view(-1,self.num_classes)


        sim_to_proxies_3d = sim_to_proxies.view(  
            -1, self.num_classes, self.proxies_per_class)
        normalized = F.softmax(sim_to_proxies_3d * self.gamma, dim=1)
        normalized_sim = torch.sum(normalized * sim_to_proxies_3d, dim=2)

        output =  F.softmax(normalized_sim + attended_output,dim=1)
        
        return output
        
    def get_default_distance(self):
        return CosineSimilarity()
    
    def get_default_weight_init_func(self):
        return c_f.TorchInitWrapper(torch.nn.init.kaiming_uniform_, a=math.sqrt(5))