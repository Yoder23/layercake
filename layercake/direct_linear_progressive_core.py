"""Dual-path core with an analytic direct-linear MLP coefficient map."""
from __future__ import annotations
import torch
from torch import nn
from .dual_path_progressive_core import DualPathProgressiveCore, DualPathProgressiveLayer

class DirectLinearProgressiveLayer(DualPathProgressiveLayer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        del self.mlp_input_projection, self.mlp_norm, self.gate_up_proj, self.down_proj
        self.mlp_coefficient_projection=nn.Linear(self.full_width,self.bottleneck_width,bias=False)
        self.mlp_residual_mean=nn.Parameter(torch.zeros(self.full_width))
    def _mlp_delta(self,hidden:torch.Tensor)->torch.Tensor:
        coefficients=self.mlp_coefficient_projection(self.post_attention_norm(hidden))
        return self.mlp_residual_mean+self.mlp_output_projection(coefficients)

class DirectLinearProgressiveCore(DualPathProgressiveCore):
    def __init__(self,*,fixed_vocab_size:int,full_width:int=3072,bottleneck_width:int=192,attention_heads:int=2,replacement_layers:int=32,intermediate_size:int=768,rms_epsilon:float=1e-5,rope_theta:float=10000.0,maximum_source_actions:int=192,maximum_target_actions:int=320,maximum_sequence_actions:int=512):
        count=int(replacement_layers)
        super().__init__(fixed_vocab_size=fixed_vocab_size,full_width=full_width,bottleneck_width=bottleneck_width,attention_heads=attention_heads,replacement_layers=0,intermediate_size=intermediate_size,rms_epsilon=rms_epsilon,rope_theta=rope_theta,maximum_source_actions=maximum_source_actions,maximum_target_actions=maximum_target_actions,maximum_sequence_actions=maximum_sequence_actions)
        self.replacement_layers=count
        self.layers=nn.ModuleList(DirectLinearProgressiveLayer(self.full_width,self.bottleneck_width,self.attention_heads,self.intermediate_size,rms_epsilon=self.rms_epsilon,rope_theta=self.rope_theta) for _ in range(count))
    @staticmethod
    def parameter_count_for_config(*,fixed_vocab_size:int,full_width:int,bottleneck_width:int,replacement_layers:int,intermediate_size:int)->int:
        copied=2*fixed_vocab_size*full_width+(2*replacement_layers+1)*full_width
        per_layer=2*full_width*bottleneck_width+4*bottleneck_width*bottleneck_width+bottleneck_width + 2*full_width*bottleneck_width + full_width
        return copied+replacement_layers*per_layer
