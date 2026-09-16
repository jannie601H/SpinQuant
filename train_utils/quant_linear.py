# coding=utf-8
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch._tensor import Tensor


class QuantizeLinear(nn.Linear):
    def forward(
        self,
        input: Tensor,
        R1=None,
        R2=None,
        transpose=False,
    ) -> Tensor:
        # quantize weight
        weight = self.weight
        bias = self.bias
        if R1 is not None:
            dtype = self.weight.dtype
            if not transpose:
                weight = (self.weight.to(torch.float64) @ R1.to(torch.float64)).to(
                    dtype
                )
            else:
                weight = (R1.T.to(torch.float64) @ self.weight.to(torch.float64)).to(
                    dtype
                )
                if bias is not None:
                    bias = (R1.T.double() @ bias.double()).to(dtype)
        if R2 is not None:
            had_dim = R2.shape[0]
            dtype = weight.dtype
            W_ = weight if transpose else weight.t()
            shape = W_.shape
            temp = W_.reshape(-1, shape[-1] // had_dim, had_dim)
            temp = temp.double() @ R2.double()
            weight = temp.reshape(shape)
            if not transpose:
                weight = weight.t()
                if bias is not None:
                    bias = (
                        bias.reshape(-1, had_dim).double() @ R2.double()
                    ).reshape_as(bias).to(dtype)
            weight = weight.to(dtype)
        if hasattr(self, "quantizer"):
            dtype = weight.dtype
            self.quantizer.find_params(weight.data)
            weight = self.quantizer.quantize(weight).to(dtype)

        return nn.functional.linear(input, weight, bias)
