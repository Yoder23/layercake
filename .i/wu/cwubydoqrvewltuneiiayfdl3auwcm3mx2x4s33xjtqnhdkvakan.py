# AOT ID: ['3_backward']
from ctypes import c_void_p, c_long, c_int
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from cmath import nanj
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile
from torch._inductor.codegen.memory_planning import _align as align
from torch import device, empty_strided
from torch._inductor.async_compile import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch._inductor.codegen.multi_kernel import MultiKernelCall
import triton
import triton.language as tl
from torch._inductor.runtime.triton_heuristics import start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_raw_stream
from torch._C import _cuda_getCurrentRawStream as get_raw_stream

aten = torch.ops.aten
inductor_ops = torch.ops.inductor
_quantized = torch.ops._quantized
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
empty_strided_cpu = torch._C._dynamo.guards._empty_strided_cpu
empty_strided_cuda = torch._C._dynamo.guards._empty_strided_cuda
empty_strided_xpu = torch._C._dynamo.guards._empty_strided_xpu
reinterpret_tensor = torch._C._dynamo.guards._reinterpret_tensor
alloc_from_pool = torch.ops.inductor._alloc_from_pool
async_compile = AsyncCompile()
empty_strided_p2p = torch._C._distributed_c10d._SymmetricMemory.empty_strided_p2p


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\3e\c3eekt3vhxpuekjjveqbdy6gmsdji4xvdoijx5i3hjvozmshd5jh.py
# Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   cross_entropy => full_default_1
# Graph fragment:
#   %full_default_1 : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_6, %full_default_1, %mul_12), kwargs = {})
#   %full_default_11 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([16, 176], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_11, [%bitwise_right_shift], %where_8, True), kwargs = {})
triton_poi_fused_embedding_dense_backward_nll_loss_forward_0 = async_compile.triton('triton_poi_fused_embedding_dense_backward_nll_loss_forward_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_nll_loss_forward_0', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_dense_backward_nll_loss_forward_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\ht\chtrvx3cxt74fwvcat3leqmpxvl4fiep5pp45ngnnn7hoh6ai7w5.py
# Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   cross_entropy => full_default
# Graph fragment:
#   %ne_4 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_2, -100), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %unsqueeze_2, %full_default), kwargs = {})
#   %full_default_5 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([135168, 16], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_default_5, 1, %where_4, -1.0), kwargs = {})
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_poi_fused_nll_loss_backward_nll_loss_forward_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_nll_loss_forward_1(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2162688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\l2\cl2b3llmznoeuwtqi2pqgq2fvuhntbqknfgbrqw7g6rn5b67msmw.py
# Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# Source node to ATen node mapping:
#   cross_entropy => full_default
# Graph fragment:
#   %ne_4 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_2, -100), kwargs = {})
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_4 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %unsqueeze_2, %full_default), kwargs = {})
#   %full_default_5 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([135168, 16], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %scatter : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_default_5, 1, %where_4, -1.0), kwargs = {})
triton_poi_fused_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_poi_fused_nll_loss_backward_nll_loss_forward_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_2', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_nll_loss_forward_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 135168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.full([1], 15, tl.int64)
    tmp2 = tmp0 & tmp1
    tmp3 = tl.full([1], -100, tl.int64)
    tmp4 = tmp2 != tmp3
    tmp5 = tl.full([1], 0, tl.int64)
    tmp6 = tl.where(tmp4, tmp2, tmp5)
    tl.device_assert((0 <= tmp6) & (tmp6 < 16), "index out of bounds: 0 <= tmp6 < 16")
    tmp8 = -1.0
    tl.store(out_ptr0 + (tmp6 + 16*x0), tmp8, None)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\au\cau77idi7jwqzuxqldo63xqa2outx7cajxgfews5cdmdkpywicow.py
# Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
# Source node to ATen node mapping:
#   cross_entropy => full_default
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0), kwargs = {dtype: torch.int64, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_5 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([135168, 16], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %ne_6 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_4, -100), kwargs = {})
#   %where_6 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_6, %unsqueeze_4, %full_default), kwargs = {})
#   %scatter_1 : [num_users=1] = call_function[target=torch.ops.aten.scatter.value](args = (%full_default_5, 1, %where_6, -1.0), kwargs = {})
triton_poi_fused_nll_loss_backward_nll_loss_forward_3 = async_compile.triton('triton_poi_fused_nll_loss_backward_nll_loss_forward_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_3', 'mutated_arg_names': ['out_ptr0'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_nll_loss_backward_nll_loss_forward_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 135168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tl.device_assert((0 <= tmp4) & (tmp4 < 16), "index out of bounds: 0 <= tmp4 < 16")
    tmp6 = -1.0
    tl.store(out_ptr0 + (tmp4 + 16*x0), tmp6, None)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\5j\c5jkfzdkgaimvi27nknvmmwzrwxzir5esjg3vqgwbmn26slyjzf4.py
# Topologically Sorted Source Nodes: [cross_entropy, cross_entropy_1], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward, aten._to_copy, aten._log_softmax_backward_data]
# Source node to ATen node mapping:
#   cross_entropy => convert_element_type_14, full_default_1
#   cross_entropy_1 => convert_element_type_17
# Graph fragment:
#   %ne_4 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_2, -100), kwargs = {})
#   %full_default_1 : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_5 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_4, %unsqueeze_3, %full_default_1), kwargs = {})
#   %mul_3 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter, %where_5), kwargs = {})
#   %convert_element_type_17 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_16, torch.float32), kwargs = {})
#   %exp_2 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%convert_element_type_17,), kwargs = {})
#   %sum_3 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_3, [1], True), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_2, %sum_3), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_3, %mul_4), kwargs = {})
#   %convert_element_type_21 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_5, torch.float16), kwargs = {})
#   %ne_6 : [num_users=2] = call_function[target=torch.ops.aten.ne.Scalar](args = (%unsqueeze_4, -100), kwargs = {})
#   %where_7 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%ne_6, %unsqueeze_3, %full_default_1), kwargs = {})
#   %mul_5 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%scatter_1, %where_7), kwargs = {})
#   %convert_element_type_14 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%convert_element_type_13, torch.float32), kwargs = {})
#   %exp_3 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%convert_element_type_14,), kwargs = {})
#   %sum_4 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_5, [1], True), kwargs = {})
#   %mul_6 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%exp_3, %sum_4), kwargs = {})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_5, %mul_6), kwargs = {})
#   %convert_element_type_25 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_6, torch.float16), kwargs = {})
triton_per_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_4 = async_compile.triton('triton_per_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 16},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_out_ptr1': '*fp16', 'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*i64', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_4', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 7, 'num_reduction': 2, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_4(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 135168
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (r0_1 + 16*x0), None)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_out_ptr0 + (r0_1 + 16*x0), None).to(tl.float32)
    tmp27 = tl.load(in_out_ptr1 + (r0_1 + 16*x0), None).to(tl.float32)
    tmp2 = tl.full([1, 1], 15, tl.int64)
    tmp3 = tmp1 & tmp2
    tmp4 = tl.full([1, 1], -100, tl.int64)
    tmp5 = tmp3 != tmp4
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tmp0 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.sum(tmp10, 1)[:, None]
    tmp15 = tmp14 != tmp4
    tmp16 = tl.where(tmp15, tmp6, tmp7)
    tmp17 = tmp13 * tmp16
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None]
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tl_math.exp(tmp22)
    tmp24 = tmp23 * tmp12
    tmp25 = tmp9 - tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tl_math.exp(tmp28)
    tmp30 = tmp29 * tmp20
    tmp31 = tmp17 - tmp30
    tmp32 = tmp31.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 16*x0), tmp26, None)
    tl.store(in_out_ptr1 + (r0_1 + 16*x0), tmp32, None)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\2q\c2qm5lps6cwenx4j4x4wgayahmouvngjkfuogvjrtmn7ssohhcwc.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_32 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mm_1, torch.float32), kwargs = {})
triton_poi_fused__to_copy_5 = async_compile.triton('triton_poi_fused__to_copy_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_5', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_5(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\ke\ckey35xzsl3qiqt4wgowtwt23pw4rhd33waeocpm57kb6mcgsapm.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_21, [0], True), kwargs = {dtype: torch.float32})
triton_red_fused_sum_6 = async_compile.triton('triton_red_fused_sum_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 32768, 'r0_': 128},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_6', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_6(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16896
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 16)
    x1 = xindex // 16
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 16*r0_2 + 2048*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp3, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\ez\cezazh3y4yiknvoc5xlksucg7fyvh4znz57cbcxxqvsndfdoc7mt.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
# Source node to ATen node mapping:
# Graph fragment:
#   %sum_5 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_21, [0], True), kwargs = {dtype: torch.float32})
triton_red_fused_sum_7 = async_compile.triton('triton_red_fused_sum_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 16, 'r0_': 2048},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_sum_7', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused_sum_7(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 1056
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 16*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\nr\cnrsvl4hcggwd7qj377rgjkhxwdhuj2bbszgdlrdjd5qbwgzyd45.py
# Topologically Sorted Source Nodes: [cross_entropy, add, mul, add_1, low_hidden], Original ATen: [aten.nll_loss_forward, aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.mul, aten.native_layer_norm, aten.embedding_dense_backward]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   cross_entropy => full_default_1
#   low_hidden => mul_1, sub
#   mul => mul
# Graph fragment:
#   %full_default_1 : [num_users=4] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %convert_element_type_31 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_16, torch.float32), kwargs = {})
#   %mul_8 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, %primals_7), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, 176), kwargs = {})
#   %sum_6 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_8, [3], True), kwargs = {})
#   %add : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, 1.0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %add), kwargs = {})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %mul_1 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_10 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_8, %mul_1), kwargs = {})
#   %sum_7 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_10, [3], True), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %sum_7), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%mul_9, %sum_6), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_8, %mul_11), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt, 176), kwargs = {})
#   %mul_12 : [num_users=3] = call_function[target=torch.ops.aten.mul.Tensor](args = (%div, %sub_9), kwargs = {})
#   %where_8 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_6, %full_default_1, %mul_12), kwargs = {})
#   %full_default_11 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([16, 176], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %index_put : [num_users=1] = call_function[target=torch.ops.aten.index_put.default](args = (%full_default_11, [%bitwise_right_shift], %where_8, True), kwargs = {})
#   %mul_14 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %primals_1), kwargs = {})
#   %mul_15 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_12, %add), kwargs = {})
#   %where_9 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%unsqueeze_6, %full_default_1, %mul_14), kwargs = {})
#   %index_put_1 : [num_users=1] = call_function[target=torch.ops.aten.index_put_.default](args = (%full_default_11, [%bitwise_right_shift], %where_9, True), kwargs = {})
#   %convert_element_type_39 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_19, torch.float32), kwargs = {})
#   %add_5 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_15, %convert_element_type_39), kwargs = {})
triton_per_fused__to_copy_add_embedding_dense_backward_mul_native_layer_norm_native_layer_norm_backward_nll_loss_forward_8 = async_compile.triton('triton_per_fused__to_copy_add_embedding_dense_backward_mul_native_layer_norm_native_layer_norm_backward_nll_loss_forward_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 262144, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_out_ptr1': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp32', 'in_ptr5': '*i64', 'in_ptr6': '*fp16', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_embedding_dense_backward_mul_native_layer_norm_native_layer_norm_backward_nll_loss_forward_8', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'out_ptr3', 'out_ptr4'], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 9, 'num_reduction': 2, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_embedding_dense_backward_mul_native_layer_norm_native_layer_norm_backward_nll_loss_forward_8(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 135168
    r0_numel = 176
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 176*x0), r0_mask, other=0.0)
    tmp1 = tl.load(in_out_ptr1 + (r0_1 + 176*x0), r0_mask, other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r0_1 + 176*x0), r0_mask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (r0_1 + 176*x0), r0_mask, other=0.0).to(tl.float32)
    tmp13 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp45 = tl.load(in_ptr6 + (r0_1 + 176*x0), r0_mask, other=0.0).to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 * tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 - tmp7
    tmp10 = tmp8 * tmp9
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 * tmp10
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 0.005681818181818182
    tmp25 = tmp9 * tmp24
    tmp26 = 176.0
    tmp27 = tmp14 * tmp26
    tmp28 = tmp27 - tmp18
    tmp29 = tmp10 * tmp23
    tmp30 = tmp28 - tmp29
    tmp31 = tmp25 * tmp30
    tmp33 = tl.full([XBLOCK, R0_BLOCK], 16, tl.int32)
    tmp34 = tmp32 + tmp33
    tmp35 = tmp32 < 0
    tmp36 = tl.where(tmp35, tmp34, tmp32)
    tl.device_assert((0 <= tmp36) & (tmp36 < 16), "index out of bounds: 0 <= tmp36 < 16")
    tmp38 = tl.full([1, 1], -1, tl.int64)
    tmp39 = tmp32 == tmp38
    tmp40 = 0.0
    tmp41 = tl.where(tmp39, tmp40, tmp31)
    tmp42 = tmp31 * tmp0
    tmp43 = tl.where(tmp39, tmp40, tmp42)
    tmp44 = tmp31 * tmp3
    tmp46 = tmp45.to(tl.float32)
    tmp47 = tmp44 + tmp46
    tl.store(in_out_ptr0 + (r0_1 + 176*x0), tmp10, r0_mask)
    tl.atomic_add(out_ptr3 + (tl.broadcast_to(r0_1 + 176*tmp36, [XBLOCK, R0_BLOCK])), tmp41, r0_mask, sem='relaxed')
    tl.atomic_add(out_ptr4 + (tl.broadcast_to(r0_1 + 176*tmp36, [XBLOCK, R0_BLOCK])), tmp43, r0_mask, sem='relaxed')
    tl.store(in_out_ptr1 + (r0_1 + 176*x0), tmp47, r0_mask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\65\c65k7q5l4qehhubo67nqhgb6omisnn7c2ultcxh5sime7d2h6tg4.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_31 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_16, torch.float32), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, %mul_1), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_13, [0, 1, 2]), kwargs = {})
#   %sum_9 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%convert_element_type_31, [0, 1, 2]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_backward_9 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_backward_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 65536, 'r0_': 1024},
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_backward_9', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 2, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_backward_9(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 33792
    r0_numel = 704
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 176)
    x1 = xindex // 176
    _tmp5 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    _tmp8 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 176*r0_2 + 123904*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (x0 + 176*r0_2 + 123904*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 * tmp2
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(r0_mask & xmask, tmp6, _tmp5)
        tmp7 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(r0_mask & xmask, tmp9, _tmp8)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp5, xmask)
    tl.store(out_ptr1 + (x3), tmp8, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\7f\c7fkh7egauzgmgb7l4gfkluawmwexzst6hfjy2tk7esknki4gav7.py
# Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
# Graph fragment:
#   %convert_element_type_31 : [num_users=3] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_16, torch.float32), kwargs = {})
#   %mul_13 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_31, %mul_1), kwargs = {})
#   %sum_8 : [num_users=1] = call_function[target=torch.ops.aten.sum.dim_IntList](args = (%mul_13, [0, 1, 2]), kwargs = {})
triton_red_fused__to_copy_native_layer_norm_backward_10 = async_compile.triton('triton_red_fused__to_copy_native_layer_norm_backward_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 256, 'r0_': 256},
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_native_layer_norm_backward_10', 'mutated_arg_names': [], 'optimize_mem': True, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_native_layer_norm_backward_10(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 176
    r0_numel = 192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 176*r0_1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        tmp3 = _tmp2 + tmp1
        _tmp2 = tl.where(r0_mask & xmask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_7, bitwise_right_shift, view, embedding, embedding_1, getitem_1, rsqrt, view_2, convert_element_type_13, convert_element_type_16, permute_2, permute_6, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (128, 33, 32, 176), (185856, 5632, 176, 1))
    assert_size_stride(primals_2, (128, 33, 32), (1056, 32, 1))
    assert_size_stride(primals_7, (176, ), (1, ))
    assert_size_stride(bitwise_right_shift, (128, 33, 32), (1056, 32, 1))
    assert_size_stride(view, (135168, 176), (176, 1))
    assert_size_stride(embedding, (128, 33, 32, 176), (185856, 5632, 176, 1))
    assert_size_stride(embedding_1, (128, 33, 32, 176), (185856, 5632, 176, 1))
    assert_size_stride(getitem_1, (128, 33, 32, 1), (1056, 32, 1, 1))
    assert_size_stride(rsqrt, (128, 33, 32, 1), (1056, 32, 1, 1))
    assert_size_stride(view_2, (135168, 176), (176, 1))
    assert_size_stride(convert_element_type_13, (135168, 16), (16, 1))
    assert_size_stride(convert_element_type_16, (135168, 16), (16, 1))
    assert_size_stride(permute_2, (16, 176), (176, 1))
    assert_size_stride(permute_6, (16, 176), (176, 1))
    assert_size_stride(tangents_1, (128, 33, 32), (1056, 32, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf21 = empty_strided_cuda((16, 176), (176, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward, aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_0.run(buf21, 2816, stream=stream0)
        buf23 = empty_strided_cuda((16, 176), (176, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_dense_backward_nll_loss_forward_0.run(buf23, 2816, stream=stream0)
        buf0 = empty_strided_cuda((135168, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(buf0, 2162688, stream=stream0)
        # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_2.run(primals_2, buf0, 135168, stream=stream0)
        buf4 = empty_strided_cuda((135168, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(buf4, 2162688, stream=stream0)
        # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward, aten.nll_loss_backward]
        stream0 = get_raw_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_3.run(bitwise_right_shift, buf4, 135168, stream=stream0)
        buf3 = convert_element_type_16; del convert_element_type_16  # reuse
        buf7 = convert_element_type_13; del convert_element_type_13  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy, cross_entropy_1], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward, aten._to_copy, aten._log_softmax_backward_data]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_backward_data__to_copy_nll_loss_backward_nll_loss_forward_4.run(buf3, buf7, buf0, primals_2, tangents_1, buf4, bitwise_right_shift, 135168, 16, stream=stream0)
        del buf0
        del buf4
        del primals_2
        del tangents_1
        buf9 = empty_strided_cuda((16, 176), (176, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (16, 135168), (1, 16), 0), view_2, out=buf9)
        del view_2
        buf26 = empty_strided_cuda((16, 176), (176, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf7, (16, 135168), (1, 16), 0), view, out=buf26)
        del view
        buf12 = empty_strided_cuda((16, 176), (176, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf9, buf12, 2816, stream=stream0)
        del buf9
        buf30 = empty_strided_cuda((16, 176), (176, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_5.run(buf26, buf30, 2816, stream=stream0)
        del buf26
        buf10 = empty_strided_cuda((1, 16, 1056), (16896, 1, 16), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_6.run(buf3, buf10, 16896, 128, stream=stream0)
        buf11 = empty_strided_cuda((1, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_7.run(buf10, buf11, 16, 1056, stream=stream0)
        buf27 = buf10; del buf10  # reuse
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_6.run(buf7, buf27, 16896, 128, stream=stream0)
        buf28 = empty_strided_cuda((1, 16), (16, 1), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.sum]
        stream0 = get_raw_stream(0)
        triton_red_fused_sum_7.run(buf27, buf28, 16, 1056, stream=stream0)
        del buf27
        buf8 = empty_strided_cuda((135168, 176), (176, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf3, permute_2, out=buf8)
        del buf3
        del permute_2
        buf25 = empty_strided_cuda((135168, 176), (176, 1), torch.float16)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf7, permute_6, out=buf25)
        del buf7
        del permute_6
        buf14 = embedding_1; del embedding_1  # reuse
        buf29 = embedding; del embedding  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy, add, mul, add_1, low_hidden], Original ATen: [aten.nll_loss_forward, aten._to_copy, aten.native_layer_norm_backward, aten.add, aten.mul, aten.native_layer_norm, aten.embedding_dense_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_embedding_dense_backward_mul_native_layer_norm_native_layer_norm_backward_nll_loss_forward_8.run(buf14, buf29, primals_1, getitem_1, rsqrt, buf8, primals_7, bitwise_right_shift, buf25, buf21, buf23, 135168, 176, stream=stream0)
        del bitwise_right_shift
        del buf25
        del getitem_1
        del primals_1
        del primals_7
        del rsqrt
        buf17 = empty_strided_cuda((176, 192), (1, 176), torch.float32)
        buf19 = empty_strided_cuda((176, 192), (1, 176), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_9.run(buf8, buf14, buf17, buf19, 33792, 704, stream=stream0)
        del buf14
        del buf8
        buf18 = empty_strided_cuda((176, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_10.run(buf17, buf18, 176, 192, stream=stream0)
        del buf17
        buf20 = empty_strided_cuda((176, ), (1, ), torch.float32)
        # Topologically Sorted Source Nodes: [], Original ATen: [aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_native_layer_norm_backward_10.run(buf19, buf20, 176, 192, stream=stream0)
        del buf19
    return (buf29, None, buf30, reinterpret_tensor(buf28, (16, ), (1, ), 0), buf23, buf21, buf18, buf20, buf12, reinterpret_tensor(buf11, (16, ), (1, ), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 33, 32, 176), (185856, 5632, 176, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 33, 32), (1056, 32, 1), device='cuda:0', dtype=torch.int64)
    primals_7 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    bitwise_right_shift = rand_strided((128, 33, 32), (1056, 32, 1), device='cuda:0', dtype=torch.int64)
    view = rand_strided((135168, 176), (176, 1), device='cuda:0', dtype=torch.float16)
    embedding = rand_strided((128, 33, 32, 176), (185856, 5632, 176, 1), device='cuda:0', dtype=torch.float32)
    embedding_1 = rand_strided((128, 33, 32, 176), (185856, 5632, 176, 1), device='cuda:0', dtype=torch.float32)
    getitem_1 = rand_strided((128, 33, 32, 1), (1056, 32, 1, 1), device='cuda:0', dtype=torch.float32)
    rsqrt = rand_strided((128, 33, 32, 1), (1056, 32, 1, 1), device='cuda:0', dtype=torch.float32)
    view_2 = rand_strided((135168, 176), (176, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_13 = rand_strided((135168, 16), (16, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_16 = rand_strided((135168, 16), (16, 1), device='cuda:0', dtype=torch.float16)
    permute_2 = rand_strided((16, 176), (176, 1), device='cuda:0', dtype=torch.float16)
    permute_6 = rand_strided((16, 176), (176, 1), device='cuda:0', dtype=torch.float16)
    tangents_1 = rand_strided((128, 33, 32), (1056, 32, 1), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_7, bitwise_right_shift, view, embedding, embedding_1, getitem_1, rsqrt, view_2, convert_element_type_13, convert_element_type_16, permute_2, permute_6, tangents_1])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
