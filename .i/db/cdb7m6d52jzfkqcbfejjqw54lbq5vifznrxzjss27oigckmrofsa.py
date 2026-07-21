# AOT ID: ['3_forward']
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


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\57\c57u5uv6mm2umis2l5h57hxreasj22c6bmeml26d4kh2te3u5pdu.py
# Topologically Sorted Source Nodes: [high_target, high_logits, embedding, add, mul, embedding_1, add_1, low_hidden, low_logits], Original ATen: [aten.bitwise_right_shift, aten._to_copy, aten.embedding, aten.add, aten.mul, aten.native_layer_norm]
# Source node to ATen node mapping:
#   add => add
#   add_1 => add_1
#   embedding => embedding
#   embedding_1 => embedding_1
#   high_logits => convert_element_type_2
#   high_target => bitwise_right_shift
#   low_hidden => add_2, add_3, mul_1, mul_2, rsqrt, sub, var_mean
#   low_logits => convert_element_type_8
#   mul => mul
# Graph fragment:
#   %bitwise_right_shift : [num_users=4] = call_function[target=torch.ops.aten.bitwise_right_shift.Tensor_Scalar](args = (%primals_2, 4), kwargs = {})
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_1, torch.float16), kwargs = {})
#   %embedding : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_5, %bitwise_right_shift), kwargs = {})
#   %add : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%embedding, 1.0), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%primals_1, %add), kwargs = {})
#   %embedding_1 : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_6, %bitwise_right_shift), kwargs = {})
#   %add_1 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul, %embedding_1), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_1, [3]), kwargs = {correction: 0, keepdim: True})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_2,), kwargs = {})
#   %sub : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_1, %getitem_1), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub, %rsqrt), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_1, %primals_7), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_2, %primals_8), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_3, torch.float16), kwargs = {})
triton_per_fused__to_copy_add_bitwise_right_shift_embedding_mul_native_layer_norm_0 = async_compile.triton('triton_per_fused__to_copy_add_bitwise_right_shift_embedding_mul_native_layer_norm_0', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr0': '*i64', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp16', 'out_ptr5': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]], (12,): [['tt.divisibility', 16]], (13,): [['tt.divisibility', 16]], (14,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_bitwise_right_shift_embedding_mul_native_layer_norm_0', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_bitwise_right_shift_embedding_mul_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    x0 = xindex
    r0_1 = r0_index
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (r0_1 + 176*x0), r0_mask, other=0.0)
    tmp39 = tl.load(in_ptr4 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp41 = tl.load(in_ptr5 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1, 1], 4, tl.int64)
    tmp2 = tmp0 >> tmp1
    tmp3 = tl.full([XBLOCK, R0_BLOCK], 16, tl.int32)
    tmp4 = tmp2 + tmp3
    tmp5 = tmp2 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp2)
    tl.device_assert((0 <= tmp6) & (tmp6 < 16), "index out of bounds: 0 <= tmp6 < 16")
    tmp8 = tl.load(in_ptr1 + (r0_1 + 176*tmp6), r0_mask, other=0.0)
    tmp9 = tl.load(in_ptr2 + (r0_1 + 176*tmp6), r0_mask, other=0.0)
    tmp11 = 1.0
    tmp12 = tmp8 + tmp11
    tmp13 = tmp10 * tmp12
    tmp14 = tmp13 + tmp9
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(r0_mask, tmp15, 0)
    tmp18 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp20 = tl.where(r0_mask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = tl.full([XBLOCK, 1], 176, tl.int32)
    tmp23 = tmp22.to(tl.float32)
    tmp24 = (tmp21 / tmp23)
    tmp25 = tmp15 - tmp24
    tmp26 = tmp25 * tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.where(r0_mask, tmp27, 0)
    tmp30 = tl.sum(tmp29, 1)[:, None]
    tmp31 = 176.0
    tmp32 = (tmp30 / tmp31)
    tmp33 = 1e-05
    tmp34 = tmp32 + tmp33
    tmp35 = libdevice.rsqrt(tmp34)
    tmp36 = tmp10.to(tl.float32)
    tmp37 = tmp14 - tmp24
    tmp38 = tmp37 * tmp35
    tmp40 = tmp38 * tmp39
    tmp42 = tmp40 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (r0_1 + 176*x0), tmp8, r0_mask)
    tl.store(out_ptr2 + (r0_1 + 176*x0), tmp9, r0_mask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp35, None)
    tl.store(out_ptr4 + (r0_1 + 176*x0), tmp36, r0_mask)
    tl.store(out_ptr5 + (r0_1 + 176*x0), tmp43, r0_mask)
    tl.store(out_ptr3 + (x0), tmp24, None)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\gm\cgmcofp3bnffbjnjfbmhbf7s2fzqu3qprmtfbwphtij2nw2xdix3.py
# Topologically Sorted Source Nodes: [high_logits], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   high_logits => convert_element_type_1, permute
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float16), kwargs = {})
#   %permute : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_1, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_1 = async_compile.triton('triton_poi_fused__to_copy_t_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4096}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\xe\cxeq3c6o2yzfch4ivrrvetolpmlssetpqndza6ny2vgwboccqenr.py
# Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax]
# Source node to ATen node mapping:
#   cross_entropy => convert_element_type_12, convert_element_type_13, log, sub_2
# Graph fragment:
#   %convert_element_type_12 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_4, torch.float32), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%convert_element_type_12, 1), kwargs = {})
#   %sub_tensor_1 : [num_users=2] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_12, %getitem_4), kwargs = {})
#   %log : [num_users=1] = call_function[target=torch.ops.aten.log.default](args = (%getitem_5,), kwargs = {})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%sub_tensor_1, %log), kwargs = {})
#   %convert_element_type_13 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%sub_2, torch.float16), kwargs = {})
triton_per_fused__log_softmax_2 = async_compile.triton('triton_per_fused__log_softmax_2', '''
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
    triton_meta={'signature': {'in_out_ptr0': '*fp16', 'in_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__log_softmax_2', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 4, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__log_softmax_2(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 16*x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (r0_1), None, eviction_policy='evict_last')
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp9 = triton_helpers.max2(tmp7, 1)[:, None]
    tmp10 = tmp5 - tmp9
    tmp11 = tl_math.exp(tmp10)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.sum(tmp12, 1)[:, None]
    tmp15 = tmp4 - tmp9
    tmp16 = tl_math.log(tmp14)
    tmp17 = tmp15 - tmp16
    tmp18 = tmp17.to(tl.float32)
    tl.store(in_out_ptr0 + (r0_1 + 16*x0), tmp18, None)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\d2\cd2tlbd3bdy2bxou56q26kp4brpo4upgall46ebnmimfgj42urg5.py
# Topologically Sorted Source Nodes: [nll], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   nll => add_4
# Graph fragment:
#   %add_4 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_6, %view_9), kwargs = {})
triton_poi_fused_add_3 = async_compile.triton('triton_poi_fused_add_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp16', 'in_ptr2': '*i64', 'in_ptr3': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 135168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp15 = tl.load(in_ptr2 + (x2), None)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tl.full([XBLOCK], 16, tl.int32)
    tmp6 = tmp4 + tmp5
    tmp7 = tmp4 < 0
    tmp8 = tl.where(tmp7, tmp6, tmp4)
    tl.device_assert((0 <= tmp8) & (tmp8 < 16), "index out of bounds: 0 <= tmp8 < 16")
    tmp10 = tl.load(in_ptr1 + (tmp8 + 16*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = -tmp11
    tmp13 = 0.0
    tmp14 = tl.where(tmp2, tmp12, tmp13)
    tmp16 = tl.full([1], 15, tl.int64)
    tmp17 = tmp15 & tmp16
    tmp18 = tmp17 != tmp1
    tmp19 = tl.where(tmp18, tmp17, tmp3)
    tmp20 = tmp19 + tmp5
    tmp21 = tmp19 < 0
    tmp22 = tl.where(tmp21, tmp20, tmp19)
    tl.device_assert((0 <= tmp22) & (tmp22 < 16), "index out of bounds: 0 <= tmp22 < 16")
    tmp24 = tl.load(in_ptr3 + (tmp22 + 16*x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tmp24.to(tl.float32)
    tmp26 = -tmp25
    tmp27 = tl.where(tmp18, tmp26, tmp13)
    tmp28 = tmp14 + tmp27
    tl.store(out_ptr0 + (x2), tmp28, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10 = args
    args.clear()
    assert_size_stride(primals_1, (128, 33, 32, 176), (185856, 5632, 176, 1))
    assert_size_stride(primals_2, (128, 33, 32), (1056, 32, 1))
    assert_size_stride(primals_3, (16, 176), (176, 1))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (16, 176), (176, 1))
    assert_size_stride(primals_6, (16, 176), (176, 1))
    assert_size_stride(primals_7, (176, ), (1, ))
    assert_size_stride(primals_8, (176, ), (1, ))
    assert_size_stride(primals_9, (16, 176), (176, 1))
    assert_size_stride(primals_10, (16, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 33, 32), (1056, 32, 1), torch.int64)
        buf4 = empty_strided_cuda((128, 33, 32, 176), (185856, 5632, 176, 1), torch.float32)
        buf5 = empty_strided_cuda((128, 33, 32, 176), (185856, 5632, 176, 1), torch.float32)
        buf6 = empty_strided_cuda((128, 33, 32, 1), (1056, 32, 1, 1), torch.float32)
        buf7 = empty_strided_cuda((128, 33, 32, 1), (1056, 32, 1, 135168), torch.float32)
        buf9 = reinterpret_tensor(buf7, (128, 33, 32, 1), (1056, 32, 1, 1), 0); del buf7  # reuse
        buf1 = empty_strided_cuda((128, 33, 32, 176), (185856, 5632, 176, 1), torch.float16)
        buf10 = empty_strided_cuda((128, 33, 32, 176), (185856, 5632, 176, 1), torch.float16)
        # Topologically Sorted Source Nodes: [high_target, high_logits, embedding, add, mul, embedding_1, add_1, low_hidden, low_logits], Original ATen: [aten.bitwise_right_shift, aten._to_copy, aten.embedding, aten.add, aten.mul, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_bitwise_right_shift_embedding_mul_native_layer_norm_0.run(buf9, primals_2, primals_5, primals_6, primals_1, primals_7, primals_8, buf0, buf4, buf5, buf6, buf1, buf10, 135168, 176, stream=stream0)
        del primals_5
        del primals_6
        del primals_8
        buf2 = empty_strided_cuda((176, 16), (1, 176), torch.float16)
        # Topologically Sorted Source Nodes: [high_logits], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_1.run(primals_3, buf2, 2816, stream=stream0)
        del primals_3
        buf3 = empty_strided_cuda((135168, 16), (16, 1), torch.float16)
        # Topologically Sorted Source Nodes: [high_logits], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf1, (135168, 176), (176, 1), 0), buf2, out=buf3)
        buf11 = empty_strided_cuda((176, 16), (1, 176), torch.float16)
        # Topologically Sorted Source Nodes: [low_logits], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_1.run(primals_9, buf11, 2816, stream=stream0)
        del primals_9
        buf12 = empty_strided_cuda((135168, 16), (16, 1), torch.float16)
        # Topologically Sorted Source Nodes: [low_logits], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf10, (135168, 176), (176, 1), 0), buf11, out=buf12)
        buf15 = buf3; del buf3  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_2.run(buf15, primals_4, 135168, 16, stream=stream0)
        del primals_4
        buf18 = buf12; del buf12  # reuse
        # Topologically Sorted Source Nodes: [cross_entropy_1], Original ATen: [aten._log_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__log_softmax_2.run(buf18, primals_10, 135168, 16, stream=stream0)
        del primals_10
        buf19 = empty_strided_cuda((128, 33, 32), (1056, 32, 1), torch.float32)
        # Topologically Sorted Source Nodes: [nll], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_3.run(buf0, buf15, primals_2, buf18, buf19, 135168, stream=stream0)
    return (buf19, primals_1, primals_2, primals_7, buf0, reinterpret_tensor(buf1, (135168, 176), (176, 1), 0), buf4, buf5, buf6, buf9, reinterpret_tensor(buf10, (135168, 176), (176, 1), 0), buf15, buf18, reinterpret_tensor(buf11, (16, 176), (176, 1), 0), reinterpret_tensor(buf2, (16, 176), (176, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 33, 32, 176), (185856, 5632, 176, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, 33, 32), (1056, 32, 1), device='cuda:0', dtype=torch.int64)
    primals_3 = rand_strided((16, 176), (176, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((16, 176), (176, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((16, 176), (176, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, 176), (176, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
