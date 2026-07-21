# AOT ID: ['0_forward']
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


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\ag\cagym3vyqwxlco3lkaj4vfj77htwxc7wwx5b3h25iri7rr6akpiy.py
# Topologically Sorted Source Nodes: [patch_h], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   patch_h => convert_element_type_2
# Graph fragment:
#   %convert_element_type_2 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_2, torch.float16), kwargs = {})
triton_poi_fused__to_copy_0 = async_compile.triton('triton_poi_fused__to_copy_0', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 4194304}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'in_ptr1': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_0', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3244032
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x2 = xindex
    x0 = (xindex % 768)
    tmp0 = tl.load(in_ptr0 + (x2 // 24), None, eviction_policy='evict_last')
    tmp1 = tl.full([XBLOCK], 256, tl.int32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp0 < 0
    tmp4 = tl.where(tmp3, tmp2, tmp0)
    tl.device_assert((0 <= tmp4) & (tmp4 < 256), "index out of bounds: 0 <= tmp4 < 256")
    tmp6 = tl.load(in_ptr1 + (24*tmp4 + ((x0 % 24))), None)
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp7, None)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\bm\cbmvfz545jhjacv2hiiujuzdwjsc4cn6qgkwtlwmsxg7wntdv2ny.py
# Topologically Sorted Source Nodes: [patch_h], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   patch_h => convert_element_type_1
# Graph fragment:
#   %convert_element_type_1 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_3, torch.float16), kwargs = {})
triton_poi_fused__to_copy_1 = async_compile.triton('triton_poi_fused__to_copy_1', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_1', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\vv\cvvdupo7x2bep7vlwx66hc53h4iun62vsksaiigmv6gzolfdmdra.py
# Topologically Sorted Source Nodes: [patch_h], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   patch_h => convert_element_type
# Graph fragment:
#   %convert_element_type : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_4, torch.float16), kwargs = {})
triton_poi_fused__to_copy_2 = async_compile.triton('triton_poi_fused__to_copy_2', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 256}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_2', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\jd\cjdsnaat2homdpxr7d5ytxlxoazm2kse5ja3pahww2n5pk7oiw2d.py
# Topologically Sorted Source Nodes: [positions], Original ATen: [aten.arange]
# Source node to ATen node mapping:
#   positions => iota
# Graph fragment:
#   %iota : [num_users=6] = call_function[target=torch.ops.prims.iota.default](args = (33,), kwargs = {start: 0, step: 1, dtype: torch.int64, device: cuda:0, requires_grad: False})
triton_poi_fused_arange_3 = async_compile.triton('triton_poi_fused_arange_3', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_arange_3', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 0, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_arange_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 33
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = x0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\4z\c4z3byaodk2dvh6zuztapgwg7z4smyqkhjk35nrfflvpdpzslvl6.py
# Topologically Sorted Source Nodes: [embedding_1], Original ATen: [aten.embedding]
# Source node to ATen node mapping:
#   embedding_1 => embedding_1
# Graph fragment:
#   %embedding_1 : [num_users=2] = call_function[target=torch.ops.aten.embedding.default](args = (%primals_5, %iota), kwargs = {})
triton_poi_fused_embedding_4 = async_compile.triton('triton_poi_fused_embedding_4', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 8192}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_4', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_embedding_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5544
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\tl\ctlgt34wz7dxcqarbq55ib2v3z5m4iixt5fflatcomlwjrlpfcnz.py
# Topologically Sorted Source Nodes: [patch_h_1, normalized, linear_1], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_8
#   normalized => add_1, add_2, mul, mul_1, rsqrt, sub_1, var_mean
#   patch_h_1 => add
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %unsqueeze), kwargs = {})
#   %var_mean : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_1 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem, 1e-05), kwargs = {})
#   %rsqrt : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_1,), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_6), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_7), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float16), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_5 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_5', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_5', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 4, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_5(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4224
    r0_numel = 168
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 33)
    x1 = xindex // 33
    tmp0 = tl.load(in_ptr0 + (r0_2 + 168*x3), xmask & r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_2 + 168*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
    tmp27 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp6 = tl.where(r0_mask & xmask, tmp4, 0)
    tmp7 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp9 = tl.where(r0_mask & xmask, tmp7, 0)
    tmp10 = tl.sum(tmp9, 1)[:, None]
    tmp11 = tl.full([XBLOCK, 1], 168, tl.int32)
    tmp12 = tmp11.to(tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp4 - tmp13
    tmp15 = tmp14 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(r0_mask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 168.0
    tmp21 = (tmp19 / tmp20)
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp3 - tmp13
    tmp26 = tmp25 * tmp24
    tmp28 = tmp26 * tmp27
    tmp30 = tmp28 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp24, xmask)
    tl.store(out_ptr1 + (r0_2 + 168*x0 + 5568*x1), tmp31, xmask & r0_mask)
    tl.store(out_ptr0 + (x3), tmp13, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\qg\cqgn2ckyl4tpoxwctelfrjhmna64fcrgqsr74o5sbknqaohevbc3.py
# Topologically Sorted Source Nodes: [patch_h_1, normalized, linear_1], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.view]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_8, view_5
#   normalized => add_2, mul, mul_1, sub_1
#   patch_h_1 => add
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %unsqueeze), kwargs = {})
#   %sub_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add, %getitem_1), kwargs = {})
#   %mul : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_1, %rsqrt), kwargs = {})
#   %mul_1 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul, %primals_6), kwargs = {})
#   %add_2 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_1, %primals_7), kwargs = {})
#   %convert_element_type_8 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_2, torch.float16), kwargs = {})
#   %view_5 : [num_users=2] = call_function[target=torch.ops.aten.reshape.default](args = (%convert_element_type_8, [4224, 168]), kwargs = {})
triton_poi_fused__to_copy_add_native_layer_norm_view_6 = async_compile.triton('triton_poi_fused__to_copy_add_native_layer_norm_view_6', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_native_layer_norm_view_6', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_native_layer_norm_view_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 709632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 168)
    x1 = xindex // 168
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 168*((x1 % 33)) + 5568*(x1 // 33)), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\ze\czeqg7cgdyl2e5jq44s7okzunyhnyuojgtddcruhbgsfm6hhcynp.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_7
# Graph fragment:
#   %convert_element_type_7 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_8, torch.float16), kwargs = {})
triton_poi_fused__to_copy_7 = async_compile.triton('triton_poi_fused__to_copy_7', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_7', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 84672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\eb\cebkskvvywv4w32vn7hmv77bygtckdsupwg6kcqrlpevuqac22kz.py
# Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   linear_1 => convert_element_type_6
# Graph fragment:
#   %convert_element_type_6 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_9, torch.float16), kwargs = {})
triton_poi_fused__to_copy_8 = async_compile.triton('triton_poi_fused__to_copy_8', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 512}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_8', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\2j\c2j5vyz2fwadt7ddfa3s2nm32aw4wb4mskrqntnozqwfbl5bjwnq.py
# Topologically Sorted Source Nodes: [query_2], Original ATen: [aten._to_copy, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   query_2 => convert_element_type_12, mean, pow_1
# Graph fragment:
#   %convert_element_type_12 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_2, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_12, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [3], True), kwargs = {})
triton_per_fused__to_copy_mean_pow_9 = async_compile.triton('triton_per_fused__to_copy_mean_pow_9', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mean_pow_9', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_mean_pow_9(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16896
    r0_numel = 42
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 42*x0 + 504*x1), xmask & r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\e5\ce5okbrl46lvmfgssdcbidpwtskczogneoeoyoaz2prznwbyyhle.py
# Topologically Sorted Source Nodes: [query_2], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
# Source node to ATen node mapping:
#   query_2 => add_3, convert_element_type_12, mean, pow_1, rsqrt_1
# Graph fragment:
#   %convert_element_type_12 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_2, torch.float32), kwargs = {})
#   %pow_1 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_12, 2), kwargs = {})
#   %mean : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_1, [3], True), kwargs = {})
#   %add_3 : [num_users=1] = call_function[target=torch.ops.aten.add.Scalar](args = (%mean, 1.1920928955078125e-07), kwargs = {})
#   %rsqrt_1 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_3,), kwargs = {})
triton_poi_fused__to_copy_add_mean_pow_rsqrt_10 = async_compile.triton('triton_poi_fused__to_copy_add_mean_pow_rsqrt_10', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 512, 'x': 64}, tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_mean_pow_rsqrt_10', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_mean_pow_rsqrt_10(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 33
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 4)
    y1 = yindex // 4
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 4*x2 + 132*y1), ymask & xmask, eviction_policy='evict_last')
    tmp1 = 42.0
    tmp2 = (tmp0 / tmp1)
    tmp3 = 1.1920928955078125e-07
    tmp4 = tmp2 + tmp3
    tmp5 = libdevice.rsqrt(tmp4)
    tl.store(out_ptr0 + (x2 + 33*y3), tmp5, ymask & xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\y5\cy5jto4zpmq5phflwrb6gr4pjahl2csgarvf3rkcbpbmeldebxxl.py
# Topologically Sorted Source Nodes: [key_2], Original ATen: [aten._to_copy, aten.pow, aten.mean]
# Source node to ATen node mapping:
#   key_2 => convert_element_type_14, mean_1, pow_2
# Graph fragment:
#   %convert_element_type_14 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_3, torch.float32), kwargs = {})
#   %pow_2 : [num_users=1] = call_function[target=torch.ops.aten.pow.Tensor_Scalar](args = (%convert_element_type_14, 2), kwargs = {})
#   %mean_1 : [num_users=1] = call_function[target=torch.ops.aten.mean.dim](args = (%pow_2, [3], True), kwargs = {})
triton_per_fused__to_copy_mean_pow_11 = async_compile.triton('triton_per_fused__to_copy_mean_pow_11', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_mean_pow_11', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 1, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_mean_pow_11(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16896
    r0_numel = 42
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 4)
    x1 = xindex // 4
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (168 + r0_2 + 42*x0 + 504*x1), xmask & r0_mask, other=0.0).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1 * tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp6, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\tb\ctbrs3ywazk2dntamydal7taf3mdmxfqgxn3jtpkcjifwcot3kii.py
# Topologically Sorted Source Nodes: [attended], Original ATen: [aten.ones, aten.tril]
# Source node to ATen node mapping:
#   attended => full_default, le, logical_and, sub_2
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([33, 33], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_2 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_3, %unsqueeze_4), kwargs = {})
#   %le : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub_2, 0), kwargs = {})
#   %logical_and : [num_users=2] = call_function[target=torch.ops.aten.logical_and.default](args = (%le, %full_default), kwargs = {})
triton_poi_fused_ones_tril_12 = async_compile.triton('triton_poi_fused_ones_tril_12', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2048}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i1', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_ones_tril_12', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_ones_tril_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1089
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 33)
    x1 = xindex // 33
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tmp2 <= tmp3
    tmp5 = tl.full([1], True, tl.int1)
    tmp6 = tmp4 & tmp5
    tl.store(out_ptr0 + (x2), tmp6, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\n6\cn6w3juz2opnwz4amfyaawpb55rdmxmxfezlaa4ys4oyc6eoirwu.py
# Topologically Sorted Source Nodes: [query_2, attended], Original ATen: [aten._to_copy, aten.mul, aten.clone]
# Source node to ATen node mapping:
#   attended => clone, mul_4
#   query_2 => convert_element_type_12, mul_2
# Graph fragment:
#   %convert_element_type_12 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_2, torch.float32), kwargs = {})
#   %mul_2 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_12, %rsqrt_1), kwargs = {})
#   %mul_4 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%mul_2, 0.39281465090051304), kwargs = {})
#   %clone : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused__to_copy_clone_mul_13 = async_compile.triton('triton_poi_fused__to_copy_clone_mul_13', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_mul_13', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_mul_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 709632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 42)
    x1 = ((xindex // 42) % 33)
    x2 = ((xindex // 1386) % 4)
    x3 = xindex // 5544
    x4 = xindex // 42
    x5 = (xindex % 1386)
    x6 = xindex // 1386
    tmp0 = tl.load(in_ptr0 + (x0 + 42*x2 + 504*x1 + 16632*x3), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = 0.39281465090051304
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + (x5 + 1408*x6), tmp5, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\kg\ckgwo67ifajzywaswf7kblw4gbsm7vxutcvflp5abpjvnqmrholp.py
# Topologically Sorted Source Nodes: [attended], Original ATen: [aten.mul, aten.clone]
# Source node to ATen node mapping:
#   attended => clone_1, mul_5
# Graph fragment:
#   %mul_5 : [num_users=1] = call_function[target=torch.ops.aten.mul.Scalar](args = (%permute_5, 0.39281465090051304), kwargs = {})
#   %clone_1 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_1,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_mul_14 = async_compile.triton('triton_poi_fused_clone_mul_14', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'y': 32768, 'x': 64}, tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'out_ptr0': '*fp32', 'ynumel': 'i32', 'xnumel': 'i32', 'YBLOCK': 'constexpr', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid2D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_mul_14', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_mul_14(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 21504
    xnumel = 33
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = tl.full([XBLOCK, YBLOCK], True, tl.int1)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y2 = yindex // 168
    y4 = (yindex % 168)
    y5 = yindex // 42
    y0 = (yindex % 42)
    tmp0 = tl.load(in_ptr0 + (168 + y4 + 504*x3 + 16632*y2), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (x3 + 33*y5), xmask, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = 0.39281465090051304
    tmp5 = tmp3 * tmp4
    tl.store(out_ptr0 + (x3 + 33*y0 + 1408*y5), tmp5, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\nt\cntt3flujiwo3wquykm2fyz53x2z2ztuyspthu5hu2qciaa73y6b.py
# Topologically Sorted Source Nodes: [attended], Original ATen: [aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
# Source node to ATen node mapping:
#   attended => add_5, any_1, div, eq, full_default_1, full_default_2, full_default_3, logical_not, logical_not_1, where_1, where_2
# Graph fragment:
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_1 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and, %full_default_2, %full_default_1), kwargs = {})
#   %add_5 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_10, %where_1), kwargs = {})
#   %prepare_softmax_online_default_1 : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_5, -1), kwargs = {})
#   %sub_tensor_1 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_5, %getitem_18), kwargs = {})
#   %exp_default_1 : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor_1,), kwargs = {})
#   %div : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default_1, %getitem_19), kwargs = {})
#   %eq : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_5, -inf), kwargs = {})
#   %logical_not : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq,), kwargs = {})
#   %any_1 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not, -1, True), kwargs = {})
#   %logical_not_1 : [num_users=2] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_1,), kwargs = {})
#   %full_default_3 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([128, 4, 33, 33], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %where_2 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_1, %full_default_3, %div), kwargs = {})
triton_per_fused__safe_softmax_add_scalar_tensor_where_15 = async_compile.triton('triton_per_fused__safe_softmax_add_scalar_tensor_where_15', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*i1', 'in_ptr0': '*fp32', 'in_ptr1': '*i1', 'out_ptr0': '*fp32', 'out_ptr1': '*fp32', 'out_ptr2': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_add_scalar_tensor_where_15', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 5, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__safe_softmax_add_scalar_tensor_where_15(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16896
    r0_numel = 33
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 33)
    x1 = xindex // 33
    tmp0 = tl.load(in_ptr0 + (r0_2 + 33*x3), xmask & r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 33*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.int1)
    tmp2 = 0.0
    tmp3 = float("-inf")
    tmp4 = tl.where(tmp1, tmp2, tmp3)
    tmp5 = tmp0 + tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp8 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, float("-inf"))
    tmp11 = triton_helpers.max2(tmp10, 1)[:, None]
    tmp12 = tmp6 - tmp11
    tmp13 = tl_math.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = tmp5 == tmp3
    tmp19 = tmp18 == 0
    tmp20 = tmp19.to(tl.int64)
    tmp21 = (tmp20 != 0)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask & xmask, tmp22, False)
    tmp25 = triton_helpers.any(tmp24, 1)[:, None]
    tmp26 = tmp25 == 0
    tmp27 = tmp5 - tmp11
    tmp28 = tl_math.exp(tmp27)
    tmp29 = (tmp28 / tmp17)
    tmp30 = tl.where(tmp26, tmp2, tmp29)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp26, xmask)
    tl.store(out_ptr2 + (r0_2 + 33*x0 + 1120*x1), tmp30, xmask & r0_mask)
    tl.store(out_ptr0 + (x3), tmp11, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\3p\c3psaqjgxxbkbaufrjit2xkvsp454gem7szkgy67c66h2rs7jskt.py
# Topologically Sorted Source Nodes: [attended], Original ATen: [aten._to_copy, aten.clone]
# Source node to ATen node mapping:
#   attended => clone_2, convert_element_type_18
# Graph fragment:
#   %convert_element_type_18 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_4, torch.float32), kwargs = {})
#   %clone_2 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%expand_3,), kwargs = {memory_format: torch.contiguous_format})
#   %convert_element_type_196 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%permute_80, torch.float16), kwargs = {})
triton_poi_fused__to_copy_clone_16 = async_compile.triton('triton_poi_fused__to_copy_clone_16', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'out_ptr0': '*fp32', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_clone_16', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_clone_16(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 709632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 42)
    x1 = ((xindex // 42) % 33)
    x2 = ((xindex // 1386) % 4)
    x3 = xindex // 5544
    x4 = (xindex % 1386)
    x5 = xindex // 1386
    tmp0 = tl.load(in_ptr0 + (336 + x0 + 42*x2 + 504*x1 + 16632*x3), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tl.store(out_ptr0 + (x4 + 1408*x5), tmp1, xmask)
    tl.store(out_ptr1 + (x4 + 1408*x5), tmp2, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\lw\clwwtobaaskbry5sjx7kogwkgk7vatfic3ob6kq55dqak4vzfrf3.py
# Topologically Sorted Source Nodes: [attended_1], Original ATen: [aten.clone]
# Source node to ATen node mapping:
#   attended_1 => clone_3
# Graph fragment:
#   %clone_3 : [num_users=1] = call_function[target=torch.ops.aten.clone.default](args = (%permute_6,), kwargs = {memory_format: torch.contiguous_format})
triton_poi_fused_clone_17 = async_compile.triton('triton_poi_fused_clone_17', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_17', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_clone_17(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 709632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 42)
    x1 = ((xindex // 42) % 4)
    x2 = ((xindex // 168) % 33)
    x3 = xindex // 5544
    x4 = (xindex % 5544)
    tmp0 = tl.load(in_ptr0 + (x0 + 42*x2 + 1386*x1 + 5544*x3), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x4 + 5568*x3), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\iq\ciq2fu7d4v64secm3gtf5aqyzoeoptvx6cmt3p5am445qr6xqjoa.py
# Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   linear_2 => convert_element_type_22, permute_7
# Graph fragment:
#   %convert_element_type_22 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_10, torch.float16), kwargs = {})
#   %permute_7 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_22, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_18 = async_compile.triton('triton_poi_fused__to_copy_t_18', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 32768}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_18', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 28224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\2l\c2l3gdw6j7yudnpjbtoqt72yopdwjzwkqzdig6jc4fmsts4lchwm.py
# Topologically Sorted Source Nodes: [patch_h_1, h, normalized_1, gate], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   gate => convert_element_type_27
#   h => add_6
#   normalized_1 => add_7, add_8, mul_6, mul_7, rsqrt_3, sub_4, var_mean_1
#   patch_h_1 => add
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %unsqueeze), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_16), kwargs = {})
#   %var_mean_1 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_6, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_7 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_5, 1e-05), kwargs = {})
#   %rsqrt_3 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_7,), kwargs = {})
#   %sub_4 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_6, %getitem_6), kwargs = {})
#   %mul_6 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_4, %rsqrt_3), kwargs = {})
#   %mul_7 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_6, %primals_12), kwargs = {})
#   %add_8 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_7, %primals_13), kwargs = {})
#   %convert_element_type_27 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_8, torch.float16), kwargs = {})
#   %div_7 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_3, 168), kwargs = {})
triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19 = async_compile.triton('triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'in_ptr5': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 4, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4224
    r0_numel = 168
    R0_BLOCK: tl.constexpr = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 33)
    x1 = xindex // 33
    tmp0 = tl.load(in_ptr0 + (r0_2 + 168*x3), xmask & r0_mask, other=0.0).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (r0_2 + 168*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
    tmp4 = tl.load(in_ptr2 + (r0_2 + 168*x3), xmask & r0_mask, other=0.0).to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp33 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp3 + tmp8
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, R0_BLOCK])
    tmp12 = tl.where(r0_mask & xmask, tmp10, 0)
    tmp13 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp15 = tl.where(r0_mask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tl.full([XBLOCK, 1], 168, tl.int32)
    tmp18 = tmp17.to(tl.float32)
    tmp19 = (tmp16 / tmp18)
    tmp20 = tmp10 - tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(r0_mask & xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None]
    tmp26 = tmp9 - tmp19
    tmp27 = 168.0
    tmp28 = (tmp25 / tmp27)
    tmp29 = 1e-05
    tmp30 = tmp28 + tmp29
    tmp31 = libdevice.rsqrt(tmp30)
    tmp32 = tmp26 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp38 = 0.005952380952380952
    tmp39 = tmp31 * tmp38
    tl.store(out_ptr2 + (r0_2 + 168*x0 + 5568*x1), tmp32, xmask & r0_mask)
    tl.store(out_ptr3 + (r0_2 + 168*x0 + 5568*x1), tmp37, xmask & r0_mask)
    tl.store(out_ptr4 + (x3), tmp39, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\my\cmyc5qmyi5smbzvuormnsqrwqvb2qhgiypwqdwjj2xhdyycyf6mw.py
# Topologically Sorted Source Nodes: [gate], Original ATen: [aten._to_copy, aten.t]
# Source node to ATen node mapping:
#   gate => convert_element_type_26, permute_8
# Graph fragment:
#   %convert_element_type_26 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_14, torch.float16), kwargs = {})
#   %permute_8 : [num_users=2] = call_function[target=torch.ops.aten.permute.default](args = (%convert_element_type_26, [1, 0]), kwargs = {})
triton_poi_fused__to_copy_t_20 = async_compile.triton('triton_poi_fused__to_copy_t_20', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 131072}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_t_20', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_t_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 75264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\op\copinuy7tcoz6kdkl4scgwuvaebhvh3ljmpaowrgcytqzeflfkh5.py
# Topologically Sorted Source Nodes: [silu, mul], Original ATen: [aten.silu, aten.mul]
# Source node to ATen node mapping:
#   mul => mul_9
#   silu => convert_element_type_34, convert_element_type_35, mul_8, sigmoid
# Graph fragment:
#   %convert_element_type_34 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_18, torch.float32), kwargs = {})
#   %sigmoid : [num_users=1] = call_function[target=torch.ops.aten.sigmoid.default](args = (%convert_element_type_34,), kwargs = {})
#   %mul_8 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_34, %sigmoid), kwargs = {})
#   %convert_element_type_35 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%mul_8, torch.float16), kwargs = {})
#   %mul_9 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%convert_element_type_35, %view_20), kwargs = {})
triton_poi_fused_mul_silu_21 = async_compile.triton('triton_poi_fused_mul_silu_21', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 2097152}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_21', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 2, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_mul_silu_21(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1892352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, None)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\ej\cejjmvcqjntwpmhgzseb4i5qej23775g7ncyfhqo7nwtej4oq236.py
# Topologically Sorted Source Nodes: [patch_h_1, h, h_1, normalized_2, linear_6], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   h => add_6
#   h_1 => add_9
#   linear_6 => convert_element_type_41
#   normalized_2 => add_10, add_11, mul_10, mul_11, rsqrt_4, sub_5, var_mean_2
#   patch_h_1 => add
# Graph fragment:
#   %add : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_4, %unsqueeze), kwargs = {})
#   %add_6 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add, %view_16), kwargs = {})
#   %add_9 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_6, %view_22), kwargs = {})
#   %var_mean_2 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_9, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_10 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_7, 1e-05), kwargs = {})
#   %rsqrt_4 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_10,), kwargs = {})
#   %sub_5 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_9, %getitem_8), kwargs = {})
#   %mul_10 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_5, %rsqrt_4), kwargs = {})
#   %mul_11 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_10, %primals_17), kwargs = {})
#   %add_11 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_11, %primals_18), kwargs = {})
#   %convert_element_type_41 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_11, torch.float16), kwargs = {})
#   %div_6 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_4, 168), kwargs = {})
triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_22 = async_compile.triton('triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_22', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp16', 'in_ptr3': '*fp32', 'in_ptr4': '*fp16', 'in_ptr5': '*fp32', 'in_ptr6': '*fp32', 'out_ptr0': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp16', 'out_ptr5': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]], (10,): [['tt.divisibility', 16]], (11,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_22', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr3, out_ptr4, out_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4224
    r0_numel = 168
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 33)
    x1 = xindex // 33
    tmp14_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp14_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp14_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 168*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr1 + (r0_2 + 168*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.load(in_ptr2 + (r0_2 + 168*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr4 + (r0_2 + 168*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp1 + tmp2
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp4 + tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp3 + tmp8
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp9 + tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp14_mean_next, tmp14_m2_next, tmp14_weight_next = triton_helpers.welford_reduce(
            tmp13, tmp14_mean, tmp14_m2, tmp14_weight, roffset == 0
        )
        tmp14_mean = tl.where(r0_mask & xmask, tmp14_mean_next, tmp14_mean)
        tmp14_m2 = tl.where(r0_mask & xmask, tmp14_m2_next, tmp14_m2)
        tmp14_weight = tl.where(r0_mask & xmask, tmp14_weight_next, tmp14_weight)
        tl.store(out_ptr0 + (r0_2 + 168*x0 + 5568*x1), tmp12, xmask & r0_mask)
    tmp17, tmp18, tmp19 = triton_helpers.welford(tmp14_mean, tmp14_m2, tmp14_weight, 1)
    tmp14 = tmp17[:, None]
    tmp15 = tmp18[:, None]
    tmp16 = tmp19[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp20 = tl.load(out_ptr0 + (r0_2 + 168*x0 + 5568*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp28 = tl.load(in_ptr5 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr6 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp21 = tmp20 - tmp14
        tmp22 = 168.0
        tmp23 = (tmp15 / tmp22)
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tmp32 = tmp31.to(tl.float32)
        tl.store(out_ptr3 + (r0_2 + 168*x0 + 5568*x1), tmp27, xmask & r0_mask)
        tl.store(out_ptr4 + (r0_2 + 168*x0 + 5568*x1), tmp32, xmask & r0_mask)
    tmp33 = 168.0
    tmp34 = (tmp15 / tmp33)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = 0.005952380952380952
    tmp39 = tmp37 * tmp38
    tl.store(out_ptr5 + (x3), tmp39, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\62\c62nalclzgwx4e7de5ibozbusr5742wx6daptcp73wjgcevedutz.py
# Topologically Sorted Source Nodes: [attended, attended_2], Original ATen: [aten.ones, aten.scalar_tensor, aten._safe_softmax, aten.tril, aten.where, aten.add]
# Source node to ATen node mapping:
#   attended => full_default, full_default_1, full_default_2, full_default_3
#   attended_2 => add_14, any_2, div_1, eq_1, le_1, logical_and_1, logical_not_2, logical_not_3, sub_6, where_3, where_4
# Graph fragment:
#   %full_default : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([33, 33], True), kwargs = {dtype: torch.bool, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_1 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], -inf), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_2 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([], 0.0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %full_default_3 : [num_users=2] = call_function[target=torch.ops.aten.full.default](args = ([128, 4, 33, 33], 0), kwargs = {dtype: torch.float32, layout: torch.strided, device: cuda:0, pin_memory: False})
#   %sub_6 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%unsqueeze_3, %unsqueeze_6), kwargs = {})
#   %le_1 : [num_users=1] = call_function[target=torch.ops.aten.le.Scalar](args = (%sub_6, 0), kwargs = {})
#   %logical_and_1 : [num_users=1] = call_function[target=torch.ops.aten.logical_and.default](args = (%le_1, %full_default), kwargs = {})
#   %where_3 : [num_users=1] = call_function[target=torch.ops.aten.where.self](args = (%logical_and_1, %full_default_2, %full_default_1), kwargs = {})
#   %add_14 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%view_28, %where_3), kwargs = {})
#   %prepare_softmax_online_default : [num_users=2] = call_function[target=torch.ops.prims.prepare_softmax_online.default](args = (%add_14, -1), kwargs = {})
#   %sub_tensor : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_14, %getitem_16), kwargs = {})
#   %exp_default : [num_users=1] = call_function[target=torch.ops.aten.exp.default](args = (%sub_tensor,), kwargs = {})
#   %div_1 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%exp_default, %getitem_17), kwargs = {})
#   %eq_1 : [num_users=1] = call_function[target=torch.ops.aten.eq.Scalar](args = (%add_14, -inf), kwargs = {})
#   %logical_not_2 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%eq_1,), kwargs = {})
#   %any_2 : [num_users=1] = call_function[target=torch.ops.aten.any.dim](args = (%logical_not_2, -1, True), kwargs = {})
#   %logical_not_3 : [num_users=1] = call_function[target=torch.ops.aten.logical_not.default](args = (%any_2,), kwargs = {})
#   %where_4 : [num_users=2] = call_function[target=torch.ops.aten.where.self](args = (%logical_not_3, %full_default_3, %div_1), kwargs = {})
triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_23 = async_compile.triton('triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_23', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*i64', 'out_ptr3': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_23', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 5, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_23(in_ptr0, in_ptr1, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16896
    r0_numel = 33
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 33)
    x1 = xindex // 33
    tmp0 = tl.load(in_ptr0 + (r0_2 + 33*x3), xmask & r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp4 = tl.full([1, 1], 0, tl.int64)
    tmp5 = tmp3 <= tmp4
    tmp6 = tl.full([1, 1], True, tl.int1)
    tmp7 = tmp5 & tmp6
    tmp8 = 0.0
    tmp9 = float("-inf")
    tmp10 = tl.where(tmp7, tmp8, tmp9)
    tmp11 = tmp0 + tmp10
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
    tmp14 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(r0_mask & xmask, tmp14, float("-inf"))
    tmp17 = triton_helpers.max2(tmp16, 1)[:, None]
    tmp18 = tmp12 - tmp17
    tmp19 = tl_math.exp(tmp18)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
    tmp22 = tl.where(r0_mask & xmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = tmp11 == tmp9
    tmp25 = tmp24 == 0
    tmp26 = tmp25.to(tl.int64)
    tmp27 = (tmp26 != 0)
    tmp28 = tl.broadcast_to(tmp27, [XBLOCK, R0_BLOCK])
    tmp30 = tl.where(r0_mask & xmask, tmp28, False)
    tmp31 = triton_helpers.any(tmp30, 1)[:, None]
    tmp32 = tmp31 == 0
    tmp33 = tmp11 - tmp17
    tmp34 = tl_math.exp(tmp33)
    tmp35 = (tmp34 / tmp23)
    tmp36 = tl.where(tmp32, tmp8, tmp35)
    tl.store(out_ptr3 + (r0_2 + 33*x0 + 1120*x1), tmp36, xmask & r0_mask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\jo\cjoggaanzuuroniwpgg5n6xkmiiayiofpq6bvrnvsxtdbw2rw4nz.py
# Topologically Sorted Source Nodes: [h_2, normalized_3, gate_1], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
# Source node to ATen node mapping:
#   gate_1 => convert_element_type_60
#   h_2 => add_15
#   normalized_3 => add_16, add_17, mul_16, mul_17, rsqrt_7, sub_8, var_mean_3
# Graph fragment:
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_34), kwargs = {})
#   %var_mean_3 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%add_15, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_16 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_12, 1e-05), kwargs = {})
#   %rsqrt_7 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_16,), kwargs = {})
#   %sub_8 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%add_15, %getitem_13), kwargs = {})
#   %mul_16 : [num_users=2] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_8, %rsqrt_7), kwargs = {})
#   %mul_17 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_16, %primals_23), kwargs = {})
#   %add_17 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_17, %primals_24), kwargs = {})
#   %convert_element_type_60 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_17, torch.float16), kwargs = {})
#   %div_3 : [num_users=1] = call_function[target=torch.ops.aten.div.Tensor](args = (%rsqrt_7, 168), kwargs = {})
triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_24 = async_compile.triton('triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_24', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.reduction(
    size_hints={'x': 8192, 'r0_': 256},
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'in_ptr4': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp16', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr', 'R0_BLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_24', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 8, 'num_reduction': 2, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4224
    r0_numel = 168
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 33)
    x1 = xindex // 33
    x3 = xindex
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 168*x0 + 5568*x1), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 168*x3), xmask & r0_mask, eviction_policy='evict_last', other=0.0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp0 + tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(r0_mask & xmask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask & xmask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask & xmask, tmp8_weight_next, tmp8_weight)
    tmp11, tmp12, tmp13 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)
    tmp8 = tmp11[:, None]
    tmp9 = tmp12[:, None]
    tmp10 = tmp13[:, None]
    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp14 = tl.load(in_ptr0 + (r0_2 + 168*x0 + 5568*x1), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr1 + (r0_2 + 168*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0).to(tl.float32)
        tmp16 = tl.load(in_ptr2 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp28 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp30 = tl.load(in_ptr4 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp17 = tmp16.to(tl.float32)
        tmp18 = tmp15 + tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tmp14 + tmp19
        tmp21 = tmp20 - tmp8
        tmp22 = 168.0
        tmp23 = (tmp9 / tmp22)
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = libdevice.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp29 = tmp27 * tmp28
        tmp31 = tmp29 + tmp30
        tmp32 = tmp31.to(tl.float32)
        tl.store(out_ptr2 + (r0_2 + 168*x0 + 5568*x1), tmp27, xmask & r0_mask)
        tl.store(out_ptr3 + (r0_2 + 168*x0 + 5568*x1), tmp32, xmask & r0_mask)
    tmp33 = 168.0
    tmp34 = (tmp9 / tmp33)
    tmp35 = 1e-05
    tmp36 = tmp34 + tmp35
    tmp37 = libdevice.rsqrt(tmp36)
    tmp38 = 0.005952380952380952
    tmp39 = tmp37 * tmp38
    tl.store(out_ptr4 + (x3), tmp39, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\4q\c4qxoeo6djrq4wppwyx6lpmy47xtmoxvjujcvljkakdtzxdl6pop.py
# Topologically Sorted Source Nodes: [h_2, h_3, input_1], Original ATen: [aten.add, aten._to_copy]
# Source node to ATen node mapping:
#   h_2 => add_15
#   h_3 => add_18
#   input_1 => convert_element_type_74
# Graph fragment:
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_34), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_40), kwargs = {})
#   %convert_element_type_74 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_18, torch.float16), kwargs = {})
triton_poi_fused__to_copy_add_25 = async_compile.triton('triton_poi_fused__to_copy_add_25', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_25', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 4, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_add_25(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 709632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 5544
    x3 = (xindex % 5544)
    x4 = xindex
    x0 = (xindex % 168)
    tmp0 = tl.load(in_ptr0 + (x3 + 5568*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x4), xmask).to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x3 + 5568*x2), tmp10, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\u6\cu6hrt6bunvestg6bodlhzxk6dm42if5v5tf4fjldl725b5sd24r.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_73
# Graph fragment:
#   %convert_element_type_73 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_28, torch.float16), kwargs = {})
triton_poi_fused__to_copy_26 = async_compile.triton('triton_poi_fused__to_copy_26', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 16384}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_26', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_26(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10752
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\zu\czuci32fjpvr3juvebrrrb7eh3ek6f7jk4rl73tqgyvfk42bc4dr.py
# Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
# Source node to ATen node mapping:
#   input_1 => convert_element_type_72
# Graph fragment:
#   %convert_element_type_72 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%primals_29, torch.float16), kwargs = {})
triton_poi_fused__to_copy_27 = async_compile.triton('triton_poi_fused__to_copy_27', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 64}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'out_ptr0': '*fp16', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_27', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused__to_copy_27(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\zq\czqzhrv6ttbrcfungbbvpqfe3xfrfszen2ojpxjv2chiiwzads7s.py
# Topologically Sorted Source Nodes: [input_2, linear_12], Original ATen: [aten._to_copy, aten.native_layer_norm]
# Source node to ATen node mapping:
#   input_2 => add_19, add_20, convert_element_type_78, mul_20, mul_21, rsqrt_8, sub_9, var_mean_4
#   linear_12 => convert_element_type_81
# Graph fragment:
#   %convert_element_type_78 : [num_users=2] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%view_42, torch.float32), kwargs = {})
#   %var_mean_4 : [num_users=2] = call_function[target=torch.ops.aten.var_mean.correction](args = (%convert_element_type_78, [2]), kwargs = {correction: 0, keepdim: True})
#   %add_19 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%getitem_14, 1e-05), kwargs = {})
#   %rsqrt_8 : [num_users=2] = call_function[target=torch.ops.aten.rsqrt.default](args = (%add_19,), kwargs = {})
#   %sub_9 : [num_users=1] = call_function[target=torch.ops.aten.sub.Tensor](args = (%convert_element_type_78, %getitem_15), kwargs = {})
#   %mul_20 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%sub_9, %rsqrt_8), kwargs = {})
#   %mul_21 : [num_users=1] = call_function[target=torch.ops.aten.mul.Tensor](args = (%mul_20, %primals_30), kwargs = {})
#   %add_20 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%mul_21, %primals_31), kwargs = {})
#   %convert_element_type_81 : [num_users=1] = call_function[target=torch.ops.prims.convert_element_type.default](args = (%add_20, torch.float16), kwargs = {})
triton_per_fused__to_copy_native_layer_norm_28 = async_compile.triton('triton_per_fused__to_copy_native_layer_norm_28', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 8192, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp16', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'out_ptr0': '*fp32', 'out_ptr1': '*fp16', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_native_layer_norm_28', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 3, 'num_reduction': 4, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False}
)
@triton.jit
def triton_per_fused__to_copy_native_layer_norm_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4224
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), xmask, other=0.0).to(tl.float32)
    tmp25 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp4 = tl.where(xmask, tmp2, 0)
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp10 = tmp9.to(tl.float32)
    tmp11 = (tmp8 / tmp10)
    tmp12 = tmp2 - tmp11
    tmp13 = tmp12 * tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None]
    tmp18 = 64.0
    tmp19 = (tmp17 / tmp18)
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = libdevice.rsqrt(tmp21)
    tmp23 = tmp1 - tmp11
    tmp24 = tmp23 * tmp22
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = tmp28.to(tl.float32)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp22, xmask)
    tl.store(out_ptr1 + (r0_1 + 64*x0), tmp29, xmask)
    tl.store(out_ptr0 + (x0), tmp11, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\kj\ckjudg6rpro4iozv7aqdxp7fiuzvkbut7aaa6u7vsktbp3idu4qn.py
# Topologically Sorted Source Nodes: [h_2, h_3, add_5], Original ATen: [aten.add]
# Source node to ATen node mapping:
#   add_5 => add_21
#   h_2 => add_15
#   h_3 => add_18
# Graph fragment:
#   %add_15 : [num_users=3] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_9, %view_34), kwargs = {})
#   %add_18 : [num_users=2] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_15, %view_40), kwargs = {})
#   %add_21 : [num_users=1] = call_function[target=torch.ops.aten.add.Tensor](args = (%add_18, %view_44), kwargs = {})
triton_poi_fused_add_29 = async_compile.triton('triton_poi_fused_add_29', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 1048576}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*fp32', 'in_ptr1': '*fp16', 'in_ptr2': '*fp32', 'in_ptr3': '*fp16', 'in_ptr4': '*fp16', 'in_ptr5': '*fp32', 'out_ptr0': '*fp32', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_29', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 6, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_add_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 709632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex // 5544
    x3 = (xindex % 5544)
    x4 = xindex
    x0 = (xindex % 168)
    tmp0 = tl.load(in_ptr0 + (x3 + 5568*x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask).to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x4), xmask).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (x4), xmask).to(tl.float32)
    tmp11 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 + tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 + tmp5
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 + tmp8
    tmp12 = tmp11.to(tl.float32)
    tmp13 = tmp10 + tmp12
    tmp14 = tmp13.to(tl.float32)
    tmp15 = tmp9 + tmp14
    tl.store(out_ptr0 + (x4), tmp15, xmask)
''', device_str='cuda')


# kernel path: C:\PYTHON~1\LA8F52~1\LAYERC~3\.i\6e\c6eexeqw65l6u7mqol7j77xme2oi2zalrk7sgdfgltgnlxo4w4if.py
# Topologically Sorted Source Nodes: [padded, getitem_11], Original ATen: [aten.constant_pad_nd, aten.index]
# Source node to ATen node mapping:
#   getitem_11 => index
#   padded => constant_pad_nd
# Graph fragment:
#   %constant_pad_nd : [num_users=1] = call_function[target=torch.ops.aten.constant_pad_nd.default](args = (%primals_1, [0, 32], 0.0), kwargs = {})
#   %index : [num_users=1] = call_function[target=torch.ops.aten.index.Tensor](args = (%constant_pad_nd, [None, %add_23]), kwargs = {})
triton_poi_fused_constant_pad_nd_index_30 = async_compile.triton('triton_poi_fused_constant_pad_nd_index_30', '''
import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.pointwise(
    size_hints={'x': 262144}, 
    filename=__file__,
    triton_meta={'signature': {'in_ptr0': '*i64', 'out_ptr0': '*i64', 'xnumel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=48, cc=86, major=8, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_index_30', 'mutated_arg_names': [], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 1, 'num_reduction': 0, 'backend_hash': '68DF81A04633C6847EAFD466D0567A88CA373A8F2320769BC1A4590A9BF2303E', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False},
    min_elem_per_thread=0
)
@triton.jit
def triton_poi_fused_constant_pad_nd_index_30(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 135168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)
    x0 = (xindex % 1056)
    x2 = xindex
    tmp0 = 32 + x0
    tmp1 = tl.full([1], 1056, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (32 + x2), tmp2, other=0.0)
    tl.store(out_ptr0 + (x2), tmp3, None)
''', device_str='cuda')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33 = args
    args.clear()
    assert_size_stride(primals_1, (128, 1056), (1056, 1))
    assert_size_stride(primals_2, (256, 24), (24, 1))
    assert_size_stride(primals_3, (168, 768), (768, 1))
    assert_size_stride(primals_4, (168, ), (1, ))
    assert_size_stride(primals_5, (64, 168), (168, 1))
    assert_size_stride(primals_6, (168, ), (1, ))
    assert_size_stride(primals_7, (168, ), (1, ))
    assert_size_stride(primals_8, (504, 168), (168, 1))
    assert_size_stride(primals_9, (504, ), (1, ))
    assert_size_stride(primals_10, (168, 168), (168, 1))
    assert_size_stride(primals_11, (168, ), (1, ))
    assert_size_stride(primals_12, (168, ), (1, ))
    assert_size_stride(primals_13, (168, ), (1, ))
    assert_size_stride(primals_14, (448, 168), (168, 1))
    assert_size_stride(primals_15, (448, 168), (168, 1))
    assert_size_stride(primals_16, (168, 448), (448, 1))
    assert_size_stride(primals_17, (168, ), (1, ))
    assert_size_stride(primals_18, (168, ), (1, ))
    assert_size_stride(primals_19, (504, 168), (168, 1))
    assert_size_stride(primals_20, (504, ), (1, ))
    assert_size_stride(primals_21, (168, 168), (168, 1))
    assert_size_stride(primals_22, (168, ), (1, ))
    assert_size_stride(primals_23, (168, ), (1, ))
    assert_size_stride(primals_24, (168, ), (1, ))
    assert_size_stride(primals_25, (448, 168), (168, 1))
    assert_size_stride(primals_26, (448, 168), (168, 1))
    assert_size_stride(primals_27, (168, 448), (448, 1))
    assert_size_stride(primals_28, (64, 168), (168, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (64, ), (1, ))
    assert_size_stride(primals_32, (168, 64), (64, 1))
    assert_size_stride(primals_33, (168, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0)
        buf0 = empty_strided_cuda((128, 33, 768), (25344, 768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [patch_h], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_0.run(primals_1, primals_2, buf0, 3244032, stream=stream0)
        del primals_2
        buf1 = empty_strided_cuda((168, 768), (768, 1), torch.float16)
        # Topologically Sorted Source Nodes: [patch_h], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_1.run(primals_3, buf1, 129024, stream=stream0)
        del primals_3
        buf2 = empty_strided_cuda((168, ), (1, ), torch.float16)
        # Topologically Sorted Source Nodes: [patch_h], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_2.run(primals_4, buf2, 168, stream=stream0)
        del primals_4
        buf3 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [patch_h], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf2, reinterpret_tensor(buf0, (4224, 768), (768, 1), 0), reinterpret_tensor(buf1, (768, 168), (1, 768), 0), alpha=1, beta=1, out=buf3)
        del buf2
        buf4 = empty_strided_cuda((33, ), (1, ), torch.int64)
        # Topologically Sorted Source Nodes: [positions], Original ATen: [aten.arange]
        stream0 = get_raw_stream(0)
        triton_poi_fused_arange_3.run(buf4, 33, stream=stream0)
        buf5 = empty_strided_cuda((33, 168), (168, 1), torch.float32)
        # Topologically Sorted Source Nodes: [embedding_1], Original ATen: [aten.embedding]
        stream0 = get_raw_stream(0)
        triton_poi_fused_embedding_4.run(primals_5, buf5, 5544, stream=stream0)
        del primals_5
        buf6 = empty_strided_cuda((128, 33, 1), (33, 1, 1), torch.float32)
        buf7 = empty_strided_cuda((128, 33, 1), (33, 1, 4224), torch.float32)
        buf9 = reinterpret_tensor(buf7, (128, 33, 1), (33, 1, 1), 0); del buf7  # reuse
        buf10 = empty_strided_cuda((128, 33, 168), (5568, 168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [patch_h_1, normalized, linear_1], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_5.run(buf9, buf3, buf5, primals_6, primals_7, buf6, buf10, 4224, 168, stream=stream0)
        del primals_7
        buf11 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [patch_h_1, normalized, linear_1], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_layer_norm_view_6.run(buf10, buf11, 709632, stream=stream0)
        buf12 = empty_strided_cuda((504, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(primals_8, buf12, 84672, stream=stream0)
        del primals_8
        buf13 = empty_strided_cuda((504, ), (1, ), torch.float16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_9, buf13, 504, stream=stream0)
        del primals_9
        buf14 = empty_strided_cuda((4224, 504), (504, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_1], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf13, buf11, reinterpret_tensor(buf12, (168, 504), (1, 168), 0), alpha=1, beta=1, out=buf14)
        buf15 = empty_strided_cuda((128, 4, 33, 1), (132, 1, 4, 16896), torch.float32)
        # Topologically Sorted Source Nodes: [query_2], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_pow_9.run(buf14, buf15, 16896, 42, stream=stream0)
        buf16 = empty_strided_cuda((128, 4, 33, 1), (132, 33, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_2], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_10.run(buf15, buf16, 512, 33, stream=stream0)
        buf17 = buf15; del buf15  # reuse
        # Topologically Sorted Source Nodes: [key_2], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_pow_11.run(buf14, buf17, 16896, 42, stream=stream0)
        buf18 = empty_strided_cuda((128, 4, 33, 1), (132, 33, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_2], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_10.run(buf17, buf18, 512, 33, stream=stream0)
        buf19 = empty_strided_cuda((33, 33), (33, 1), torch.bool)
        # Topologically Sorted Source Nodes: [attended], Original ATen: [aten.ones, aten.tril]
        stream0 = get_raw_stream(0)
        triton_poi_fused_ones_tril_12.run(buf4, buf19, 1089, stream=stream0)
        buf20 = empty_strided_cuda((128, 4, 33, 42), (5632, 1408, 42, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_2, attended], Original ATen: [aten._to_copy, aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_mul_13.run(buf14, buf16, buf20, 709632, stream=stream0)
        buf21 = empty_strided_cuda((128, 4, 42, 33), (5632, 1408, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attended], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_14.run(buf14, buf18, buf21, 21504, 33, stream=stream0)
        buf22 = empty_strided_cuda((512, 33, 33), (1089, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attended], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (512, 33, 42), (1408, 42, 1), 0), reinterpret_tensor(buf21, (512, 42, 33), (1408, 33, 1), 0), out=buf22)
        buf23 = reinterpret_tensor(buf17, (128, 4, 33, 1), (132, 33, 1, 1), 0); del buf17  # reuse
        buf24 = empty_strided_cuda((128, 4, 33, 1), (132, 33, 1, 1), torch.float32)
        buf25 = empty_strided_cuda((128, 4, 33, 1), (132, 33, 1, 16896), torch.bool)
        buf26 = reinterpret_tensor(buf25, (128, 4, 33, 1), (132, 33, 1, 1), 0); del buf25  # reuse
        buf27 = empty_strided_cuda((128, 4, 33, 33), (4480, 1120, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attended], Original ATen: [aten.scalar_tensor, aten.where, aten.add, aten._safe_softmax]
        stream0 = get_raw_stream(0)
        triton_per_fused__safe_softmax_add_scalar_tensor_where_15.run(buf26, buf22, buf19, buf23, buf24, buf27, 16896, 33, stream=stream0)
        buf28 = reinterpret_tensor(buf21, (128, 4, 33, 42), (5632, 1408, 42, 1), 0); del buf21  # reuse
        buf105 = empty_strided_cuda((512, 42, 33), (1408, 1, 42), torch.float16)
        # Topologically Sorted Source Nodes: [attended], Original ATen: [aten._to_copy, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_16.run(buf14, buf28, buf105, 709632, stream=stream0)
        buf29 = empty_strided_cuda((512, 33, 42), (1386, 42, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attended], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf27, (512, 33, 33), (1120, 33, 1), 0), reinterpret_tensor(buf28, (512, 33, 42), (1408, 42, 1), 0), out=buf29)
        buf30 = reinterpret_tensor(buf10, (128, 33, 4, 42), (5568, 168, 42, 1), 0); del buf10  # reuse
        # Topologically Sorted Source Nodes: [attended_1], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_17.run(buf29, buf30, 709632, stream=stream0)
        buf31 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_layer_norm_view_6.run(buf30, buf31, 709632, stream=stream0)
        buf32 = empty_strided_cuda((168, 168), (1, 168), torch.float16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_18.run(primals_10, buf32, 28224, stream=stream0)
        del primals_10
        buf33 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_2], Original ATen: [aten.addmm]
        extern_kernels.mm(buf31, buf32, out=buf33)
        buf37 = empty_strided_cuda((128, 33, 168), (5568, 168, 1), torch.float32)
        buf39 = reinterpret_tensor(buf30, (128, 33, 168), (5568, 168, 1), 0); del buf30  # reuse
        buf104 = empty_strided_cuda((128, 33, 1), (33, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [patch_h_1, h, normalized_1, gate], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_19.run(buf3, buf5, buf33, primals_11, primals_12, primals_13, buf37, buf39, buf104, 4224, 168, stream=stream0)
        del primals_13
        buf38 = empty_strided_cuda((168, 448), (1, 168), torch.float16)
        # Topologically Sorted Source Nodes: [gate], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_20.run(primals_14, buf38, 75264, stream=stream0)
        del primals_14
        buf40 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [normalized_1, gate], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_layer_norm_view_6.run(buf39, buf40, 709632, stream=stream0)
        buf41 = empty_strided_cuda((4224, 448), (448, 1), torch.float16)
        # Topologically Sorted Source Nodes: [gate], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, buf38, out=buf41)
        buf42 = empty_strided_cuda((168, 448), (1, 168), torch.float16)
        # Topologically Sorted Source Nodes: [up], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_20.run(primals_15, buf42, 75264, stream=stream0)
        del primals_15
        buf43 = empty_strided_cuda((4224, 448), (448, 1), torch.float16)
        # Topologically Sorted Source Nodes: [up], Original ATen: [aten.mm]
        extern_kernels.mm(buf40, buf42, out=buf43)
        buf44 = empty_strided_cuda((448, 168), (1, 448), torch.float16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_20.run(primals_16, buf44, 75264, stream=stream0)
        del primals_16
        buf45 = empty_strided_cuda((128, 33, 448), (14784, 448, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu, mul], Original ATen: [aten.silu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_21.run(buf41, buf43, buf45, 1892352, stream=stream0)
        buf46 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf45, (4224, 448), (448, 1), 0), buf44, out=buf46)
        buf47 = empty_strided_cuda((128, 33, 168), (5568, 168, 1), torch.float32)
        buf51 = empty_strided_cuda((128, 33, 168), (5568, 168, 1), torch.float32)
        buf52 = buf39; del buf39  # reuse
        buf103 = empty_strided_cuda((128, 33, 1), (33, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [patch_h_1, h, h_1, normalized_2, linear_6], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_22.run(buf3, buf5, buf33, primals_11, buf46, primals_17, primals_18, buf47, buf51, buf52, buf103, 4224, 168, stream=stream0)
        del primals_11
        del primals_18
        buf53 = buf46; del buf46  # reuse
        # Topologically Sorted Source Nodes: [normalized_2, linear_6], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_layer_norm_view_6.run(buf52, buf53, 709632, stream=stream0)
        buf54 = empty_strided_cuda((504, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_7.run(primals_19, buf54, 84672, stream=stream0)
        del primals_19
        buf55 = buf13; del buf13  # reuse
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_8.run(primals_20, buf55, 504, stream=stream0)
        del primals_20
        buf56 = empty_strided_cuda((4224, 504), (504, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_6], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf55, buf53, reinterpret_tensor(buf54, (168, 504), (1, 168), 0), alpha=1, beta=1, out=buf56)
        del buf55
        buf57 = empty_strided_cuda((128, 4, 33, 1), (132, 1, 4, 16896), torch.float32)
        # Topologically Sorted Source Nodes: [query_5], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_pow_9.run(buf56, buf57, 16896, 42, stream=stream0)
        buf58 = empty_strided_cuda((128, 4, 33, 1), (132, 33, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [query_5], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_10.run(buf57, buf58, 512, 33, stream=stream0)
        buf59 = buf57; del buf57  # reuse
        # Topologically Sorted Source Nodes: [key_5], Original ATen: [aten._to_copy, aten.pow, aten.mean]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_mean_pow_11.run(buf56, buf59, 16896, 42, stream=stream0)
        buf60 = empty_strided_cuda((128, 4, 33, 1), (132, 33, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [key_5], Original ATen: [aten._to_copy, aten.pow, aten.mean, aten.add, aten.rsqrt]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_mean_pow_rsqrt_10.run(buf59, buf60, 512, 33, stream=stream0)
        del buf59
        buf61 = buf28; del buf28  # reuse
        # Topologically Sorted Source Nodes: [query_5, attended_2], Original ATen: [aten._to_copy, aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_mul_13.run(buf56, buf58, buf61, 709632, stream=stream0)
        buf62 = reinterpret_tensor(buf20, (128, 4, 42, 33), (5632, 1408, 33, 1), 0); del buf20  # reuse
        # Topologically Sorted Source Nodes: [attended_2], Original ATen: [aten.mul, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_mul_14.run(buf56, buf60, buf62, 21504, 33, stream=stream0)
        buf63 = empty_strided_cuda((512, 33, 33), (1089, 33, 1), torch.float32)
        # Topologically Sorted Source Nodes: [attended_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf61, (512, 33, 42), (1408, 42, 1), 0), reinterpret_tensor(buf62, (512, 42, 33), (1408, 33, 1), 0), out=buf63)
        del buf61
        buf67 = buf27; del buf27  # reuse
        # Topologically Sorted Source Nodes: [attended, attended_2], Original ATen: [aten.ones, aten.scalar_tensor, aten._safe_softmax, aten.tril, aten.where, aten.add]
        stream0 = get_raw_stream(0)
        triton_per_fused__safe_softmax_add_ones_scalar_tensor_tril_where_23.run(buf63, buf4, buf67, 16896, 33, stream=stream0)
        del buf63
        buf68 = reinterpret_tensor(buf62, (128, 4, 33, 42), (5632, 1408, 42, 1), 0); del buf62  # reuse
        buf102 = empty_strided_cuda((512, 42, 33), (1408, 1, 42), torch.float16)
        # Topologically Sorted Source Nodes: [attended_2], Original ATen: [aten._to_copy, aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_clone_16.run(buf56, buf68, buf102, 709632, stream=stream0)
        buf69 = buf29; del buf29  # reuse
        # Topologically Sorted Source Nodes: [attended_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf67, (512, 33, 33), (1120, 33, 1), 0), reinterpret_tensor(buf68, (512, 33, 42), (1408, 42, 1), 0), out=buf69)
        del buf68
        buf70 = reinterpret_tensor(buf52, (128, 33, 4, 42), (5568, 168, 42, 1), 0); del buf52  # reuse
        # Topologically Sorted Source Nodes: [attended_3], Original ATen: [aten.clone]
        stream0 = get_raw_stream(0)
        triton_poi_fused_clone_17.run(buf69, buf70, 709632, stream=stream0)
        buf71 = buf33; del buf33  # reuse
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_layer_norm_view_6.run(buf70, buf71, 709632, stream=stream0)
        buf72 = empty_strided_cuda((168, 168), (1, 168), torch.float16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_18.run(primals_21, buf72, 28224, stream=stream0)
        del primals_21
        buf73 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_7], Original ATen: [aten.addmm]
        extern_kernels.mm(buf71, buf72, out=buf73)
        buf77 = empty_strided_cuda((128, 33, 168), (5568, 168, 1), torch.float32)
        buf79 = reinterpret_tensor(buf70, (128, 33, 168), (5568, 168, 1), 0); del buf70  # reuse
        buf101 = empty_strided_cuda((128, 33, 1), (33, 1, 1), torch.float32)
        # Topologically Sorted Source Nodes: [h_2, normalized_3, gate_1], Original ATen: [aten.add, aten.native_layer_norm, aten._to_copy, aten.native_layer_norm_backward]
        stream0 = get_raw_stream(0)
        triton_red_fused__to_copy_add_native_layer_norm_native_layer_norm_backward_24.run(buf47, buf73, primals_22, primals_23, primals_24, buf77, buf79, buf101, 4224, 168, stream=stream0)
        del primals_24
        buf78 = empty_strided_cuda((168, 448), (1, 168), torch.float16)
        # Topologically Sorted Source Nodes: [gate_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_20.run(primals_25, buf78, 75264, stream=stream0)
        del primals_25
        buf80 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [normalized_3, gate_1], Original ATen: [aten.native_layer_norm, aten._to_copy, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_layer_norm_view_6.run(buf79, buf80, 709632, stream=stream0)
        buf81 = empty_strided_cuda((4224, 448), (448, 1), torch.float16)
        # Topologically Sorted Source Nodes: [gate_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf80, buf78, out=buf81)
        buf82 = empty_strided_cuda((168, 448), (1, 168), torch.float16)
        # Topologically Sorted Source Nodes: [up_1], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_20.run(primals_26, buf82, 75264, stream=stream0)
        del primals_26
        buf83 = empty_strided_cuda((4224, 448), (448, 1), torch.float16)
        # Topologically Sorted Source Nodes: [up_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf80, buf82, out=buf83)
        buf84 = empty_strided_cuda((448, 168), (1, 448), torch.float16)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_t_20.run(primals_27, buf84, 75264, stream=stream0)
        del primals_27
        buf85 = empty_strided_cuda((128, 33, 448), (14784, 448, 1), torch.float16)
        # Topologically Sorted Source Nodes: [silu_1, mul_1], Original ATen: [aten.silu, aten.mul]
        stream0 = get_raw_stream(0)
        triton_poi_fused_mul_silu_21.run(buf81, buf83, buf85, 1892352, stream=stream0)
        buf86 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_10], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf85, (4224, 448), (448, 1), 0), buf84, out=buf86)
        buf87 = buf79; del buf79  # reuse
        # Topologically Sorted Source Nodes: [h_2, h_3, input_1], Original ATen: [aten.add, aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_25.run(buf47, buf73, primals_22, buf86, buf87, 709632, stream=stream0)
        buf88 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [h_2, h_3, input_1], Original ATen: [aten.add, aten._to_copy, aten.view]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_add_native_layer_norm_view_6.run(buf87, buf88, 709632, stream=stream0)
        del buf87
        buf89 = empty_strided_cuda((64, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_26.run(primals_28, buf89, 10752, stream=stream0)
        del primals_28
        buf90 = empty_strided_cuda((64, ), (1, ), torch.float16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_27.run(primals_29, buf90, 64, stream=stream0)
        del primals_29
        buf91 = empty_strided_cuda((4224, 64), (64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_1], Original ATen: [aten._to_copy, aten.addmm]
        extern_kernels.addmm(buf90, buf88, reinterpret_tensor(buf89, (168, 64), (1, 168), 0), alpha=1, beta=1, out=buf91)
        del buf90
        buf92 = empty_strided_cuda((128, 33, 1), (33, 1, 1), torch.float32)
        buf93 = empty_strided_cuda((128, 33, 1), (33, 1, 4224), torch.float32)
        buf95 = reinterpret_tensor(buf93, (128, 33, 1), (33, 1, 1), 0); del buf93  # reuse
        buf96 = empty_strided_cuda((128, 33, 64), (2112, 64, 1), torch.float16)
        # Topologically Sorted Source Nodes: [input_2, linear_12], Original ATen: [aten._to_copy, aten.native_layer_norm]
        stream0 = get_raw_stream(0)
        triton_per_fused__to_copy_native_layer_norm_28.run(buf95, buf91, primals_30, primals_31, buf92, buf96, 4224, 64, stream=stream0)
        del primals_31
        buf97 = empty_strided_cuda((64, 168), (1, 64), torch.float16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten._to_copy, aten.t]
        stream0 = get_raw_stream(0)
        triton_poi_fused__to_copy_26.run(primals_32, buf97, 10752, stream=stream0)
        del primals_32
        buf98 = empty_strided_cuda((4224, 168), (168, 1), torch.float16)
        # Topologically Sorted Source Nodes: [linear_12], Original ATen: [aten.addmm]
        extern_kernels.mm(reinterpret_tensor(buf96, (4224, 64), (64, 1), 0), buf97, out=buf98)
        buf99 = reinterpret_tensor(buf69, (128, 33, 168), (5544, 168, 1), 0); del buf69  # reuse
        # Topologically Sorted Source Nodes: [h_2, h_3, add_5], Original ATen: [aten.add]
        stream0 = get_raw_stream(0)
        triton_poi_fused_add_29.run(buf47, buf73, primals_22, buf86, buf98, primals_33, buf99, 709632, stream=stream0)
        del buf47
        del buf73
        del buf86
        del buf98
        del primals_22
        del primals_33
        buf100 = empty_strided_cuda((128, 33, 32), (1056, 32, 1), torch.int64)
        # Topologically Sorted Source Nodes: [padded, getitem_11], Original ATen: [aten.constant_pad_nd, aten.index]
        stream0 = get_raw_stream(0)
        triton_poi_fused_constant_pad_nd_index_30.run(primals_1, buf100, 135168, stream=stream0)
    return (buf99, buf100, primals_1, primals_1, primals_6, primals_12, primals_17, primals_23, primals_30, reinterpret_tensor(buf0, (4224, 768), (768, 1), 0), buf3, buf4, buf5, buf6, buf9, buf11, reinterpret_tensor(buf14, (128, 4, 33, 42), (16632, 42, 504, 1), 0), reinterpret_tensor(buf14, (128, 4, 33, 42), (16632, 42, 504, 1), 168), buf16, buf18, buf19, buf22, buf23, buf24, buf26, buf31, buf37, buf40, buf41, buf43, reinterpret_tensor(buf45, (4224, 448), (448, 1), 0), buf51, buf53, reinterpret_tensor(buf56, (128, 4, 33, 42), (16632, 42, 504, 1), 0), reinterpret_tensor(buf56, (128, 4, 33, 42), (16632, 42, 504, 1), 168), buf58, buf60, buf67, buf71, buf77, buf80, buf81, buf83, reinterpret_tensor(buf85, (4224, 448), (448, 1), 0), buf88, buf91, buf92, buf95, reinterpret_tensor(buf96, (4224, 64), (64, 1), 0), reinterpret_tensor(buf97, (168, 64), (64, 1), 0), buf89, reinterpret_tensor(buf84, (168, 448), (448, 1), 0), reinterpret_tensor(buf82, (448, 168), (168, 1), 0), reinterpret_tensor(buf78, (448, 168), (168, 1), 0), buf101, reinterpret_tensor(buf72, (168, 168), (168, 1), 0), buf102, buf54, buf103, reinterpret_tensor(buf44, (168, 448), (448, 1), 0), reinterpret_tensor(buf42, (448, 168), (168, 1), 0), reinterpret_tensor(buf38, (448, 168), (168, 1), 0), buf104, reinterpret_tensor(buf32, (168, 168), (168, 1), 0), buf105, buf12, buf1, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, 1056), (1056, 1), device='cuda:0', dtype=torch.int64)
    primals_2 = rand_strided((256, 24), (24, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((168, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 168), (168, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((504, 168), (168, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((168, 168), (168, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((448, 168), (168, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((448, 168), (168, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((168, 448), (448, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((504, 168), (168, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((504, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((168, 168), (168, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((448, 168), (168, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((448, 168), (168, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((168, 448), (448, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 168), (168, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((168, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    fn = lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33])
    return print_performance(fn, times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
