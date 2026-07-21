
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
