"""Fused Triton inference kernels for CountCake GPU generation."""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised on CPU-only installations
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _lookup(keys, values, query, size: tl.constexpr, steps: tl.constexpr):
        low = tl.zeros(query.shape, tl.int32)
        high = tl.full(query.shape, size, tl.int32)
        for _ in tl.static_range(steps):
            middle = (low + high) // 2
            safe = tl.minimum(middle, size - 1)
            found_key = tl.load(keys + safe)
            move_right = (middle < size) & (found_key < query)
            high = tl.where(move_right, high, middle)
            low = tl.where(move_right, middle + 1, low)
        safe = tl.minimum(low, size - 1)
        found_key = tl.load(keys + safe)
        value = tl.load(values + safe)
        return tl.where((low < size) & (found_key == query), value, 0.0)


    @triton.jit
    def _greedy_patch_kernel(
        history,
        neural_probability,
        gates,
        output,
        unigram,
        keys1,
        counts1,
        contexts1,
        totals1,
        keys2,
        counts2,
        contexts2,
        totals2,
        keys3,
        counts3,
        contexts3,
        totals3,
        keys4,
        counts4,
        contexts4,
        totals4,
        unigram_total,
        n1: tl.constexpr,
        nc1: tl.constexpr,
        n2: tl.constexpr,
        nc2: tl.constexpr,
        n3: tl.constexpr,
        nc3: tl.constexpr,
        n4: tl.constexpr,
        nc4: tl.constexpr,
        patch_size: tl.constexpr,
    ):
        byte = tl.arange(0, 256)
        oldest = tl.load(history)
        older = tl.load(history + 1)
        previous = tl.load(history + 2)
        latest = tl.load(history + 3)
        offset = 0
        while offset < patch_size:
            probability = (tl.load(unigram + byte) + 0.5) / (
                unigram_total + 128.0
            )
            context1 = latest
            joint1 = byte + context1 * 256
            joint_count = _lookup(keys1, counts1, joint1, n1, 14)
            total = _lookup(contexts1, totals1, context1, nc1, 8)
            probability = (joint_count + 128.0 * probability) / (total + 128.0)

            context2 = latest + previous * 256
            joint2 = byte + context2 * 256
            joint_count = _lookup(keys2, counts2, joint2, n2, 17)
            total = _lookup(contexts2, totals2, context2, nc2, 14)
            probability = (joint_count + 32.0 * probability) / (total + 32.0)

            context3 = latest + previous * 256 + older * 65536
            joint3 = byte + context3 * 256
            joint_count = _lookup(keys3, counts3, joint3, n3, 19)
            total = _lookup(contexts3, totals3, context3, nc3, 17)
            probability = (joint_count + 64.0 * probability) / (total + 64.0)

            context4 = (
                latest
                + previous * 256
                + older * 65536
                + oldest * 16777216
            )
            joint4 = byte.to(tl.int64) + context4.to(tl.int64) * 256
            joint_count = _lookup(keys4, counts4, joint4, n4, 19)
            total = _lookup(contexts4, totals4, context4, nc4, 17)
            probability = (joint_count + 32.0 * probability) / (total + 32.0)

            gate = tl.load(gates + offset)
            neural = tl.load(neural_probability + offset * 256 + byte)
            mixed = (1.0 - gate) * probability + gate * neural
            selected = tl.argmax(
                mixed, axis=0, tie_break_left=True
            ).to(tl.int64)
            tl.store(output + offset, selected)
            oldest = older
            older = previous
            previous = latest
            latest = selected
            offset += 1


    @triton.jit
    def _lower_bound(keys, query, size: tl.constexpr, steps: tl.constexpr):
        low = tl.zeros(query.shape, tl.int32)
        high = tl.full(query.shape, size, tl.int32)
        for _ in tl.static_range(steps):
            middle = (low + high) // 2
            safe = tl.minimum(middle, size - 1)
            found_key = tl.load(keys + safe)
            move_right = (middle < size) & (found_key < query)
            high = tl.where(move_right, high, middle)
            low = tl.where(move_right, middle + 1, low)
        safe = tl.minimum(low, size - 1)
        return low


    @triton.jit
    def _packed_stage(
        probability,
        byte,
        history,
        position,
        keys,
        counts,
        contexts,
        totals,
        starts,
        ends,
        strength,
        order: tl.constexpr,
        n: tl.constexpr,
        nc: tl.constexpr,
    ):
        context = tl.load(history + position - 1).to(tl.int64) * 0
        for lag in tl.static_range(order):
            history_value = tl.load(history + position - 1 - lag).to(tl.int64)
            context += history_value << (8 * lag)
        context_index = _lower_bound(contexts, context, nc, 21)
        context_safe = tl.minimum(context_index, nc - 1)
        found = (context_index < nc) & (
            tl.load(contexts + context_safe) == context
        )
        start = tl.load(starts + context_safe)
        end = tl.load(ends + context_safe)
        joint_count = tl.zeros(byte.shape, tl.float32)
        cursor = start
        while cursor < end:
            joint = tl.load(keys + cursor)
            target = joint & 255
            continuation_count = tl.load(counts + cursor)
            joint_count += tl.where(
                found & (byte == target), continuation_count, 0.0
            )
            cursor += 1
        total = tl.where(found, tl.load(totals + context_safe), 0.0)
        return (joint_count + strength * probability) / (total + strength)


    @triton.jit
    def _hashed_stage(
        probability,
        byte,
        history,
        position,
        keys,
        counts,
        contexts,
        totals,
        starts,
        ends,
        strength,
        order: tl.constexpr,
        hash_bits: tl.constexpr,
        n: tl.constexpr,
        nc: tl.constexpr,
    ):
        context = tl.load(history + position - 1).to(tl.int64) * 0
        mask = (1 << hash_bits) - 1
        for offset in tl.static_range(order):
            history_value = tl.load(
                history + position - order + offset
            ).to(tl.int64)
            context = (context * 257 + history_value + 1) & mask
        context_index = _lower_bound(contexts, context, nc, 21)
        safe = tl.minimum(context_index, nc - 1)
        found = (context_index < nc) & (tl.load(contexts + safe) == context)
        start = tl.load(starts + safe)
        end = tl.load(ends + safe)
        joint_count = tl.zeros(byte.shape, tl.float32)
        cursor = start
        while cursor < end:
            joint = tl.load(keys + cursor)
            target = joint & 255
            continuation_count = tl.load(counts + cursor)
            joint_count += tl.where(
                found & (byte == target), continuation_count, 0.0
            )
            cursor += 1
        total = tl.where(found, tl.load(totals + safe), 0.0)
        return (joint_count + strength * probability) / (total + strength)


    @triton.jit
    def _byte_class(value):
        value = tl.where((value >= 65) & (value <= 90), value + 32, value)
        value = tl.where((value >= 48) & (value <= 57), 48, value)
        whitespace = (value == 9) | (value == 10) | (value == 13) | (value == 32)
        return tl.where(whitespace, 32, value)


    @triton.jit
    def _pack_lags(
        history,
        position,
        start_lag: tl.constexpr,
        length: tl.constexpr,
        normalized: tl.constexpr,
    ):
        key = tl.zeros(position.shape, tl.int64)
        for offset in tl.static_range(length):
            lag = start_lag + offset
            value = tl.load(
                history + position - 1 - lag,
                mask=position > lag,
                other=0,
            ).to(tl.int64)
            if normalized:
                value = _byte_class(value)
            key |= value << (8 * offset)
        return key


    @triton.jit
    def _prefill_context_keys_kernel(
        history,
        context_keys,
        length,
        capacity,
        block_size: tl.constexpr,
    ):
        position = tl.program_id(0) * block_size + tl.arange(0, block_size)
        active = position < length
        values0 = _pack_lags(history, position, start_lag=0, length=8, normalized=False)
        values1 = _pack_lags(history, position, start_lag=0, length=6, normalized=False)
        values2 = _pack_lags(history, position, start_lag=0, length=4, normalized=False)
        values3 = _pack_lags(history, position, start_lag=0, length=2, normalized=False)
        values4 = _pack_lags(history, position, start_lag=8, length=8, normalized=False)
        values5 = _pack_lags(history, position, start_lag=16, length=8, normalized=False)
        values6 = _pack_lags(history, position, start_lag=8, length=4, normalized=False)
        values7 = _pack_lags(history, position, start_lag=8, length=2, normalized=False)
        values8 = _pack_lags(history, position, start_lag=0, length=5, normalized=True)
        values9 = _pack_lags(history, position, start_lag=0, length=3, normalized=True)
        tl.store(context_keys + 0 * capacity + position, values0, mask=active)
        tl.store(context_keys + 1 * capacity + position, values1, mask=active)
        tl.store(context_keys + 2 * capacity + position, values2, mask=active)
        tl.store(context_keys + 3 * capacity + position, values3, mask=active)
        tl.store(context_keys + 4 * capacity + position, values4, mask=active)
        tl.store(context_keys + 5 * capacity + position, values5, mask=active)
        tl.store(context_keys + 6 * capacity + position, values6, mask=active)
        tl.store(context_keys + 7 * capacity + position, values7, mask=active)
        tl.store(context_keys + 8 * capacity + position, values8, mask=active)
        tl.store(context_keys + 9 * capacity + position, values9, mask=active)


    @triton.jit
    def _cache_stats_kernel(
        history,
        context_keys,
        capacity,
        position,
        stats,
        window: tl.constexpr,
        block_size: tl.constexpr,
    ):
        lag = tl.arange(0, block_size) + 1
        previous_position = position - lag
        valid = (lag <= window) & (previous_position >= 0)
        previous_target = tl.load(
            history + previous_position, mask=valid, other=0
        ).to(tl.int64)

        current0 = _pack_lags(history, position, start_lag=0, length=8, normalized=False)
        current1 = _pack_lags(history, position, start_lag=0, length=6, normalized=False)
        current2 = _pack_lags(history, position, start_lag=0, length=4, normalized=False)
        current3 = _pack_lags(history, position, start_lag=0, length=2, normalized=False)
        current4 = _pack_lags(history, position, start_lag=8, length=8, normalized=False)
        current5 = _pack_lags(history, position, start_lag=16, length=8, normalized=False)
        current6 = _pack_lags(history, position, start_lag=8, length=4, normalized=False)
        current7 = _pack_lags(history, position, start_lag=8, length=2, normalized=False)
        current8 = _pack_lags(history, position, start_lag=0, length=5, normalized=True)
        current9 = _pack_lags(history, position, start_lag=0, length=3, normalized=True)
        tl.store(context_keys + 0 * capacity + position, current0)
        tl.store(context_keys + 1 * capacity + position, current1)
        tl.store(context_keys + 2 * capacity + position, current2)
        tl.store(context_keys + 3 * capacity + position, current3)
        tl.store(context_keys + 4 * capacity + position, current4)
        tl.store(context_keys + 5 * capacity + position, current5)
        tl.store(context_keys + 6 * capacity + position, current6)
        tl.store(context_keys + 7 * capacity + position, current7)
        tl.store(context_keys + 8 * capacity + position, current8)
        tl.store(context_keys + 9 * capacity + position, current9)

        previous0 = tl.load(context_keys + 0 * capacity + previous_position, mask=valid, other=0)
        previous1 = tl.load(context_keys + 1 * capacity + previous_position, mask=valid, other=0)
        previous2 = tl.load(context_keys + 2 * capacity + previous_position, mask=valid, other=0)
        previous3 = tl.load(context_keys + 3 * capacity + previous_position, mask=valid, other=0)
        previous4 = tl.load(context_keys + 4 * capacity + previous_position, mask=valid, other=0)
        previous5 = tl.load(context_keys + 5 * capacity + previous_position, mask=valid, other=0)
        previous6 = tl.load(context_keys + 6 * capacity + previous_position, mask=valid, other=0)
        previous7 = tl.load(context_keys + 7 * capacity + previous_position, mask=valid, other=0)
        previous8 = tl.load(context_keys + 8 * capacity + previous_position, mask=valid, other=0)
        previous9 = tl.load(context_keys + 9 * capacity + previous_position, mask=valid, other=0)

        match8 = valid & (previous_position >= 8) & (previous0 == current0)
        match6 = valid & (previous_position >= 6) & (previous1 == current1)
        match4 = valid & (previous_position >= 4) & (previous2 == current2)
        match2 = valid & (previous_position >= 2) & (previous3 == current3)
        normalized5 = valid & (previous_position >= 5) & (previous8 == current8)
        normalized3 = valid & (previous_position >= 3) & (previous9 == current9)
        broadcast = tl.zeros(lag.shape, tl.int32)
        tl.atomic_add(stats + 0 * 256 + previous_target, match8.to(tl.int64), mask=match8)
        tl.atomic_add(stats + 1 * 256 + previous_target, match6.to(tl.int64), mask=match6)
        tl.atomic_add(stats + 2 * 256 + previous_target, match4.to(tl.int64), mask=match4)
        tl.atomic_add(stats + 3 * 256 + previous_target, match2.to(tl.int64), mask=match2)
        tl.atomic_add(stats + 4 * 256 + previous_target, normalized5.to(tl.int64), mask=normalized5)
        tl.atomic_add(stats + 5 * 256 + previous_target, normalized3.to(tl.int64), mask=normalized3)
        tl.atomic_add(stats + 1536 + 0 + broadcast, match8.to(tl.int64), mask=match8)
        tl.atomic_add(stats + 1536 + 1 + broadcast, match6.to(tl.int64), mask=match6)
        tl.atomic_add(stats + 1536 + 2 + broadcast, match4.to(tl.int64), mask=match4)
        tl.atomic_add(stats + 1536 + 3 + broadcast, match2.to(tl.int64), mask=match2)
        tl.atomic_add(stats + 1536 + 4 + broadcast, normalized5.to(tl.int64), mask=normalized5)
        tl.atomic_add(stats + 1536 + 5 + broadcast, normalized3.to(tl.int64), mask=normalized3)

        match24 = valid & (previous_position >= 24) & (previous0 == current0) & (previous4 == current4) & (previous5 == current5)
        match16 = valid & (previous_position >= 16) & (previous0 == current0) & (previous4 == current4)
        match12 = valid & (previous_position >= 12) & (previous0 == current0) & (previous6 == current6)
        match10 = valid & (previous_position >= 10) & (previous0 == current0) & (previous7 == current7)
        encoded = (previous_position + 1).to(tl.int64) * 256 + previous_target
        tl.atomic_max(stats + 1542 + broadcast, encoded, mask=match24)
        tl.atomic_max(stats + 1543 + broadcast, encoded, mask=match16)
        tl.atomic_max(stats + 1544 + broadcast, encoded, mask=match12)
        tl.atomic_max(stats + 1545 + broadcast, encoded, mask=match10)


    @triton.jit
    def _context_equal(
        history,
        left_position,
        right_position,
        order: tl.constexpr,
        normalized: tl.constexpr,
    ):
        equal = (left_position >= order) & (right_position >= order)
        for offset in tl.static_range(order):
            left = tl.load(
                history + left_position - order + offset,
                mask=left_position >= order,
                other=0,
            ).to(tl.int32)
            right = tl.load(
                history + right_position - order + offset,
                mask=right_position >= order,
                other=0,
            ).to(tl.int32)
            if normalized:
                left = _byte_class(left)
                right = _byte_class(right)
            equal &= left == right
        return equal


    @triton.jit
    def _map_slot(
        map_keys,
        map_occupied,
        stage,
        key,
        capacity: tl.constexpr,
    ):
        mask = capacity - 1
        slot = (key ^ (key >> 32)) & mask
        result = slot
        done = False
        probe = 0
        while (probe < capacity) & ~done:
            index = stage * capacity + slot
            occupied = tl.load(map_occupied + index) != 0
            candidate = tl.load(map_keys + index)
            hit = occupied & (candidate == key)
            empty = ~occupied
            take = (hit | empty) & ~done
            result = tl.where(take, slot, result)
            done |= hit | empty
            slot = (slot + 1) & mask
            probe += 1
        return result


    @triton.jit
    def _map_update(
        map_keys,
        map_occupied,
        map_counts,
        map_totals,
        stage,
        key,
        target,
        delta: tl.constexpr,
        capacity: tl.constexpr,
    ):
        slot = _map_slot(
            map_keys, map_occupied, stage, key, capacity=capacity
        )
        index = stage * capacity + slot
        occupied = tl.load(map_occupied + index) != 0
        candidate = tl.load(map_keys + index)
        found = occupied & (candidate == key)
        if delta > 0:
            tl.store(map_keys + index, key, mask=~found)
            tl.store(map_occupied + index, 1, mask=~found)
            apply = True
        else:
            apply = found
        count_index = index * 256 + target
        old_count = tl.load(map_counts + count_index, mask=apply, other=0)
        old_total = tl.load(map_totals + index, mask=apply, other=0)
        tl.store(map_counts + count_index, old_count + delta, mask=apply)
        tl.store(map_totals + index, old_total + delta, mask=apply)


    @triton.jit
    def _recent_map_slot(
        keys0,
        keys1,
        keys2,
        occupied_map,
        stage,
        key0,
        key1,
        key2,
        capacity: tl.constexpr,
    ):
        mask = capacity - 1
        mixed = key0 ^ (key0 >> 32) ^ key1 ^ (key1 >> 29) ^ key2
        slot = mixed & mask
        result = slot
        done = False
        probe = 0
        while (probe < capacity) & ~done:
            index = stage * capacity + slot
            occupied = tl.load(occupied_map + index) != 0
            hit = (
                occupied
                & (tl.load(keys0 + index) == key0)
                & (tl.load(keys1 + index) == key1)
                & (tl.load(keys2 + index) == key2)
            )
            empty = ~occupied
            take = (hit | empty) & ~done
            result = tl.where(take, slot, result)
            done |= hit | empty
            slot = (slot + 1) & mask
            probe += 1
        return result


    @triton.jit
    def _recent_map_update(
        keys0,
        keys1,
        keys2,
        occupied_map,
        latest_map,
        stage,
        key0,
        key1,
        key2,
        position,
        capacity: tl.constexpr,
    ):
        slot = _recent_map_slot(
            keys0,
            keys1,
            keys2,
            occupied_map,
            stage,
            key0,
            key1,
            key2,
            capacity=capacity,
        )
        index = stage * capacity + slot
        occupied = tl.load(occupied_map + index) != 0
        found = (
            occupied
            & (tl.load(keys0 + index) == key0)
            & (tl.load(keys1 + index) == key1)
            & (tl.load(keys2 + index) == key2)
        )
        tl.store(keys0 + index, key0, mask=~found)
        tl.store(keys1 + index, key1, mask=~found)
        tl.store(keys2 + index, key2, mask=~found)
        tl.store(occupied_map + index, 1, mask=~found)
        tl.store(latest_map + index, position)


    @triton.jit
    def _recurrent_cached_byte_kernel(
        history,
        position,
        neural_probability,
        gate,
        output,
        cache_context_keys,
        cache_capacity,
        cache_stats,
        cache_map_keys,
        cache_map_occupied,
        cache_map_counts,
        cache_map_totals,
        recent_map_keys0,
        recent_map_keys1,
        recent_map_keys2,
        recent_map_occupied,
        recent_map_latest,
        unigram,
        keys1, counts1, contexts1, totals1, starts1, ends1,
        keys2, counts2, contexts2, totals2, starts2, ends2,
        keys3, counts3, contexts3, totals3, starts3, ends3,
        keys4, counts4, contexts4, totals4, starts4, ends4,
        keys5, counts5, contexts5, totals5, starts5, ends5,
        keys6, counts6, contexts6, totals6, starts6, ends6,
        keys7, counts7, contexts7, totals7, starts7, ends7,
        keys8, counts8, contexts8, totals8, starts8, ends8,
        keys9, counts9, contexts9, totals9, starts9, ends9,
        keys10, counts10, contexts10, totals10, starts10, ends10,
        keys11, counts11, contexts11, totals11, starts11, ends11,
        keys12, counts12, contexts12, totals12, starts12, ends12,
        unigram_total,
        strength1, strength2, strength3, strength4,
        strength5, strength6, strength7, strength8,
        strength9, strength10, strength11, strength12,
        exact8_strength, exact6_strength, exact4_strength, exact2_strength,
        recent24_strength, recent16_strength, recent12_strength, recent10_strength,
        normalized5_strength, normalized3_strength,
        n1: tl.constexpr, nc1: tl.constexpr,
        n2: tl.constexpr, nc2: tl.constexpr,
        n3: tl.constexpr, nc3: tl.constexpr,
        n4: tl.constexpr, nc4: tl.constexpr,
        n5: tl.constexpr, nc5: tl.constexpr,
        n6: tl.constexpr, nc6: tl.constexpr,
        n7: tl.constexpr, nc7: tl.constexpr, hash7: tl.constexpr,
        n8: tl.constexpr, nc8: tl.constexpr, hash8: tl.constexpr,
        n9: tl.constexpr, nc9: tl.constexpr, hash9: tl.constexpr,
        n10: tl.constexpr, nc10: tl.constexpr, hash10: tl.constexpr,
        n11: tl.constexpr, nc11: tl.constexpr, hash11: tl.constexpr,
        n12: tl.constexpr, nc12: tl.constexpr, hash12: tl.constexpr,
        window: tl.constexpr,
        map_capacity: tl.constexpr,
    ):
        byte = tl.arange(0, 256)
        probability = (tl.load(unigram + byte) + 0.5) / (
            unigram_total + 128.0
        )
        probability = _packed_stage(probability, byte, history, position, keys1, counts1, contexts1, totals1, starts1, ends1, strength1, order=1, n=n1, nc=nc1)
        probability = _packed_stage(probability, byte, history, position, keys2, counts2, contexts2, totals2, starts2, ends2, strength2, order=2, n=n2, nc=nc2)
        probability = _packed_stage(probability, byte, history, position, keys3, counts3, contexts3, totals3, starts3, ends3, strength3, order=3, n=n3, nc=nc3)
        probability = _packed_stage(probability, byte, history, position, keys4, counts4, contexts4, totals4, starts4, ends4, strength4, order=4, n=n4, nc=nc4)
        probability = _packed_stage(probability, byte, history, position, keys5, counts5, contexts5, totals5, starts5, ends5, strength5, order=5, n=n5, nc=nc5)
        probability = _packed_stage(probability, byte, history, position, keys6, counts6, contexts6, totals6, starts6, ends6, strength6, order=6, n=n6, nc=nc6)
        probability = _hashed_stage(probability, byte, history, position, keys7, counts7, contexts7, totals7, starts7, ends7, strength7, order=7, hash_bits=hash7, n=n7, nc=nc7)
        probability = _hashed_stage(probability, byte, history, position, keys8, counts8, contexts8, totals8, starts8, ends8, strength8, order=8, hash_bits=hash8, n=n8, nc=nc8)
        probability = _hashed_stage(probability, byte, history, position, keys9, counts9, contexts9, totals9, starts9, ends9, strength9, order=9, hash_bits=hash9, n=n9, nc=nc9)
        probability = _hashed_stage(probability, byte, history, position, keys10, counts10, contexts10, totals10, starts10, ends10, strength10, order=10, hash_bits=hash10, n=n10, nc=nc10)
        probability = _hashed_stage(probability, byte, history, position, keys11, counts11, contexts11, totals11, starts11, ends11, strength11, order=11, hash_bits=hash11, n=n11, nc=nc11)
        probability = _hashed_stage(probability, byte, history, position, keys12, counts12, contexts12, totals12, starts12, ends12, strength12, order=12, hash_bits=hash12, n=n12, nc=nc12)
        gate_value = tl.load(gate)
        probability = (
            (1.0 - gate_value) * probability
            + gate_value * tl.load(neural_probability + byte)
        )

        current0 = _pack_lags(history, position, start_lag=0, length=8, normalized=False)
        current1 = _pack_lags(history, position, start_lag=0, length=6, normalized=False)
        current2 = _pack_lags(history, position, start_lag=0, length=4, normalized=False)
        current3 = _pack_lags(history, position, start_lag=0, length=2, normalized=False)
        current4 = _pack_lags(history, position, start_lag=8, length=8, normalized=False)
        current5 = _pack_lags(history, position, start_lag=16, length=8, normalized=False)
        current6 = _pack_lags(history, position, start_lag=8, length=4, normalized=False)
        current7 = _pack_lags(history, position, start_lag=8, length=2, normalized=False)
        current8 = _pack_lags(history, position, start_lag=0, length=5, normalized=True)
        current9 = _pack_lags(history, position, start_lag=0, length=3, normalized=True)
        tl.store(cache_context_keys + 0 * cache_capacity + position, current0)
        tl.store(cache_context_keys + 1 * cache_capacity + position, current1)
        tl.store(cache_context_keys + 2 * cache_capacity + position, current2)
        tl.store(cache_context_keys + 3 * cache_capacity + position, current3)
        tl.store(cache_context_keys + 4 * cache_capacity + position, current4)
        tl.store(cache_context_keys + 5 * cache_capacity + position, current5)
        tl.store(cache_context_keys + 6 * cache_capacity + position, current6)
        tl.store(cache_context_keys + 7 * cache_capacity + position, current7)
        tl.store(cache_context_keys + 8 * cache_capacity + position, current8)
        tl.store(cache_context_keys + 9 * cache_capacity + position, current9)

        slot0 = _map_slot(cache_map_keys, cache_map_occupied, 0, current0, capacity=map_capacity)
        slot1 = _map_slot(cache_map_keys, cache_map_occupied, 1, current1, capacity=map_capacity)
        slot2 = _map_slot(cache_map_keys, cache_map_occupied, 2, current2, capacity=map_capacity)
        slot3 = _map_slot(cache_map_keys, cache_map_occupied, 3, current3, capacity=map_capacity)
        slot4 = _map_slot(cache_map_keys, cache_map_occupied, 4, current8, capacity=map_capacity)
        slot5 = _map_slot(cache_map_keys, cache_map_occupied, 5, current9, capacity=map_capacity)
        index0 = 0 * map_capacity + slot0
        index1 = 1 * map_capacity + slot1
        index2 = 2 * map_capacity + slot2
        index3 = 3 * map_capacity + slot3
        index4 = 4 * map_capacity + slot4
        index5 = 5 * map_capacity + slot5
        found0 = (tl.load(cache_map_occupied + index0) != 0) & (tl.load(cache_map_keys + index0) == current0)
        found1 = (tl.load(cache_map_occupied + index1) != 0) & (tl.load(cache_map_keys + index1) == current1)
        found2 = (tl.load(cache_map_occupied + index2) != 0) & (tl.load(cache_map_keys + index2) == current2)
        found3 = (tl.load(cache_map_occupied + index3) != 0) & (tl.load(cache_map_keys + index3) == current3)
        found4 = (tl.load(cache_map_occupied + index4) != 0) & (tl.load(cache_map_keys + index4) == current8)
        found5 = (tl.load(cache_map_occupied + index5) != 0) & (tl.load(cache_map_keys + index5) == current9)
        exact8_count = tl.where(found0, tl.load(cache_map_counts + index0 * 256 + byte), 0.0)
        exact6_count = tl.where(found1, tl.load(cache_map_counts + index1 * 256 + byte), 0.0)
        exact4_count = tl.where(found2, tl.load(cache_map_counts + index2 * 256 + byte), 0.0)
        exact2_count = tl.where(found3, tl.load(cache_map_counts + index3 * 256 + byte), 0.0)
        normalized5_count = tl.where(found4, tl.load(cache_map_counts + index4 * 256 + byte), 0.0)
        normalized3_count = tl.where(found5, tl.load(cache_map_counts + index5 * 256 + byte), 0.0)
        exact8_total = tl.where(found0, tl.load(cache_map_totals + index0), 0.0)
        exact6_total = tl.where(found1, tl.load(cache_map_totals + index1), 0.0)
        exact4_total = tl.where(found2, tl.load(cache_map_totals + index2), 0.0)
        exact2_total = tl.where(found3, tl.load(cache_map_totals + index3), 0.0)
        normalized5_total = tl.where(found4, tl.load(cache_map_totals + index4), 0.0)
        normalized3_total = tl.where(found5, tl.load(cache_map_totals + index5), 0.0)

        zero_key = current0 * 0
        recent_slot0 = _recent_map_slot(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, 0, current0, current4, current5, capacity=map_capacity)
        recent_slot1 = _recent_map_slot(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, 1, current0, current4, zero_key, capacity=map_capacity)
        recent_slot2 = _recent_map_slot(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, 2, current0, current6, zero_key, capacity=map_capacity)
        recent_slot3 = _recent_map_slot(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, 3, current0, current7, zero_key, capacity=map_capacity)
        recent_index0 = 0 * map_capacity + recent_slot0
        recent_index1 = 1 * map_capacity + recent_slot1
        recent_index2 = 2 * map_capacity + recent_slot2
        recent_index3 = 3 * map_capacity + recent_slot3
        recent24_found = (tl.load(recent_map_occupied + recent_index0) != 0) & (tl.load(recent_map_keys0 + recent_index0) == current0) & (tl.load(recent_map_keys1 + recent_index0) == current4) & (tl.load(recent_map_keys2 + recent_index0) == current5)
        recent16_found = (tl.load(recent_map_occupied + recent_index1) != 0) & (tl.load(recent_map_keys0 + recent_index1) == current0) & (tl.load(recent_map_keys1 + recent_index1) == current4) & (tl.load(recent_map_keys2 + recent_index1) == zero_key)
        recent12_found = (tl.load(recent_map_occupied + recent_index2) != 0) & (tl.load(recent_map_keys0 + recent_index2) == current0) & (tl.load(recent_map_keys1 + recent_index2) == current6) & (tl.load(recent_map_keys2 + recent_index2) == zero_key)
        recent10_found = (tl.load(recent_map_occupied + recent_index3) != 0) & (tl.load(recent_map_keys0 + recent_index3) == current0) & (tl.load(recent_map_keys1 + recent_index3) == current7) & (tl.load(recent_map_keys2 + recent_index3) == zero_key)
        recent24_position = tl.load(recent_map_latest + recent_index0, mask=recent24_found, other=0)
        recent16_position = tl.load(recent_map_latest + recent_index1, mask=recent16_found, other=0)
        recent12_position = tl.load(recent_map_latest + recent_index2, mask=recent12_found, other=0)
        recent10_position = tl.load(recent_map_latest + recent_index3, mask=recent10_found, other=0)
        recent24_found &= position - recent24_position <= window
        recent16_found &= position - recent16_position <= window
        recent12_found &= position - recent12_position <= window
        recent10_found &= position - recent10_position <= window
        recent24_target = tl.load(history + recent24_position, mask=recent24_found, other=0)
        recent16_target = tl.load(history + recent16_position, mask=recent16_found, other=0)
        recent12_target = tl.load(history + recent12_position, mask=recent12_found, other=0)
        recent10_target = tl.load(history + recent10_position, mask=recent10_found, other=0)

        probability = (exact8_count + exact8_strength * probability) / (exact8_total + exact8_strength)
        probability = (exact6_count + exact6_strength * probability) / (exact6_total + exact6_strength)
        probability = (exact4_count + exact4_strength * probability) / (exact4_total + exact4_strength)
        probability = (exact2_count + exact2_strength * probability) / (exact2_total + exact2_strength)
        recent24 = ((recent24_target == byte).to(tl.float32) + recent24_strength * probability) / (1.0 + recent24_strength)
        probability = tl.where(recent24_found, recent24, probability)
        recent16 = ((recent16_target == byte).to(tl.float32) + recent16_strength * probability) / (1.0 + recent16_strength)
        probability = tl.where(recent16_found, recent16, probability)
        recent12 = ((recent12_target == byte).to(tl.float32) + recent12_strength * probability) / (1.0 + recent12_strength)
        probability = tl.where(recent12_found, recent12, probability)
        recent10 = ((recent10_target == byte).to(tl.float32) + recent10_strength * probability) / (1.0 + recent10_strength)
        probability = tl.where(recent10_found, recent10, probability)
        probability = (normalized5_count + normalized5_strength * probability) / (normalized5_total + normalized5_strength)
        probability = (normalized3_count + normalized3_strength * probability) / (normalized3_total + normalized3_strength)
        selected = tl.argmax(probability, axis=0, tie_break_left=True).to(tl.int64)
        _recent_map_update(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, recent_map_latest, 0, current0, current4, current5, position, capacity=map_capacity)
        _recent_map_update(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, recent_map_latest, 1, current0, current4, zero_key, position, capacity=map_capacity)
        _recent_map_update(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, recent_map_latest, 2, current0, current6, zero_key, position, capacity=map_capacity)
        _recent_map_update(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, recent_map_latest, 3, current0, current7, zero_key, position, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 0, current0, selected, delta=1, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 1, current1, selected, delta=1, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 2, current2, selected, delta=1, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 3, current3, selected, delta=1, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 4, current8, selected, delta=1, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 5, current9, selected, delta=1, capacity=map_capacity)
        eviction_position = position - window
        eviction_target = tl.load(history + eviction_position).to(tl.int64)
        eviction0 = tl.load(cache_context_keys + 0 * cache_capacity + eviction_position)
        eviction1 = tl.load(cache_context_keys + 1 * cache_capacity + eviction_position)
        eviction2 = tl.load(cache_context_keys + 2 * cache_capacity + eviction_position)
        eviction3 = tl.load(cache_context_keys + 3 * cache_capacity + eviction_position)
        eviction8 = tl.load(cache_context_keys + 8 * cache_capacity + eviction_position)
        eviction9 = tl.load(cache_context_keys + 9 * cache_capacity + eviction_position)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 0, eviction0, eviction_target, delta=-1, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 1, eviction1, eviction_target, delta=-1, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 2, eviction2, eviction_target, delta=-1, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 3, eviction3, eviction_target, delta=-1, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 4, eviction8, eviction_target, delta=-1, capacity=map_capacity)
        _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 5, eviction9, eviction_target, delta=-1, capacity=map_capacity)
        tl.store(history + position, selected)
        tl.store(output, selected)


    @triton.jit
    def _certified_run_kernel(
        history,
        cache_context_keys,
        cache_capacity,
        cache_map_keys,
        cache_map_occupied,
        cache_map_counts,
        cache_map_totals,
        recent_map_keys0,
        recent_map_keys1,
        recent_map_keys2,
        recent_map_occupied,
        recent_map_latest,
        position,
        max_bytes,
        output_count,
        recent24_strength,
        recent16_strength,
        recent12_strength,
        recent10_strength,
        normalized5_strength,
        normalized3_strength,
        window: tl.constexpr,
        map_capacity: tl.constexpr,
    ):
        byte = tl.arange(0, 256)
        offset = 0
        running = True
        while (offset < max_bytes) & running:
            current_position = position + offset
            current0 = _pack_lags(history, current_position, start_lag=0, length=8, normalized=False)
            current1 = _pack_lags(history, current_position, start_lag=0, length=6, normalized=False)
            current2 = _pack_lags(history, current_position, start_lag=0, length=4, normalized=False)
            current3 = _pack_lags(history, current_position, start_lag=0, length=2, normalized=False)
            current4 = _pack_lags(history, current_position, start_lag=8, length=8, normalized=False)
            current5 = _pack_lags(history, current_position, start_lag=16, length=8, normalized=False)
            current6 = _pack_lags(history, current_position, start_lag=8, length=4, normalized=False)
            current7 = _pack_lags(history, current_position, start_lag=8, length=2, normalized=False)
            current8 = _pack_lags(history, current_position, start_lag=0, length=5, normalized=True)
            current9 = _pack_lags(history, current_position, start_lag=0, length=3, normalized=True)
            tl.store(cache_context_keys + 0 * cache_capacity + current_position, current0)
            tl.store(cache_context_keys + 1 * cache_capacity + current_position, current1)
            tl.store(cache_context_keys + 2 * cache_capacity + current_position, current2)
            tl.store(cache_context_keys + 3 * cache_capacity + current_position, current3)
            tl.store(cache_context_keys + 4 * cache_capacity + current_position, current4)
            tl.store(cache_context_keys + 5 * cache_capacity + current_position, current5)
            tl.store(cache_context_keys + 6 * cache_capacity + current_position, current6)
            tl.store(cache_context_keys + 7 * cache_capacity + current_position, current7)
            tl.store(cache_context_keys + 8 * cache_capacity + current_position, current8)
            tl.store(cache_context_keys + 9 * cache_capacity + current_position, current9)

            zero_key = current0 * 0
            recent_slot0 = _recent_map_slot(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, 0, current0, current4, current5, capacity=map_capacity)
            recent_slot1 = _recent_map_slot(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, 1, current0, current4, zero_key, capacity=map_capacity)
            recent_slot2 = _recent_map_slot(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, 2, current0, current6, zero_key, capacity=map_capacity)
            recent_slot3 = _recent_map_slot(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, 3, current0, current7, zero_key, capacity=map_capacity)
            recent_index0 = 0 * map_capacity + recent_slot0
            recent_index1 = 1 * map_capacity + recent_slot1
            recent_index2 = 2 * map_capacity + recent_slot2
            recent_index3 = 3 * map_capacity + recent_slot3
            found0 = (tl.load(recent_map_occupied + recent_index0) != 0) & (tl.load(recent_map_keys0 + recent_index0) == current0) & (tl.load(recent_map_keys1 + recent_index0) == current4) & (tl.load(recent_map_keys2 + recent_index0) == current5)
            found1 = (tl.load(recent_map_occupied + recent_index1) != 0) & (tl.load(recent_map_keys0 + recent_index1) == current0) & (tl.load(recent_map_keys1 + recent_index1) == current4) & (tl.load(recent_map_keys2 + recent_index1) == zero_key)
            found2 = (tl.load(recent_map_occupied + recent_index2) != 0) & (tl.load(recent_map_keys0 + recent_index2) == current0) & (tl.load(recent_map_keys1 + recent_index2) == current6) & (tl.load(recent_map_keys2 + recent_index2) == zero_key)
            found3 = (tl.load(recent_map_occupied + recent_index3) != 0) & (tl.load(recent_map_keys0 + recent_index3) == current0) & (tl.load(recent_map_keys1 + recent_index3) == current7) & (tl.load(recent_map_keys2 + recent_index3) == zero_key)
            latest0 = tl.load(recent_map_latest + recent_index0, mask=found0, other=0)
            latest1 = tl.load(recent_map_latest + recent_index1, mask=found1, other=0)
            latest2 = tl.load(recent_map_latest + recent_index2, mask=found2, other=0)
            latest3 = tl.load(recent_map_latest + recent_index3, mask=found3, other=0)
            found0 &= current_position - latest0 <= window
            found1 &= current_position - latest1 <= window
            found2 &= current_position - latest2 <= window
            found3 &= current_position - latest3 <= window
            target0 = tl.load(history + latest0, mask=found0, other=0)
            target1 = tl.load(history + latest1, mask=found1, other=0)
            target2 = tl.load(history + latest2, mask=found2, other=0)
            target3 = tl.load(history + latest3, mask=found3, other=0)

            lower = tl.zeros(byte.shape, tl.float32)
            upper = tl.full(byte.shape, 1.0, tl.float32)
            delta = (byte == target0).to(tl.float32)
            lower = tl.where(found0, (delta + recent24_strength * lower) / (1.0 + recent24_strength), lower)
            upper = tl.where(found0, (delta + recent24_strength * upper) / (1.0 + recent24_strength), upper)
            delta = (byte == target1).to(tl.float32)
            lower = tl.where(found1, (delta + recent16_strength * lower) / (1.0 + recent16_strength), lower)
            upper = tl.where(found1, (delta + recent16_strength * upper) / (1.0 + recent16_strength), upper)
            delta = (byte == target2).to(tl.float32)
            lower = tl.where(found2, (delta + recent12_strength * lower) / (1.0 + recent12_strength), lower)
            upper = tl.where(found2, (delta + recent12_strength * upper) / (1.0 + recent12_strength), upper)
            delta = (byte == target3).to(tl.float32)
            lower = tl.where(found3, (delta + recent10_strength * lower) / (1.0 + recent10_strength), lower)
            upper = tl.where(found3, (delta + recent10_strength * upper) / (1.0 + recent10_strength), upper)

            normalized_slot5 = _map_slot(cache_map_keys, cache_map_occupied, 4, current8, capacity=map_capacity)
            normalized_slot3 = _map_slot(cache_map_keys, cache_map_occupied, 5, current9, capacity=map_capacity)
            normalized_index5 = 4 * map_capacity + normalized_slot5
            normalized_index3 = 5 * map_capacity + normalized_slot3
            normalized_found5 = (tl.load(cache_map_occupied + normalized_index5) != 0) & (tl.load(cache_map_keys + normalized_index5) == current8)
            normalized_found3 = (tl.load(cache_map_occupied + normalized_index3) != 0) & (tl.load(cache_map_keys + normalized_index3) == current9)
            normalized_count5 = tl.where(normalized_found5, tl.load(cache_map_counts + normalized_index5 * 256 + byte), 0.0)
            normalized_count3 = tl.where(normalized_found3, tl.load(cache_map_counts + normalized_index3 * 256 + byte), 0.0)
            normalized_total5 = tl.where(normalized_found5, tl.load(cache_map_totals + normalized_index5), 0.0)
            normalized_total3 = tl.where(normalized_found3, tl.load(cache_map_totals + normalized_index3), 0.0)
            lower = (normalized_count5 + normalized5_strength * lower) / (normalized_total5 + normalized5_strength)
            upper = (normalized_count5 + normalized5_strength * upper) / (normalized_total5 + normalized5_strength)
            lower = (normalized_count3 + normalized3_strength * lower) / (normalized_total3 + normalized3_strength)
            upper = (normalized_count3 + normalized3_strength * upper) / (normalized_total3 + normalized3_strength)
            selected = tl.argmax(lower, axis=0, tie_break_left=True).to(tl.int64)
            selected_lower = tl.max(tl.where(byte == selected, lower, -1.0), axis=0)
            competitor_upper = tl.max(tl.where(byte == selected, -1.0, upper), axis=0)
            certified = selected_lower > competitor_upper
            if certified:
                _recent_map_update(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, recent_map_latest, 0, current0, current4, current5, current_position, capacity=map_capacity)
                _recent_map_update(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, recent_map_latest, 1, current0, current4, zero_key, current_position, capacity=map_capacity)
                _recent_map_update(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, recent_map_latest, 2, current0, current6, zero_key, current_position, capacity=map_capacity)
                _recent_map_update(recent_map_keys0, recent_map_keys1, recent_map_keys2, recent_map_occupied, recent_map_latest, 3, current0, current7, zero_key, current_position, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 0, current0, selected, delta=1, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 1, current1, selected, delta=1, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 2, current2, selected, delta=1, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 3, current3, selected, delta=1, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 4, current8, selected, delta=1, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 5, current9, selected, delta=1, capacity=map_capacity)
                eviction_position = current_position - window
                eviction_target = tl.load(history + eviction_position).to(tl.int64)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 0, tl.load(cache_context_keys + 0 * cache_capacity + eviction_position), eviction_target, delta=-1, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 1, tl.load(cache_context_keys + 1 * cache_capacity + eviction_position), eviction_target, delta=-1, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 2, tl.load(cache_context_keys + 2 * cache_capacity + eviction_position), eviction_target, delta=-1, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 3, tl.load(cache_context_keys + 3 * cache_capacity + eviction_position), eviction_target, delta=-1, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 4, tl.load(cache_context_keys + 8 * cache_capacity + eviction_position), eviction_target, delta=-1, capacity=map_capacity)
                _map_update(cache_map_keys, cache_map_occupied, cache_map_counts, cache_map_totals, 5, tl.load(cache_context_keys + 9 * cache_capacity + eviction_position), eviction_target, delta=-1, capacity=map_capacity)
                tl.store(history + current_position, selected)
                offset += 1
            else:
                running = False
        tl.store(output_count, offset)


def is_available() -> bool:
    return triton is not None and torch.cuda.is_available()


def is_recurrent_cached_available(model) -> bool:
    """Return whether the exact v24 bounded-memory GPU kernel applies.

    The certificate is independent of the neural local decoder, so both the
    recurrent 12-order host and compact positional hosts can use it.
    """
    return bool(
        is_available()
        and 1 <= model.count_cake.max_order <= 12
        and model.count_cake.backoff_mode == "fixed"
        and model.patch_size == 32
        and not model.local_continuous
        and not model.confidence_gate_enabled
        and tuple(order for order, _ in model.online_cache_specs) == (8, 6, 4, 2)
        and tuple(order for order, _ in model.recent_cache_specs) == (24, 16, 12, 10)
        and tuple(order for order, _ in model.normalized_cache_specs) == (5, 3)
        and model.online_cache_window == 768
        and model.cache_normalization == "classes"
    )


@torch.no_grad()
def fused_recurrent_cached_byte(
    model,
    history: torch.Tensor,
    cache_context_keys: torch.Tensor,
    cache_stats: torch.Tensor,
    cache_map_keys: torch.Tensor,
    cache_map_occupied: torch.Tensor,
    cache_map_counts: torch.Tensor,
    cache_map_totals: torch.Tensor,
    recent_map_keys0: torch.Tensor,
    recent_map_keys1: torch.Tensor,
    recent_map_keys2: torch.Tensor,
    recent_map_occupied: torch.Tensor,
    recent_map_latest: torch.Tensor,
    position: int,
    neural_probability: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    """Emit one exact greedy byte for the recurrent bounded-memory recipe."""
    if not is_recurrent_cached_available(model):
        raise ValueError("model is not compatible with recurrent cached GPU decode")
    if not history.is_cuda or history.dtype != torch.int64:
        raise ValueError("accelerated history must be a CUDA int64 tensor")
    if neural_probability.shape != (256,):
        raise ValueError("neural probability must contain 256 bytes")
    if (
        not cache_context_keys.is_cuda
        or cache_context_keys.dtype != torch.int64
        or cache_context_keys.ndim != 2
        or cache_context_keys.shape[0] != 10
        or cache_context_keys.shape[1] != history.numel()
    ):
        raise ValueError("accelerated cache contexts must have shape [10, capacity]")
    if (
        not cache_stats.is_cuda
        or cache_stats.dtype != torch.int64
        or cache_stats.shape != (1546,)
    ):
        raise ValueError("accelerated cache statistics must contain 1546 int64 values")
    cake = model.count_cake
    map_capacity = int(cache_map_keys.shape[1])
    if (
        cache_map_keys.shape != (6, map_capacity)
        or cache_map_keys.dtype != torch.int64
        or cache_map_occupied.shape != cache_map_keys.shape
        or cache_map_occupied.dtype != torch.uint8
        or cache_map_counts.shape != (6, map_capacity, 256)
        or cache_map_counts.dtype != torch.int32
        or cache_map_totals.shape != cache_map_keys.shape
        or cache_map_totals.dtype != torch.int32
        or map_capacity & (map_capacity - 1)
    ):
        raise ValueError("invalid accelerated rolling-cache map")
    output = torch.empty(1, device=history.device, dtype=torch.int64)
    table_args = []
    dimensions = {}
    for order in range(1, 13):
        table_args.extend(
            [
                getattr(cake, f"keys_{order}"),
                getattr(cake, f"counts_{order}"),
                getattr(cake, f"context_keys_{order}"),
                getattr(cake, f"context_totals_{order}"),
                getattr(cake, f"decode_starts_{order}"),
                getattr(cake, f"decode_ends_{order}"),
            ]
        )
        dimensions[f"n{order}"] = getattr(cake, f"keys_{order}").numel()
        dimensions[f"nc{order}"] = getattr(
            cake, f"context_keys_{order}"
        ).numel()
        if order >= 7:
            dimensions[f"hash{order}"] = (
                cake.context_hash_bits[order - 1]
                if order <= len(cake.context_hash_bits)
                else 0
            )
    strengths = tuple(float(value) for value in cake.backoff_strengths[:12])
    strengths += (1.0,) * (12 - len(strengths))
    cache_strengths = tuple(
        float(value)
        for specs in (
            model.online_cache_specs,
            model.recent_cache_specs,
            model.normalized_cache_specs,
        )
        for _, value in specs
    )
    _recurrent_cached_byte_kernel[(1,)](
        history,
        int(position),
        neural_probability.contiguous(),
        gate,
        output,
        cache_context_keys,
        cache_context_keys.shape[1],
        cache_stats,
        cache_map_keys,
        cache_map_occupied,
        cache_map_counts,
        cache_map_totals,
        recent_map_keys0,
        recent_map_keys1,
        recent_map_keys2,
        recent_map_occupied,
        recent_map_latest,
        cake.unigram_counts,
        *table_args,
        float(cake.unigram_counts.sum()),
        *strengths,
        *cache_strengths,
        **dimensions,
        window=int(model.online_cache_window),
        map_capacity=map_capacity,
        num_warps=8,
    )
    return output


class CountCakeGPUDecoder:
    """On-device decoder with exact bounded causal memory."""

    def __init__(self, model) -> None:
        if not is_recurrent_cached_available(model):
            raise ValueError("CountCakeGPUDecoder does not support this model recipe")
        self.model = model.eval()
        cake = self.model.count_cake
        for order in range(1, 13):
            if not hasattr(cake, f"keys_{order}"):
                empty = torch.empty(
                    0, device=cake.unigram_counts.device, dtype=torch.int64
                )
                # The fixed-shape Triton ABI carries twelve table slots.  A
                # compact lower-order cake fills unused slots with empty
                # runtime buffers; these are not learned state or serialized.
                setattr(cake, f"keys_{order}", empty)
                setattr(cake, f"counts_{order}", empty)
                setattr(cake, f"context_keys_{order}", empty)
                setattr(cake, f"context_totals_{order}", empty)
            keys = getattr(cake, f"keys_{order}")
            context_ids = keys >> 8
            starts = torch.nonzero(
                torch.cat(
                    [
                        torch.ones(1, device=keys.device, dtype=torch.bool),
                        context_ids[1:] != context_ids[:-1],
                    ]
                ),
                as_tuple=False,
            ).flatten().to(torch.int32)
            ends = torch.cat(
                [
                    starts[1:],
                    torch.tensor(
                        [keys.numel()], device=keys.device, dtype=torch.int32
                    ),
                ]
            )
            setattr(cake, f"decode_starts_{order}", starts.contiguous())
            setattr(cake, f"decode_ends_{order}", ends.contiguous())

    def prepare(self, state: dict, *, generated_bytes: int) -> None:
        if generated_bytes <= 0:
            raise ValueError("generated_bytes must be positive")
        full_history = state.get("full_history")
        if full_history is None or not full_history.is_cuda:
            raise ValueError("generation state is missing its CUDA prompt history")
        buffer = torch.empty(
            full_history.numel() + int(generated_bytes),
            device=full_history.device,
            dtype=torch.int64,
        )
        buffer[: full_history.numel()].copy_(full_history)
        cache_context_keys = torch.empty(
            (10, buffer.numel()),
            device=buffer.device,
            dtype=torch.int64,
        )
        block_size = 128
        _prefill_context_keys_kernel[(triton.cdiv(full_history.numel(), block_size),)](
            buffer,
            cache_context_keys,
            full_history.numel(),
            buffer.numel(),
            block_size=block_size,
            num_warps=4,
        )
        state["gpu_history"] = buffer
        state["gpu_cache_context_keys"] = cache_context_keys
        state["gpu_cache_stats"] = torch.empty(
            1546, device=buffer.device, dtype=torch.int64
        )
        map_capacity = 8192
        map_keys = torch.zeros((6, map_capacity), dtype=torch.int64)
        map_occupied = torch.zeros((6, map_capacity), dtype=torch.uint8)
        map_counts = torch.zeros((6, map_capacity, 256), dtype=torch.int32)
        map_totals = torch.zeros((6, map_capacity), dtype=torch.int32)
        composite = state["online_cache"]
        cache_tables = list(composite.exact._counts) + list(
            composite.normalized._counts
        )
        for stage, table in enumerate(cache_tables):
            for context, continuations in table.items():
                unsigned_key = int.from_bytes(
                    context[::-1], "little", signed=False
                )
                signed_key = (
                    unsigned_key
                    if unsigned_key < (1 << 63)
                    else unsigned_key - (1 << 64)
                )
                slot = (unsigned_key ^ (unsigned_key >> 32)) & (
                    map_capacity - 1
                )
                while map_occupied[stage, slot]:
                    if int(map_keys[stage, slot]) == signed_key:
                        break
                    slot = (slot + 1) & (map_capacity - 1)
                map_occupied[stage, slot] = 1
                map_keys[stage, slot] = signed_key
                for target, count in continuations.items():
                    map_counts[stage, slot, int(target)] = int(count)
                    map_totals[stage, slot] += int(count)
        state["gpu_cache_map_keys"] = map_keys.to(buffer.device)
        state["gpu_cache_map_occupied"] = map_occupied.to(buffer.device)
        state["gpu_cache_map_counts"] = map_counts.to(buffer.device)
        state["gpu_cache_map_totals"] = map_totals.to(buffer.device)
        recent_keys0 = torch.zeros((4, map_capacity), dtype=torch.int64)
        recent_keys1 = torch.zeros((4, map_capacity), dtype=torch.int64)
        recent_keys2 = torch.zeros((4, map_capacity), dtype=torch.int64)
        recent_occupied = torch.zeros((4, map_capacity), dtype=torch.uint8)
        recent_latest = torch.zeros((4, map_capacity), dtype=torch.int32)

        def packed_chunk(payload: bytes) -> tuple[int, int]:
            unsigned = int.from_bytes(payload, "little", signed=False)
            signed = unsigned if unsigned < (1 << 63) else unsigned - (1 << 64)
            return unsigned, signed

        for stage, table in enumerate(composite.recent._recent):
            for context, (_, latest_position) in table.items():
                newest_first = context[::-1]
                unsigned0, signed0 = packed_chunk(newest_first[:8])
                unsigned1, signed1 = packed_chunk(newest_first[8:16])
                unsigned2, signed2 = packed_chunk(newest_first[16:24])
                slot = (
                    unsigned0
                    ^ (unsigned0 >> 32)
                    ^ unsigned1
                    ^ (unsigned1 >> 29)
                    ^ unsigned2
                ) & (map_capacity - 1)
                while recent_occupied[stage, slot]:
                    if (
                        int(recent_keys0[stage, slot]) == signed0
                        and int(recent_keys1[stage, slot]) == signed1
                        and int(recent_keys2[stage, slot]) == signed2
                    ):
                        break
                    slot = (slot + 1) & (map_capacity - 1)
                recent_occupied[stage, slot] = 1
                recent_keys0[stage, slot] = signed0
                recent_keys1[stage, slot] = signed1
                recent_keys2[stage, slot] = signed2
                recent_latest[stage, slot] = int(latest_position)
        state["gpu_recent_map_keys0"] = recent_keys0.to(buffer.device)
        state["gpu_recent_map_keys1"] = recent_keys1.to(buffer.device)
        state["gpu_recent_map_keys2"] = recent_keys2.to(buffer.device)
        state["gpu_recent_map_occupied"] = recent_occupied.to(buffer.device)
        state["gpu_recent_map_latest"] = recent_latest.to(buffer.device)
        state["gpu_certified_count"] = torch.empty(
            1, device=buffer.device, dtype=torch.int32
        )
        state["gpu_position"] = int(full_history.numel())

    def _certified_run(
        self,
        state: dict,
        *,
        position: int,
        max_bytes: int,
    ) -> int:
        model = self.model
        recent_strengths = [value for _, value in model.recent_cache_specs]
        normalized_strengths = [
            value for _, value in model.normalized_cache_specs
        ]
        map_capacity = state["gpu_cache_map_keys"].shape[1]
        _certified_run_kernel[(1,)](
            state["gpu_history"],
            state["gpu_cache_context_keys"],
            state["gpu_cache_context_keys"].shape[1],
            state["gpu_cache_map_keys"],
            state["gpu_cache_map_occupied"],
            state["gpu_cache_map_counts"],
            state["gpu_cache_map_totals"],
            state["gpu_recent_map_keys0"],
            state["gpu_recent_map_keys1"],
            state["gpu_recent_map_keys2"],
            state["gpu_recent_map_occupied"],
            state["gpu_recent_map_latest"],
            int(position),
            int(max_bytes),
            state["gpu_certified_count"],
            *recent_strengths,
            *normalized_strengths,
            window=int(model.online_cache_window),
            map_capacity=int(map_capacity),
            num_warps=8,
        )
        return int(state["gpu_certified_count"].item())

    @torch.no_grad()
    def generate_cached(self, state: dict, *, patches: int = 1) -> torch.Tensor:
        if patches <= 0:
            raise ValueError("patches must be positive")
        model = self.model
        history = state.get("gpu_history")
        cache_context_keys = state.get("gpu_cache_context_keys")
        cache_stats = state.get("gpu_cache_stats")
        cache_map_keys = state.get("gpu_cache_map_keys")
        cache_map_occupied = state.get("gpu_cache_map_occupied")
        cache_map_counts = state.get("gpu_cache_map_counts")
        cache_map_totals = state.get("gpu_cache_map_totals")
        recent_map_keys0 = state.get("gpu_recent_map_keys0")
        recent_map_keys1 = state.get("gpu_recent_map_keys1")
        recent_map_keys2 = state.get("gpu_recent_map_keys2")
        recent_map_occupied = state.get("gpu_recent_map_occupied")
        recent_map_latest = state.get("gpu_recent_map_latest")
        if history is None:
            raise ValueError("call prepare before accelerated generation")
        position = int(state["gpu_position"])
        required = patches * model.patch_size
        if position + required > history.numel():
            raise ValueError("prepared GPU history capacity is too small")
        output_start = position
        output_end = position + required
        high_values = torch.arange(16, device=history.device)

        def start_local_patch():
            context = state["recurrent_state"].squeeze(0)
            composed = context + model.from_abi(model.to_abi(context))
            if model.local_recurrent:
                return (
                    model.local_projection(composed).unsqueeze(0),
                    model.local_bos.reshape(1, 1, -1),
                    None,
                    None,
                )
            positions = torch.arange(model.patch_size, device=history.device)
            local = model.local_norm(
                model.local_projection(composed).unsqueeze(-2)
                + model.local_positions(positions).unsqueeze(0)
            )[0]
            high_log_probability = torch.log_softmax(
                model.high_head(local), dim=-1
            )
            low_hidden = model.low_norm(
                local.unsqueeze(-2)
                * (1.0 + model.high_scale(high_values))
                + model.high_embedding(high_values)
            )
            low_log_probability = torch.log_softmax(
                model.low_head(low_hidden), dim=-1
            )
            probability = (
                high_log_probability.unsqueeze(-1) + low_log_probability
            ).flatten(-2).exp()
            gates = torch.sigmoid(model.mixture_gate(local)).squeeze(-1)
            return None, None, probability, gates

        local_state, local_input, patch_neural, patch_gates = start_local_patch()
        patch_start = position
        offset = 0
        certified_bytes = 0
        exact_bytes = 0
        certificate_launches = 0
        while position < output_end:
            # Certification depends only on the causal byte-cache state, not on
            # the neural recurrent state.  Let the device prove the longest run
            # across patch boundaries, then replay that run through the small
            # recurrent cores in patch-sized batches.  This preserves exact
            # greedy semantics while avoiding one host synchronization per
            # patch for the common certified case.
            certified = self._certified_run(
                state,
                position=position,
                max_bytes=output_end - position,
            )
            certificate_launches += 1
            if certified:
                certified_bytes += certified
                remaining = certified
                while remaining:
                    take = min(remaining, model.patch_size - offset)
                    emitted = history[
                        position : position + take
                    ].reshape(1, -1)
                    if model.local_recurrent:
                        teacher = torch.cat(
                            [
                                local_input,
                                model.byte_embedding(emitted[:, :-1]),
                            ],
                            dim=1,
                        )
                        _, local_state = model.local_core(teacher, local_state)
                        local_input = model.byte_embedding(
                            emitted[:, -1]
                        ).reshape(1, 1, -1)
                    position += take
                    offset += take
                    remaining -= take
                    if offset == model.patch_size:
                        patch = history[patch_start:position].reshape(
                            1, model.patch_size
                        )
                        feature = torch.tanh(
                            model.patch_projection(
                                model.byte_embedding(patch).flatten(-2)
                            )
                        ).unsqueeze(1)
                        _, state["recurrent_state"] = model.patch_core(
                            feature, state["recurrent_state"]
                        )
                        patch_start = position
                        offset = 0
                        if position < output_end:
                            (
                                local_state,
                                local_input,
                                patch_neural,
                                patch_gates,
                            ) = start_local_patch()
                continue

            if model.local_recurrent:
                _, local_state = model.local_core(local_input, local_state)
                local = model.local_norm(
                    local_state.squeeze(0) + model.local_positions.weight[offset]
                )[0]
                high_log_probability = torch.log_softmax(
                    model.high_head(local), dim=-1
                )
                low_hidden = model.low_norm(
                    local.unsqueeze(0)
                    * (1.0 + model.high_scale(high_values))
                    + model.high_embedding(high_values)
                )
                low_log_probability = torch.log_softmax(
                    model.low_head(low_hidden), dim=-1
                )
                neural = (
                    high_log_probability.unsqueeze(-1) + low_log_probability
                ).flatten().exp()
                gate = torch.sigmoid(model.mixture_gate(local)).squeeze()
            else:
                neural = patch_neural[offset]
                gate = patch_gates[offset]
            next_byte = fused_recurrent_cached_byte(
                model,
                history,
                cache_context_keys,
                cache_stats,
                cache_map_keys,
                cache_map_occupied,
                cache_map_counts,
                cache_map_totals,
                recent_map_keys0,
                recent_map_keys1,
                recent_map_keys2,
                recent_map_occupied,
                recent_map_latest,
                position,
                neural,
                gate,
            )
            exact_bytes += 1
            position += 1
            offset += 1
            if model.local_recurrent:
                local_input = model.byte_embedding(next_byte).reshape(1, 1, -1)
            if offset == model.patch_size:
                patch = history[patch_start:position].reshape(1, model.patch_size)
                feature = torch.tanh(
                    model.patch_projection(model.byte_embedding(patch).flatten(-2))
                ).unsqueeze(1)
                _, state["recurrent_state"] = model.patch_core(
                    feature, state["recurrent_state"]
                )
                patch_start = position
                offset = 0
                if position < output_end:
                    (
                        local_state,
                        local_input,
                        patch_neural,
                        patch_gates,
                    ) = start_local_patch()
        if offset:
            raise RuntimeError("generation ended inside a patch")
        state["gpu_certified_bytes"] = certified_bytes
        state["gpu_exact_bytes"] = exact_bytes
        state["gpu_certificate_launches"] = certificate_launches
        state["gpu_position"] = position
        state["history"] = history[
            position - model.count_cake.max_order : position
        ].clone()
        return history[output_start:position].reshape(1, -1)


@torch.no_grad()
def fused_greedy_patch(model, context: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    """Generate one exact greedy patch with one fused CUDA kernel launch."""
    if not is_available():
        raise RuntimeError("Triton CUDA generation is unavailable")
    if context.shape[0] != 1 or history.ndim != 1:
        raise ValueError("fused generation currently supports batch size one")
    cake = model.count_cake
    if cake.max_order != 4 or model.patch_size != 32:
        raise ValueError("the production kernel requires order four and patch size 32")
    if not context.is_cuda or not history.is_cuda:
        raise ValueError("fused generation requires CUDA tensors")

    composed = context + model.from_abi(model.to_abi(context))
    positions = torch.arange(model.patch_size, device=context.device)
    local = model.local_norm(
        model.local_projection(composed).unsqueeze(-2)
        + model.local_positions(positions).unsqueeze(0)
    )[0]
    high_log_probability = F.log_softmax(model.high_head(local), dim=-1)
    high_values = torch.arange(16, device=context.device)
    low_hidden = model.low_norm(
        local.unsqueeze(-2)
        * (1.0 + model.high_scale(high_values))
        + model.high_embedding(high_values)
    )
    low_log_probability = F.log_softmax(model.low_head(low_hidden), dim=-1)
    neural_probability = (
        high_log_probability.unsqueeze(-1) + low_log_probability
    ).flatten(-2).exp().contiguous()
    gates = torch.sigmoid(model.mixture_gate(local)).squeeze(-1).contiguous()
    output = torch.empty(model.patch_size, device=context.device, dtype=torch.int64)
    history = history[-4:].to(dtype=torch.int64).contiguous()

    _greedy_patch_kernel[(1,)](
        history,
        neural_probability,
        gates,
        output,
        cake.unigram_counts,
        cake.keys_1,
        cake.counts_1,
        cake.context_keys_1,
        cake.context_totals_1,
        cake.keys_2,
        cake.counts_2,
        cake.context_keys_2,
        cake.context_totals_2,
        cake.keys_3,
        cake.counts_3,
        cake.context_keys_3,
        cake.context_totals_3,
        cake.keys_4,
        cake.counts_4,
        cake.context_keys_4,
        cake.context_totals_4,
        float(cake.unigram_counts.sum()),
        n1=cake.keys_1.numel(),
        nc1=cake.context_keys_1.numel(),
        n2=cake.keys_2.numel(),
        nc2=cake.context_keys_2.numel(),
        n3=cake.keys_3.numel(),
        nc3=cake.context_keys_3.numel(),
        n4=cake.keys_4.numel(),
        nc4=cake.context_keys_4.numel(),
        patch_size=model.patch_size,
        num_warps=8,
    )
    return output.unsqueeze(0)
