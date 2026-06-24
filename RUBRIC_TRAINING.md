# Rubric training

Rubrics define what is allowed to change and what must remain protected.

Required fields are deliberately compact:

- `rubric_id`;
- `name`;
- `branch`;
- `max_steps`;
- `trainable_modules`;
- `frozen_modules`;
- `protected_capabilities`;
- `gates`;
- `rollback_policy`.

Example rubrics are stored in `rubrics/`. The toy sequence is
`rubrics/sequences/toy_sequence.yaml`.

Gate types currently include metric minimum/maximum, regression gates, ABI hash
compatibility, input-interface compatibility, and byte-patch compatibility. Future
LayerCake scale rubrics should add generation-quality, CPU latency, memory, transfer
exactness, and receiver-after-transfer gates before any claim promotion.
