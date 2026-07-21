# Moonshot claims and limitations

## Established by the executable smoke suite

- The new foundation is tokenizer-free, causal, and physically top-1 sparse. Its default
  active fraction is under 20%, and tests show inactive experts receive no gradients.
- The baseline is a learned-BPE modern causal SwiGLU transformer, parameter matched within
  5% in the recorded run.
- `.cake` packaging is non-executable, schema-closed, safetensors-based, content addressed,
  Ed25519-capable, ABI strict, atomic, and covered by adversarial tests.
- Five domains install and route; the locked eight-case routing smoke currently records
  100% route accuracy/top-k recall and 0% false activation after conservative injection
  abstention.
- The same portable-decoder package remains bit-identical and produces zero logit
  difference on the declared deterministic FP32 CPU path across three receiver descriptors.
- Uninstall/reinstall preserves the payload hash. TorchScript export reloads with zero
  logit difference.

## Not established

The smoke corpus is tiny. Five portable cakes are genuinely trained and evaluated by
held-out BPB with random/wrong-domain controls, but no ordinary specialist task error
metric clears the required 5x reduction. The run does not establish general language
quality, a 5x domain-error reduction, 5x foundation time-to-quality,
5x inference at non-inferior quality, realistic mixed-domain task accuracy, energy wins,
or physical mobile performance. The measured smoke CPU speed ratio is below 5x and GPU is
also below 5x. The release certificate therefore cannot pass.

Existing North Star certificates retain their original narrow scopes. None is widened by
this work. Retrieval/transducer results elsewhere in the repository remain separate from
neural autoregressive generation.

The highest-value next experiment is a medium-scale, leakage-locked five-seed run on a
substantially larger public corpus, with enough steps for both same-scale models to cross a
predeclared BPB threshold, followed by trained domain cakes and the held-out mixed-domain
orchestration suite.
