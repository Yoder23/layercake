from _common import ROOT

import json
import zipfile

import torch

from layercake.cake.manifest import CakeManifest
from layercake.cake.package import load_package
from layercake.models.portable_decoder import load_cake_module
from layercake.runtime.cpu import quantize_dynamic
from layercake.training.cake import evaluate_fusion_bpb
from layercake.training.data import ByteCorpus, sha256_file
from layercake.training.foundation import load_core_checkpoint


cake_path = ROOT / "artifacts/cakes/python.cake"
with zipfile.ZipFile(cake_path) as archive:
    key_id = CakeManifest.from_json(archive.read("manifest.json")).signature["key_id"]
package = load_package(
    cake_path, trust_store={key_id: ROOT / "artifacts/cakes/python.public.pem"}
)
core, metadata = load_core_checkpoint(ROOT / "artifacts/cores/english-core-a", device="cpu")
float_cake = load_cake_module(package).cpu().eval()
int8_cake = quantize_dynamic(load_cake_module(package).cpu().eval())
corpus = ByteCorpus(ROOT / "data/moonshot/v2/python/python_test.bin")
kwargs = {
    "batch_size": 8, "sequence_bytes": 256, "batches": 8,
    "device": torch.device("cpu"), "route": int(metadata["route"]),
}
floating = evaluate_fusion_bpb(core, float_cake, corpus, **kwargs)
quantized = evaluate_fusion_bpb(core, int8_cake, corpus, **kwargs)
relative = quantized["cake_bits_per_byte"] / floating["cake_bits_per_byte"]
result = {
    "format": "layercake-cake-quantization/2",
    "status": "PASS" if relative <= 1.02 and quantized["cake_bits_per_byte"] < quantized["core_bits_per_byte"] else "FAIL",
    "method": "PyTorch dynamic qint8 Linear and GRU; runtime packed weights, not a portable int8 cake archive",
    "cake_archive_sha256": sha256_file(cake_path),
    "float": floating,
    "dynamic_int8": quantized,
    "int8_over_float_bpb": relative,
    "limitations": "no custom native int8 kernel and no quantized package format",
}
output = ROOT / "results/moonshot/v2/quantization_evidence.json"
output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["status"] == "PASS" else 1)
