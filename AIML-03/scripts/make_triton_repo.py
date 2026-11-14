"""Lay out a Triton model repository from the bundle's SavedModels (TensorFlow backend, CPU)."""
import shutil, sys, json
from pathlib import Path

bundle, repo = Path(sys.argv[1]), Path(sys.argv[2])
n_dense = json.loads((bundle / "manifest.json").read_text())["n_dense"]
CONFIG = """name: "{name}"
platform: "tensorflow_savedmodel"
max_batch_size: 0
input [
  {{ name: "dense" data_type: TYPE_FP32 dims: [ -1, {n_dense} ] }},
  {{ name: "year"  data_type: TYPE_INT32 dims: [ -1 ] }},
  {{ name: "pop"   data_type: TYPE_INT32 dims: [ -1 ] }}
]
output [
  {{ name: "logit" data_type: TYPE_FP32 dims: [ -1 ] }}
]
instance_group [ {{ kind: KIND_CPU count: 1 }} ]
"""
for key, name in (("dcn", "dcn_ranker"), ("wide_deep", "wide_deep_ranker")):
    d = repo / name; shutil.rmtree(d, ignore_errors=True); (d / "1").mkdir(parents=True)
    shutil.copytree(bundle / "saved_models" / key, d / "1" / "model.savedmodel")
    (d / "config.pbtxt").write_text(CONFIG.format(name=name, n_dense=n_dense))
    print("wrote", d)
