# CLASS-5.v0.5.16 — retract-vs-offload drain+catch hotfix for v0.5.16 (K3-base proxy)

Same bug and same composition as hotfix/RETRACT-OFFLOAD-DRAIN.{v0.5.14,
.v0.5.15.post1, v0.5.17, v0.5.20} (see RETRACT-OFFLOAD-DRAIN.README.md for the
full writeup): drain_before_retraction VERBATIM from dev (manager fields +
tp scaffold already present in .16: tp_group/tp_world_size/check_offload_progress
all exist, MIN all-reduce pattern at offload manager :208 pre-exists), plus
env knob SGLANG_DECODE_OFFLOAD_DRAIN_TIMEOUT_S=EnvFloat(30.0), plus the two
scheduler catch guards (OOM-retract defer via `return batch` — .16's consumer
`ret = running_batch if not running_batch.is_empty()` derefs the return, so
None is unsafe exactly like the other tags; pause-refuse restores
_engine_paused=False and logger.criticals). .16 differs from .15p1 only in
line offsets + the `retract_decode(self.server_args)` signature; no deeper
rework needed — pSwa/SWA offload rides the same ack_write_queue and is
therefore covered by the same drain.

Apply: `git apply --check` VERIFIED against clean v0.5.16 tag (d21f3c3a) AND
against fdebc938f7f4 (identical trees for the three touched files). Tree-side
smoke: drain extracted from the patched source executes happy path + wedged
RuntimeError under SGLANG_DECODE_OFFLOAD_DRAIN_TIMEOUT_S=0.05 (full module
import in this sandbox fails on a transformers-registry clash in
models/qwen3_asr.py — pre-existing env-vs-tag version drift, untouched file,
unrelated to this patch). K3 private branch: apply this patch with
`git apply --3way` if anchor matches (.15p1/.16 shapes on the three files).
