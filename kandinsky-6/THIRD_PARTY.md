# Source provenance

The `reference/` directory contains selected source files, model cards and configs from the public Kandinsky 6, KVAE and KVAE-Audio repositories. Revisions, source URLs and SHA-256 values are recorded in `reference/source-manifest.json`. Each source repository's LICENSE is preserved. These are research reference snapshots, not silently installed application dependencies.

The two Lite generation workflows are derived from the official Kandinsky 6 text-to-video+audio example at revision `7ceeefad7fb625011f7db602c43893a989f19c15` (MIT). Their transformer selection, wiring, captions and geometry were changed for the research baseline; beautifier, super-resolution and MagCache nodes were removed.

VCTK audio is not redistributed in Git. The preparation script records the public mirror, pinned revision, ZIP member checksums and original utterance IDs. Users must retain the corpus attribution and licence with any dataset copies. The small pilot is not the benchmark or dataset distributed with a voice-cloning paper.

Papers are linked by version. Claims in user-provided pasted advice are treated as hypotheses to audit, not as authoritative implementation or performance guarantees.

The separate `workflows/k6_sr_standalone_2x.json` and `k6_sr_standalone_2_25x.json` are modified versions of the [official Kandinsky Lab SR example](https://github.com/kandinskylab/kandinsky-6-sr/blob/b15e2aac526519c51f59dfec2b848355bb8902de/comfyui/example_workflows/Kandinsky%206.0%20Video%20Super%20Resolution.json). The installed template matches that pinned file: SHA-256 `da30efcaab1c4c0dcb0065a4abd4a1ad886d0b205f7c7b6eceaf33e422dfca4d`. The SR extension uses Apache-2.0; its [original LICENSE is retained](workflows/licenses/kandinsky-6-sr-LICENSE). Changes dated 2026-10-30: Baseline MP4 input, fixed seed, distilled two-step settings, tile batch 1, scale profiles, native output dimensions, MP4/H.264 saving and study notes. Original source-audio connections and model-download metadata are retained. A modification notice is embedded in each derived JSON.

SR weights are not redistributed. Their [model card](https://huggingface.co/kandinskylab/Kandinsky-6.0-VSR-distilled2steps-5s-Diffusers) lists MIT; this is separate from the SR extension's Apache-2.0 source licence. `workflows/sr-model-manifest.json` records the model repository, revision, artifact sizes and hashes without bundling weights or the local Baseline video.
