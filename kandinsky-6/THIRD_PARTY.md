# Source provenance

The `reference/` directory contains selected source files, model cards and configs from the public Kandinsky 6, KVAE and KVAE-Audio repositories. Revisions, source URLs and SHA-256 values are recorded in `reference/source-manifest.json`. Each source repository's LICENSE is preserved. These are research reference snapshots, not silently installed application dependencies.

The two workflows are derived from the official Kandinsky 6 text-to-video+audio example at revision `7ceeefad7fb625011f7db602c43893a989f19c15` (MIT). Their transformer selection, wiring, captions and geometry were changed for the research baseline; beautifier, super-resolution and MagCache nodes were removed.

VCTK audio is not redistributed in Git. The preparation script records the public mirror, pinned revision, ZIP member checksums and original utterance IDs. Users must retain the corpus attribution and licence with any dataset copies. The small pilot is not the benchmark or dataset distributed with a voice-cloning paper.

Papers are linked by version. Claims in user-provided pasted advice are treated as hypotheses to audit, not as authoritative implementation or performance guarantees.
