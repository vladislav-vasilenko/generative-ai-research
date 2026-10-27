# Working on this study stand

For the requested methodological/editorial review, follow [curriculum/CLAUDE_REVIEW_BRIEF.md](curriculum/CLAUDE_REVIEW_BRIEF.md). Start with [README.md](README.md), [curriculum/CONTENTS.md](curriculum/CONTENTS.md) and the existing execution reports.

The learning skeleton is MIT 6.S184 (2026). Keep a single coherent route from lectures and small Python/PyTorch labs to industrial components, ComfyUI and a separate research protocol. K6 component notebooks 07 (DiT/text CrossDiT) and 08 (dual-stream audio/video CrossDiT) need particular source-level care.

Preserve the user's uncommitted work in `lecture_01/lab_one_ru.ipynb` and `lecture_01/lab_one_ru_exercises.ipynb`; do not stage or rewrite it. Canonical MIT originals and pinned third-party snapshots remain unchanged. Keep notebook generators in sync with generated artifacts.

The user postponed large pretrained downloads and generation until at least 50 GB of SSD space is free. Do not trigger these jobs, install MLX or change ComfyUI Desktop's environment as part of editorial work. Small CPU checks use the root locked environment. Do not claim pretrained inference, joint MLX parity or voice cloning from synthetic tests.

See the review brief for source links, rendering checks, commands and acceptance criteria. Complete ordinary reversible edits and relevant verification without unnecessary confirmation; do not merge a new PR without a user request covering that merge.
