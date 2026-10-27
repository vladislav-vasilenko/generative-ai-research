# Kandinsky 6: ComfyUI baseline и исследование MLX / voice conditioning

[Общий учебный маршрут Lecture 1–5 / Lab 1–3](../curriculum/README.md) связывает математические предпосылки, PyTorch и ComfyUI с этими исследованиями. Начните с него, если важно понимать порядок и критерии готовности.

Начните с [пошагового Jupyter Notebook](notebooks/00_manual_experiment_comfy_mlx_film.ipynb): проверка диска и моделей, ComfyUI baseline, учебные примеры FiLM/attention/Flow Matching, VCTK, MLX parity и план оценки голоса. Run All не скачивает большие веса и не запускает генерацию.

**Полный запуск отложен до освобождения 50 GB+ на SSD.** Нужные в Desktop Qwen 15.45 GB и Lite 5.94 GB — оставшиеся компоненты минимального workflow; учебные ячейки доступны без них.

Подготовлены два облегчённых шаблона **Lite Distill**:

| Workflow | Размеры / длина | Назначение |
|---|---|---|
| [Smoke](workflows/k6_lite_distill_smoke.json) | 432×240, 33 frames | Проверка установки и ресурсов |
| [Baseline](workflows/k6_lite_distill_baseline.json) | 864×480, 121 frames | 5s профиль для замеров |

Локально они установлены в Templates → kandinsky6 и Workflows → K6 Research под именами **Kandinsky 6 Lite - Smoke / Baseline**. Можно также перетащить JSON в ComfyUI. Оба используют 10 steps, CFG=1, seed=42 fixed, audio scale=.417; 12 узлов, без Pro, prompt beautifier, SR и MagCache. **Связи и JSON проверены; полноценная генерация пока не проверена.** Скачивание узлов не означает, что скачаны модели.

[Подробная ручная инструкция и протокол эксперимента](COMFYUI_AND_EXPERIMENT_RU.md) содержит точные пути вашей установки, модели, команды, метрики, ограничения и критерии выводов.

В `labkit/` находится experimental audio-only MLX forward, в `scripts/` — PyTorch reference, parity checks, inspection и извлечение audio-параметров из локального checkpoint. Это **не полный MLX-порт совместного π-Flow pipeline**. Голосовой адаптер ещё не обучен; end-to-end voice cloning не заявляется. Статусы и измерения сохраняются в reports; большие веса, VCTK аудио и virtualenv в Git не включаются.

## Источники

- [K6 report](https://arxiv.org/html/2610.05608v1), [официальный K6](https://github.com/kandinskylab/kandinsky-6).
- [Voice conditioning paper](https://arxiv.org/html/2608.15690v1): full audio-backbone fine-tuning + FiLM; frozen adapter-only — отдельная гипотеза.
- [KVAE report](https://arxiv.org/html/2608.05798v1), [KVAE 2.0](https://huggingface.co/collections/kandinskylab/kvae-20), [KVAE-Audio](https://huggingface.co/collections/kandinskylab/kvae-audio).
- [Закреплённые revisions и SHA-256](reference/source-manifest.json). Upstream source snapshots сохраняют собственные LICENSE.

Базовый K6 использует HunyuanVideo и MMAudio; отдельные KVAE 2.0/Audio не объявляются совместимой заменой его кодеков. Сведения о M3 performance требуют наших измерений; результаты H100 не переносятся на Mac по размеру весов.

[Проверка учебного notebook и границы результатов](reports/VALIDATION.md).

## Структура и разработка

Материалы K6 находятся в ветке `kandinsky-6-labs` и каталоге `kandinsky-6/` основного репозитория. `notebooks/` — учебный маршрут, `workflows/` — JSON для Desktop, `scripts/` — исполняемые инструменты, `labkit/` — исследовательский код, `reference/` — pinned upstream snapshots, `reports/` — проверенные результаты и пустые таблицы замеров. `artifacts/` содержит локальные VCTK/cache/fixtures и игнорируется Git. Временная папка сборки не является частью проекта и не нужна для запуска.
