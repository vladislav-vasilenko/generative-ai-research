# Kandinsky 6: ручной запуск и протокол исследования

<!-- study-navigation:start -->
**Основа:** [MIT 6.S184 / 2026](https://diffusion.csail.mit.edu/2026/index.html) · [Основной конспект](https://diffusion.csail.mit.edu/2026/docs/lecture_notes.pdf) · [Учебная карта](../curriculum/README.md) · [Полное содержание](../curriculum/CONTENTS.md).
<!-- study-navigation:end -->

## Что уже готово и что ещё не проверено

Подготовлены два облегчённых ComfyUI workflow, скрипты проверки файлов и памяти, подготовка VCTK, PyTorch reference и **прототип audio-forward на MLX**. Это не законченный MLX-порт всего K6 и не готовое клонирование голоса. Обучение адаптера на pretrained-весах, waveform sampling с ним и custom node для подключения обученного FiLM пока не реализованы как проверенный сквозной маршрут.

| Этап | Состояние |
|---|---|
| Официальные источники и model configs | Закреплены revisions и SHA-256 в reference/source-manifest.json |
| Два Lite workflow | JSON/links проверены; Baseline 864×480 / 121 кадр сохранён на MPS через tiled decode, Sampler cache переиспользован |
| VCTK pilot | 120 клипов, четыре спикера; reference/train/validation/test; глобальная проверка пересечений текстов |
| PyTorch reference | Малый настоящий K6 класс выполнен на CPU, 50 632 audio-параметра; случайные веса, не pretrained |
| MLX numerical parity | Скрипт подготовлен; результат определяется только reports/mlx-*-parity.json с passed=true |
| Pretrained audio weights | Локальный полный Lite checkpoint установлен; отдельный joint-equivalent audio checkpoint не заявляется |
| Полное FiLM обучение и оценка голоса | Следующий исследовательский этап; готовый train/inference pipeline не заявляется |

Не называйте loss на случайной сети, header inspection или импорт workflow доказательством клонирования голоса.

## Память и диск перед запуском

Первоначально запуск был отложен до освобождения 50 GB+ SSD. **30 октября 2026 пользователь разрешил загрузку необходимых моделей**: семь файлов Lite workflow теперь доступны в локальной ComfyUI, включая TOD-VAE и оба файла BigVGAN. Baseline в Desktop затем успешно продолжен из Sampler cache и сохранён с видео и звуком; Kaggle/T4 запуск ещё не проверен.

Проверьте свободное место для swap и временных файлов перед новым запуском. Локальная библиотека моделей и CUDA/Kaggle окружение — отдельные установки; файлы Mac автоматически в Kaggle не передаются. [Kaggle notebook для 2×T4 / 32 GB RAM](notebooks/15_kaggle_t4_lite_distill.ipynb) скачивает только нужные файлы либо использует Kaggle Input.

Начните с [пошагового notebook](notebooks/00_manual_experiment_comfy_mlx_film.ipynb): 33 ячейки, лёгкие примеры FiLM/attention/Flow Matching, проверка файлов, таблицы измерений. Run All не скачивает модели и не запускает генерацию. Для запуска из корня репозитория:

```bash
uv run --locked jupyter lab kandinsky-6/notebooks/00_manual_experiment_comfy_mlx_film.ipynb
```

## 1. Ваша установка

Команды ниже используют две переменные окружения. Задайте их для своей установки ComfyUI Desktop; значения на проверенной станции M3 Max показаны в комментарии:

```bash
export COMFY_ROOT="$HOME/ComfyUI-Installs/<профиль>/ComfyUI"   # станция: ~/ComfyUI-Installs/Vladislav/ComfyUI
export COMFY_SHARED_MODELS="$HOME/ComfyUI-Shared/models"        # общая библиотека моделей Desktop
```

Тогда Python ComfyUI — `$COMFY_ROOT/.venv/bin/python`, плагин — `$COMFY_ROOT/custom_nodes/kandinsky6`.

На момент осмотра: ComfyUI 0.42.5, kandinsky6 1.0.2, Python 3.13.12, Torch 2.12.1. Устанавливать исследовательские зависимости в это окружение не требуется. MLX будет иметь отдельное окружение. Фактический model root проверьте в Desktop: установка может использовать общую библиотеку или собственную папку models.

В расширении уже есть шаблоны Text to Video+Audio и Image to Video+Audio. Исходные версии используют Pro Distill, Qwen3.5 beautifier и SR. Общая кнопка **Download models** расширения загружает этот большой комплект, включая Pro; она не превращается в Lite downloader после смены модели в loader.

## 2. Сначала импортируйте smoke workflow

В вашу установленную копию расширения добавлены **Kandinsky 6 Lite - Smoke** и **Kandinsky 6 Lite - Baseline**. После обновления окна/перезапуска Desktop ищите их в Templates → kandinsky6; сохранённые копии также лежат в Workflows → K6 Research. Эти локальные файлы не являются новыми официальными upstream templates; обновление расширения может их заменить. Копии в этом репозитории позволяют восстановить их.

Для повторной установки после обновления расширения из папки kandinsky-6:

```bash
python3 scripts/install_comfy_templates.py --comfy-root "$COMFY_ROOT" --install
```

Без `--install` это dry run. Если существующий исследовательский шаблон отличается, скрипт остановится для сравнения; `--replace` разрешает его заменить. Официальные шаблоны не заменяются, веса не скачиваются.

В ComfyUI можно также открыть меню загрузки workflow или перетащить файл:

```text
workflows/k6_lite_distill_smoke.json
```

Это 12 узлов: Lite transformer → sampler → video/audio decode → CreateVideo → SaveVideo. Для него нужны только kandinsky6 и встроенные узлы ComfyUI. SR, beautifier и MagCache удалены, включая их связи. Положительные подписи задаются непосредственно в Text Encode.

Проверьте видимые параметры:

| Параметр | Smoke | Baseline |
|---|---|---|
| Transformer | Kandinsky-6.0-Lite-distill-5s-Diffusers | тот же |
| Width × height | 432 × 256 | 864 × 480 |
| RGB frames | 33 | 121 |
| FPS | 24 | 24 |
| Batch | 1 | 1 |
| Steps | 10 | 10 |
| CFG | 1 | 1 |
| Denoise | 1 | 1 |
| Audio scaling | 0.417 | 0.417 |
| Seed | 42, fixed | 42, fixed |

Smoke — только уменьшенная проверка работоспособности. Его качество и время нельзя приписывать штатному 5s режиму: изменены разрешение и длительность. Размеры кратны 16, длина удовлетворяет T=4k+1. Baseline находится в `workflows/k6_lite_distill_baseline.json`.

Для distill узел Kandinsky6Sampler автоматически выбирает PiFlow. Видимые selectors euler/simple не означают, что здесь запущен обычный Euler sampler; они относятся к non-distilled моделям. Сохраняйте точную версию плагина.

### Если Sampler уже выполнен, а VAE decode упал

Baseline 121 frames / 864×480 с обычным `VAEDecode` на MPS запросил attention buffer **75.16 GiB**. В новом графе node 9 — **VAEDecodeTiled**: tile_size=256, overlap=64, temporal_size=16, temporal_overlap=4. Проверьте этот тип и четыре значения в открытом workflow; копии в Templates и Workflows обновлены. [Отчёт проверки](reports/comfy-tiled-vae-validation.json).

Меняйте только decoder и его downstream-часть. Сохраните ID и входы Sampler (модель, seed, captions, steps, CFG, размеры/length/fps); **не перезапускайте backend и не очищайте cache**. Тогда при следующем Run ComfyUI сможет взять joint latent из cache, если тот ещё сохранён. Сам факт завершённого Sampler в History не гарантирует наличие tensor в RAM: cache может быть вытеснен. Идентичные upstream inputs и сохранённый cache важнее положения узлов на canvas.

Для надёжного восстановления между сессиями сохраняйте совместные video/audio latents отдельно. Kaggle notebook делает это в `joint-latents.safetensors`: video/audio/export можно повторять без Qwen и DiT. В текущем Desktop графе отдельного сохранения latents нет, поэтому после перезапуска сервера один JSON workflow не восстанавливает вычисленный tensor.

Tiled decoder отдельно проверен на MPS с реальными video VAE weights: нулевой latent `[1,16,5,32,32]` → 17 RGB кадров 256×256 без NaN/Inf. Затем полный Baseline 864×480 / 121 кадр успешно продолжен из cache: выполнены только nodes 9, 11, 12, подтверждены H.264, AAC 44100 Hz, 24 FPS и 121 декодируемый кадр. Продолжение заняло 563.51 s; quality/lip-sync и сравнение untiled/tiled отдельно не оценивались.

### Прогресс 3D VAE tiler

В текущей конфигурации Baseline: **150 3D-тайлов = 10 temporal × 3 height × 5 width**. Веса не загружаются 150 раз: один decoder последовательно обрабатывает участки latent volume. Overlap смешивается на границах. Это число VAE вызовов, не DiT NFE.

Добавлен узкий [patch штатного `VAE.decode_tiled_3d`](patches/comfyui-3d-vae-progress.patch): подсчёт тайлов по фактическому правилу `range(0, length-overlap, tile-overlap)` и передача `ProgressBar` в уже существующий tiler. [Native CPU checks](reports/comfy-3d-progress-validation.json) проверили totals 150/1/2/8, batch и неизменность результата identity decoder.

Patch уже установлен в локальный `comfy/sd.py`, но **backend не перезапускался**, чтобы сохранить cache. Прогресс станет доступен после обычного следующего перезапуска. Это не checkpointing тайлов: незавершённый VAE decode после остановки придётся выполнить заново. Для другой установки сначала проверьте применимость patch к версии ComfyUI, затем из её корня выполните `git apply --check <path-to-patch>` и `git apply <path-to-patch>`; обновление ComfyUI может потребовать перенести этот patch.

## 3. Проверьте обязательные модели

Пути ниже относительны к активному каталогу models. Уже скачанные стандартные файлы можно использовать повторно; не переносите их наугад между каталогами.

| Компонент | Файл |
|---|---|
| Lite transformer | diffusion_models/Kandinsky-6.0-Lite-distill-5s-Diffusers/transformer/diffusion_pytorch_model.safetensors |
| Text Qwen | text_encoders/qwen_2.5_vl_7b.safetensors |
| Text CLIP | text_encoders/clip_l.safetensors |
| Video VAE | vae/hunyuan_video_vae_bf16.safetensors |
| Audio VAE | audio_vae/v1-44.pth |
| Vocoder | audio_vae/bigvgan_vocoder/bigvgan_generator.pt |
| Vocoder config | audio_vae/bigvgan_vocoder/config.json |

Ссылки на модели сохранены в свойствах узлов и в [официальной инструкции](reference/kandinsky-6/comfyui/README.md). После импорта проверьте, что transformer выбран именно **Lite**. Для первого опыта сохраняйте weight_dtype=default и исходные BF16/FP16 веса; FP8/GGUF — отдельная абляция.

Независимая проверка без загрузки тензоров в память, из папки kandinsky-6:

```bash
python3 scripts/inspect_comfy_models.py \
  --models "$COMFY_SHARED_MODELS" \
  --models "$COMFY_ROOT/models"
```

Смотрите `reports/comfy-model-inventory.json`. Поле structurally_complete проверяет длину safetensors по offsets; это не проверка качества весов и не подтверждение их происхождения. Во время подготовки найдены CLIP и video VAE; перечень при повторном запуске может расшириться.

## 4. Первый запуск и перенос на полный baseline

1. Перезапустите ComfyUI после установки расширения.
2. Импортируйте smoke JSON, проверьте выбор всех loader-файлов.
3. Оставьте исходный prompt, seed=42 и нажмите Queue один раз.
4. Сохраните workflow, консольный log и выходной MP4.
5. Проверьте наличие видео, аудио, реальную длительность, отсутствие полностью чёрного кадра/тишины/NaN ошибки.
6. При успехе импортируйте baseline JSON и повторите при 864×480, 121 кадре.

Если возникают OOM, отсутствующие узлы, несовместимые operators или ошибки codec — сохраните точный текст, шаг и profile. Не заменяйте checkpoint и precision незаметно. Уменьшение геометрии создаёт другой опыт, который нужно назвать отдельно. На M3 Max CPU и GPU используют общий пул 48 GB; CPU offload не добавляет физическую RAM.

## 5. Замеры baseline

Создайте строки в `reports/templates/baseline_runs.csv`. Первый cold run включает загрузку моделей. Следующие warm runs измеряйте отдельно. Не сравнивайте время скачивания с временем generation.

| Замер | Как записывать |
|---|---|
| Версии | ComfyUI, plugin, Torch, Python, macOS; model filename и SHA-256 |
| Условия | profile, width/height, frames/fps, precision, steps, CFG, prompt, seed |
| Время | cold/warm отдельно; полный prompt time из console, sampler time отдельно, если оно доступно |
| Память | observed MPS/torch показатели, RSS процесса, общая memory pressure и swap |
| Выход | video/audio duration, sample rate, файл результата, любые ошибки |

Для warm time используйте, например, **seed 43, 44, 45**. Повтор того же seed с неизменным графом может вернуть cached outputs без реального sampler execution. Нулевое время такого cache hit не является ускорением модели. Проверку повторяемости одного seed делайте в отдельных свежих процессах, отмечая cold-loading overhead.

Для read-only наблюдения памяти можно запустить перед Queue:

```bash
python3 scripts/measure_comfy.py \
  --url http://127.0.0.1:8188 --seconds 900 \
  --output reports/baseline-memory.csv
```

URL замените на фактический адрес сервера Desktop; 8188 — только default скрипта. Скрипт не отправляет prompts. Он опрашивает `/system_stats`; максимум с шагом 1 секунду — **наблюдаемый**, не гарантированный истинный пик. API может не предоставлять все MPS поля. `torch allocated`, driver memory, process RSS, системная RAM и swap — разные величины; не складывайте их как независимые пулы.

На Mac дополнительно сохраните значения Memory Pressure и Swap Used из Activity Monitor. Инструментальная ошибка измерения входит в отчёт.

## 6. Повторное использование checkpoint для MLX

ComfyUI хранит единый мультимодальный checkpoint. Его параметры имеют отдельно именованные video/audio и cross-modal части. Их можно прочитать вне ComfyUI. Это не означает, что выбрасывание cross-modal слоёв сохраняет полный генератор.

**Основной путь при ограниченном SSD — читать уже скачанный оригинальный Lite checkpoint напрямую**, без второй копии и без сетевой загрузки. В следующем разделе используйте `--weights` с полным путём. Проверка pinned оригинала включает размер и SHA-256; этот read-only проход читает файл полностью и занимает время.

Дополнительное извлечение audio tensor bytes нужно только при необходимости отдельного subset (оно занимает ещё около 1.1 GiB):

```bash
python3 scripts/extract_audio_local.py \
  --checkpoint /ПОЛНЫЙ/ПУТЬ/К/diffusion_pytorch_model.safetensors \
  --output artifacts/weights/k6-lite-distill-audio.safetensors
```

Замените единственный placeholder фактическим путём из inventory. Скрипт не изменяет источник, не квантирует и не переобучает. Он отклоняет неполные и неподдерживаемые quantized файлы. Для опубликованной Lite Distill версии из нашего manifest заголовок содержит 2399 тензоров; audio subset — 816 тензоров и 574 096 016 параметров. Другая revision может отличаться, поэтому сравнивайте shapes/конфигурацию и provenance.

Рядом создаётся completion stamp с SHA-256 полученного subset. Прерванный sparse download без stamp не разрешается использовать как pretrained модель. Локальный subset hash не равен подтверждению SHA-256 всего исходного Hub checkpoint.

## 7. Проверка MLX-порта до обучения

Первое обязательное сравнение — маленький случайный fixture с ненулевыми gates, проверяющий математику. Он **не** оценивает pretrained качество. Команда использует существующий Python ComfyUI только для CPU reference и ничего в него не устанавливает:

```bash
"$COMFY_ROOT/.venv/bin/python" \
  scripts/torch_parity_reference.py
```

Установку MLX также отложили; попытка установки была остановлена после network timeout, numerical parity пока не выполнен. После освобождения места для MLX создайте отдельное Python 3.12 окружение. На данной машине используется macOS 27; pinned mlx-metal 0.32.4 требует подходящий macOS wheel.

```bash
uv venv --python 3.12 .venv-mlx
UV_HTTP_TIMEOUT=120 uv pip install --python .venv-mlx/bin/python \
  mlx==0.32.4 numpy safetensors
.venv-mlx/bin/python scripts/mlx_parity.py
```

Продолжайте только при `passed=true`. Затем, после полной загрузки и проверки оригинальных Lite весов:

```bash
"$COMFY_ROOT/.venv/bin/python" \
  scripts/torch_parity_reference.py --pretrained \
  --weights /ПОЛНЫЙ/ПУТЬ/К/LITE/diffusion_pytorch_model.safetensors
.venv-mlx/bin/python scripts/mlx_parity.py --pretrained
```

Скрипты записывают max absolute error, relative L2 error, ошибку zero-FiLM identity, времена cold и трёх warm forward, активную и пиковую MLX память. Начальный gate: FP32 max_abs<1e-3, relative_L2<1e-4 и zero-FiLM error=0. Порог установлен для этого короткого forward и не является универсальной гарантией совпадения генерации. При непрохождении сначала исследуйте LayerNorm/RMSNorm epsilon, порядок reshape, RoPE, dtype и двойную normalization audio output head.

Наш MLX prototype реализует **native audio-forward**, без VAE/Qwen/vocoder и совместного video path. Distill head имеет 400 выходных каналов = 10 grid points × 40 audio channels. Официальный PiFlow sampler требует оба потока; проверка audio head не является переносом всего sampler и не доказывает waveform equivalence. Время короткого 12-token forward нельзя называть latency 5s генерации.

## 8. Постановка эксперимента FiLM

**H1:** при замороженном pretrained audio backbone один speaker-conditioned FiLM повышает сходство с reference voice на отложенных текстах при приемлемой регрессии разборчивости. Это наша абляция. [Статья voice cloning, §2.4](https://arxiv.org/html/2608.15690v1#S2.SS4) дообучает также существующий audio backbone; adapter-only режим не является её точным воспроизведением. Обещания «10–20 записей, 5–15 минут» не подтверждены нашим экспериментом.

Контролируемые варианты:

| Вариант | Reference latents | FiLM | Что учится |
|---|---|---|---|
| B0 | нет | нет | ничего; исходный baseline |
| B1 | нет | zero-init | ничего; identity check при одинаковом контексте |
| B2 | есть | zero-init | ничего; отдельный эффект добавления reference tokens |
| A1 | есть | trainable | только один FiLM |
| A2, позднее | есть | trainable | FiLM + audio LoRA/backbone, отдельный бюджет |

FiLM: `h_target * (1 + gamma) + beta`, где `(gamma,beta)=W(e)`, W и bias инициализированы нулями. Reference tokens не модулируются и не обновляются diffusion sampler. Один слой используется в одном заранее описанном месте; 32 отдельных слоя уже являются другим экспериментом.

Условие identity относится к **одинаковым входам и attention context**. Нулевой FiLM не компенсирует изменение self-attention после добавления reference prefix. Поэтому B1 и B2 нужны раздельно.

До обучения необходимо реализовать и проверить preprocessing/cache настоящих MMAudio latents, Qwen/CLIP embeddings и speaker embeddings, а также sampler с reference prefix и masked target loss. Qwen3-TTS использует embedding 1024, Resemblyzer обычно 256; их нельзя объявлять эквивалентными по одному названию. Dimension и encoder revision сохраняются в config.

Для первого корректного FM adapter training следует использовать non-distilled Lite checkpoint с 40-channel velocity output и соответствующий ему codec scale. **Не применяйте обычный velocity-MSE к 400-channel PiFlow head**. Продолжение distill обучения требует отдельно обоснованной objective/teacher policy; это пока не подготовленный training маршрут. Смена checkpoint — явное изменение варианта опыта, а не скрытый fallback.

## 9. Данные VCTK и защита от утечек

Подготовлен небольшой pilot из p225, p226, p227, p228: всего 120 клипов. Это не 674-pair benchmark статьи. Для воспроизведения:

```bash
python3 scripts/prepare_vctk.py --download
```

Скрипт читает ZIP directory и выбранные byte ranges, а не весь 6.28-GB архив. Записываются dataset revision, CRC и SHA-256 аудио, исходные utterance IDs и тексты. Сначала отбираются reference/train/validation/test, затем одинаковые нормализованные тексты глобально объединяются в один split с приоритетом holdout. Точные итоговые размеры читайте в manifest: после удаления утечек они могут отличаться от первоначальных 16/64/20/20.

Reference, target и test должны быть разными utterances. Нельзя нарезать одну запись на train/test и считать это generalization. Правила trim, normalization, resampling, reference length и clipping фильтров фиксируются до эксперимента. MMAudio K6 — 44.1 kHz/40 channels, KVAE-Audio — 48 kHz/64 channels; подмена codec требует другой модели/адаптации.

## 10. План обучения и необходимые измерения

Этот раздел задаёт протокол следующей реализации; команды несуществующего train pipeline здесь намеренно не выдаются за готовые.

1. Зафиксировать backbone hash и все cache hashes; сохранить optimizer parameter names. Убедиться, что backbone не входит в optimizer.
2. Проверить B0/B1 на одном noise, time, prompt и context. Проверить gradients W/bias: конечны и ненулевые, reference сохраняется.
3. Выполнить короткий overfit test на 1–2 train utterances, чтобы выявить wiring/target/sign ошибки. Это проверка реализации, а не результат voice cloning.
4. Для A1 использовать три seed и заранее выбранный budget. Снимать train/validation loss на фиксированных held-out noise/time, gradient norm, parameter delta, ms/step, peak memory.
5. Выбрать checkpoint только по validation. После этого один раз оценить test и no-reference regression set. При продолжении подбора test перестаёт быть test.
6. Сохранить только adapter и config, затем перезапустить процесс и проверить reload equality. Нужна отдельная поддержка адаптера в sampler/Comfy custom node; обычный LoRA Loader не обязан понимать FiLM.

Flow convention для non-distilled опыта: data при t=0, noise при t=1, `x_t=(1-t)z+t*eps`, target `eps-z`; sampling идёт к t=0. Loss применяется к target tokens, не к фиксированному reference. Independent text/reference dropout 0.1 создаёт четыре режима: около .81/.09/.09/.01, а не четыре равновероятные группы. Обучение этих режимов должно предшествовать split CFG; добавление трёх forward на inference само по себе не обучает voice conditioning.

## 11. Как оценить именно голос

| Метрика | Требование |
|---|---|
| SECS | Один и тот же frozen verifier и preprocessing во всех вариантах; отдельно vs reference и vs speaker centroid |
| WER/CER | Одинаковый ASR и text normalization; сохранять числители ошибок и число reference слов/символов |
| Естественность | Слепое прослушивание; UTMOS как дополнительная model-based оценка, не истина |
| FAD | Dataset-level distributional metric с указанным embedding backbone; не на одном клипе и не на четырёх tiny references |
| CLAP | Соответствие тексту/аудиособытиям; не идентичность speaker |
| No-reference regression | B0 vs обученный adapter при отключённом reference; тот же test protocol |
| Audio-only/AV difference | Одинаковые prompts, seeds, duration, NFE и codec; отдельно visual/sync metrics для полного AV |

При малом pilot отчёт должен показывать per-speaker значения, paired differences и разброс по seed. Bootstrap желательно группировать по speaker; строки одного speaker не являются независимыми людьми. Для четырёх speaker CI будет неустойчивым — это pilot, не доказательство обобщения. Статистический тест выбирается до просмотра результата; p-value из статьи нельзя переносить на новый запуск.

## 12. Критерии вывода и перехода на Vast.AI

- Parity не пройден → вывод о проблеме порта, обучение не интерпретируем.
- Train loss падает, held-out SECS/WER не улучшаются → адаптер мог переобучиться; клонирование не подтверждено.
- SECS растёт, WER/естественность портятся → есть измеримый tradeoff, а не безусловный успех.
- Улучшение стабильно на test и нескольких seed при no-reference контроле → основания для следующего более крупного опыта.
- OOM/неподдерживаемый operator/неприемлемый step time → сохранить точный profile и ошибку; переносить этот же experiment на CUDA, не менять одновременно dataset/objective/checkpoint.

На Vast сначала нужен smoke run и оценка peak VRAM **и host RAM**. RTX 4090 24 GB не хранит 29B BF16 модель целиком: только веса требуют около 58 GB в десятичных единицах. Offload, quantization и меньшая модель — отдельные условия. Покупать GPU лишь по числу trainable FiLM параметров недостаточно: backward проходит через frozen downstream backbone.

В итоговом отчёте перечислите, какие этапы реально выполнены, какие пропущены, и приложите config, логи, таблицы и файлы выходов. Отрицательный результат и найденный bottleneck — допустимые результаты исследования.
