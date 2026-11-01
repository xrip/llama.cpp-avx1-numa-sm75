# Патч экспериментов SM75

База: публичный `master` репозитория `xrip/llama.cpp-avx1-numa-sm75`, коммит `af084752abb72652d51785602ef9536f631fec9d`.

Патч не переключает ветку, не создаёт коммит и ничего не отправляет на GitHub. Это обычный unified diff для `git apply`, не файл для `git am`.

## Применение

Из каталога своего репозитория укажите путь к распакованному архиву. Сначала выполните проверку. Если она завершилась ошибкой, не применяйте патч частично и не используйте `--reject`.

```sh
git rev-parse HEAD
git apply --check /path/to/sm75-prefill-patch/sm75-ninfer-prefill-af084752.patch && \
    git apply /path/to/sm75-prefill-patch/sm75-ninfer-prefill-af084752.patch
```

В архиве есть дополнительная проверка исходных файлов и контекста патча:

```sh
python /path/to/sm75-prefill-patch/verify-patch.py --repo .
```

Она ничего не изменяет и должна выполняться до применения патча. Локальные файлы могут отличаться от публичной базы: скрипт сообщает об этом отдельно, затем запускает `git apply --check`.

## Содержимое

`sm75-ninfer-prefill-af084752.patch` — изменения backend и документация `docs/sm75-ninfer-prefill-experiments.md`.

`validation/` — отдельные host/CUDA-тесты. Они не устанавливаются патчем в основное дерево tests и не добавляются в CI.

`VALIDATION.md` и `validation-results.json` — фактически выполненные проверки и ограничения.

`base-manifest.json` — SHA исходных файлов публичной базы; `SHA256SUMS` — контрольные суммы файлов комплекта.

## Проверка арифметики и политики без GPU

После применения патча:

```sh
cmake -S /path/to/sm75-prefill-patch/validation -B build-sm75-check \
    -DGGML_SOURCE_DIR="$PWD" -DCMAKE_BUILD_TYPE=Release
cmake --build build-sm75-check --parallel 2
ctest --test-dir build-sm75-check --output-on-failure
```

## Проверка CUDA-helper на SM75

```sh
cmake -S /path/to/sm75-prefill-patch/validation -B build-sm75-check-cuda \
    -DGGML_SOURCE_DIR="$PWD" -DSM75_TEST_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-sm75-check-cuda --parallel 2
ctest --test-dir build-sm75-check-cuda --output-on-failure
```

Тест сам выбирает доступную SM75. Можно ограничить устройства через `CUDA_VISIBLE_DEVICES`. При отсутствии подходящей GPU он возвращает 77 (пропуск), а не успешную проверку. Проверяются все 65 536 пар INT8 с 27 масштабами FP16, включая нули, субнормальные значения, максимальные конечные значения, бесконечности и NaN.

Этот небольшой CUDA-тест не заменяет полную сборку backend и тестирование модели, CUDA Graphs, tensor split и длинного контекста.

## Что включать

По умолчанию сохраняются прежняя политика `auto`, стандартная геометрия MMQ и старая Q8-распаковка. Само применение патча не включает предполагаемые ускорения.

Для проверки cuBLAS перед запуском прежней полной команды:

```sh
export GGML_CUDA_SM75_PREFILL_MODE=cublas
export GGML_CUDA_SM75_PREFILL_MIN_BATCH=1024
```

Отдельно для выбора более узкого MMQ-тайла:

```sh
export GGML_CUDA_SM75_PREFILL_MODE=mmq
export GGML_CUDA_SM75_MMQ_MAX_J=64
```

Q8-helper включается при конфигурации сборки через `-DGGML_CUDA_SM75_Q8_0_FAST_HALF=ON`. Начните со сборки с OFF. Полное описание ограничений и измерений находится в добавляемом документе.

Важное уточнение к первоначальному разбору: в исходной ветке уже был cuBLAS-путь для нескольких FFN-форм на SM75. Здесь он сохранён в auto и дополнен экспериментальными режимами, а не добавлен заново.
