#!/usr/bin/env bash
# Статус индексации imgsim. Запуск откуда угодно.
DB=/Users/user/tmp/dinov2-image-search
python3 - "$DB" <<'PY'
import sys
sys.path.insert(0, sys.argv[1])
from imgsim.store import ImageStore
print("Записей в индексе:", ImageStore("./image_db").rows_count())
PY
if pgrep -f "imgsim index" >/dev/null; then
    echo "Статус: ИДЕТ индексация (замок image_db/.indexing)."
    tail -c 200 /tmp/index_log.txt | tr '\r' '\n' | tail -1
else
    echo "Статус: ГОТОВО (замок снят)."
fi
