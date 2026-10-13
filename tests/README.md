# Tests

اتفاقية الاختبارات:

- `tests/unit/<module>/` — اختبارات الوحدة لكل مجلد (مثلًا `tests/unit/rag/`).
- `tests/e2e/` — اختبارات التكامل الشاملة (تُبنى في المراحل الأخيرة).
- التشغيل: `pytest tests/ -q` أو `make test`.
