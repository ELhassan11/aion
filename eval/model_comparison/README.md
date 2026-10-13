# مقارنة النماذج (P0-T06)

## الفكرة

بدل ما نختار النموذج بالتخمين، بنشغّل كل المرشحين على **نفس الاختبارات العربية**
(`test_cases.json` — 8 حالات مبنية على بياناتنا التجريبية) ونقارن الدرجات.

## التشغيل

```bash
# نموذج API مدفوع (محتاج OPENAI_API_KEY في .env)
OPENAI_API_KEY=sk-... python eval/model_comparison/compare.py --models gpt-4o-mini

# نموذج مفتوح محليًا (محتاج Ollama شغال)
python eval/model_comparison/compare.py --models ollama:qwen2.5:7b

# مقارنة الاتنين مع بعض
OPENAI_API_KEY=sk-... python eval/model_comparison/compare.py \
  --models gpt-4o-mini,ollama:qwen2.5:7b
```

### OpenRouter (موديلات مجانية 💰=0)

1. اعمل حساب مجاني على openrouter.ai وخد API key.
2. الموديلات المجانية اسمها بينتهي بـ `:free` (القايمة بتتغير — شوف openrouter.ai/models).
3. شغل المقارنة:

```bash
OPENROUTER_API_KEY=sk-or-... python eval/model_comparison/compare.py \
  --base-url https://openrouter.ai/api/v1 \
  --models "google/gemini-2.0-flash-exp:free,openai/gpt-oss-120b:free"
```

المجاني محدود (حوالي 50 طلب/يوم) — والـ 8 اختبارات بتوعنا بيستهلكوا 8 طلبات بس لكل موديل، يعني في الأمان.

## التقييم

الدرجة = نسبة الكلمات المفتاحية المتوقعة الموجودة في الإجابة.
مقياس بسيط وشفاف — كافٍ للمقارنة الأولية (التقييم العميق بأسلوب Ragas يأتي في P5).

## النتائج

تُحفظ كل جولة في `results/run-<timestamp>.json`، وتُلخص في `docs/model-selection.md`.
