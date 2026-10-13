# عقود الواجهات — AION (v1)

تحدد هذه الوثيقة **أشكال البيانات** التي تتبادلها طبقات المشروع، حتى يبني كل جزء
مستقلًا ثم تتركب الأجزاء دون مفاجآت. التنفيذ الفعلي للعقود: `backend/models/`.

**قاعدة الإصدارات:** العقود نسخة `v1` (مسودة قابلة للتطوير). أي تغيير يكسر
التوافق (حذف حقل، تغيير نوع) يتطلب إصدارًا جديدًا واتفاق الفريق — لا تغييرات
صامتة.

---

## 1. عقود بين الطبقات

| من | إلى | العقد | الملف |
|---|---|---|---|
| الاستيعاب | التخزين / RAG | `DocumentChunk` — وحدة النص غير القابلة للتعديل بعد الكتابة | `models/document.py` |
| RAG | الوكلاء | `RetrievalResponse` — قائمة `RetrievedChunk` مرتبة مع الدرجات | `models/retrieval.py` |
| الوكلاء | الـ Backend | `AgentResponse` — الإجابة + الاستشهادات + الثقة + خطوات التفكير + هل تحتاج موافقة | `models/agent.py` |
| الإنسان | الوكلاء | `ApprovalRequest` / `ApprovalDecision` — طلبات الموافقة وقراراتها | `models/agent.py` |
| وحدة العملاء | اللوحة | `CustomerProfile` (360) / `Ticket` / `NextBestAction` | `models/customer.py` |
| التقييم | التقارير | `EvalCase` / `EvalResult` — الحالة والمقاييس لكل منهجية | `models/eval.py` |
| أي طبقة | أي طبقة | `Citation` — ربط أي جملة مولدة بمصدرها | `models/common.py` |

## 2. واجهات REST (v1)

البادئة: `/api/v1`

| الطريقة | المسار | الطلب | الرد |
|---|---|---|---|
| POST | `/chat` | `AgentQuery` | `AgentResponse` |
| POST | `/ingest` | `IngestRequest` | `IngestResponse` |
| GET | `/documents` | `PaginationParams` | `list[DocumentChunk]` (ملخص) |
| GET | `/documents/{id}/chunks` | — | `list[DocumentChunk]` |
| GET | `/customers` | `PaginationParams` | `list[CustomerProfile]` |
| GET | `/customers/{id}` | — | `CustomerProfile` + `list[Ticket]` |
| GET/POST | `/tickets` | `Ticket` (للإنشاء) | `Ticket` |
| GET | `/approvals` | `?status=pending` | `list[ApprovalRequest]` |
| POST | `/approvals/{id}/decision` | `ApprovalDecision` | `ApprovalRequest` (محدث) |
| GET | `/eval/runs` | — | `list[EvalResult]` (ملخص) |
| GET | `/health` | — | `{"status":"ok"}` ✅ يعمل |
| GET | `/ready` | — | حالة الاعتماديات ✅ يعمل |

الأخطاء: كل الأخطاء ترجع `ErrorResponse` — `{code, message, details?}` مع كود HTTP المناسب.

## 3. ملاحظات تصميمية

1. **الاستشهادات إلزامية:** أي `AgentResponse` بلا `citations` يُعتبر ناقصًا — الوكيل الذي لا يستشهد بمصادره مرفوض في المراجعة.
2. **`needs_approval`:** إذا كان `true`، لا يُنفذ أي إجراء حتى يأتي `ApprovalDecision` بالموافقة.
3. **`DocumentChunk` غير قابلة للتعديل:** التصحيح يكون بإصدار chunk جديدة، لا بتعديل القديمة (قابلية التتبع).
4. **الثقة `confidence`:** رقم 0–1 للعرض فقط — القرارات الحرجة تعتمد على `needs_approval` لا على الثقة.
