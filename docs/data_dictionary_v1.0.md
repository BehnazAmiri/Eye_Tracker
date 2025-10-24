# واژه‌نامهٔ داده (Data Dictionary) — نسخهٔ 1.0

## 1) `questions.json`
| ستون | نوع | توضیح |
|---|---|---|
| `id` | int | شناسهٔ یکتاى سؤال (۱..۱۵) |
| `question` | str | متن سؤال |
| `options[]` | list | آرایهٔ گزینه‌ها؛ هر گزینه شامل: |
| `options[].id` | str | الگوی `qid-index[-C]`؛ وجود `-C` یعنی پاسخِ درست در **بانک سؤال** |
| `options[].text` | str | متن گزینه |

## 2) `question_exams/Participant_X.json`
| ستون | نوع | توضیح |
|---|---|---|
| `Part1[]/Part2[]` | list | فهرست سؤال‌های همان بخش برای این شرکت‌کننده |
| `question_id` | int | ریفرنس به `questions.json` |
| `options[]` | list | ۴ گزینهٔ انتخاب‌شده برای نمایش **به همان ترتیب نمایش روی صفحه** (اندیس ۰..۳ → بالا-چپ، بالا-راست، پایین-چپ، پایین-راست) |
| `has_correct_answer` | bool | اگر `true`، دقیقاً یک گزینه‌ی `-C` در بین این ۴ مورد وجود دارد؛ اگر `false`، هیچ‌کدام درست نیستند |

## 3) `outputs/Participant_X/answers.json`
| ستون | نوع | توضیح |
|---|---|---|
| `section` | str | `Part 1` یا `Part 2` |
| `question_number` | int | شمارهٔ نمایشی ۱..۱۵ |
| `question_id` | int | شناسهٔ سؤال |
| `chosen_option` | str \| null | شناسهٔ گزینهٔ انتخاب‌شده؛ در تایم‌اوت `null` |
| `time_spent` | float | مدت پاسخ‌دهی برحسب ثانیه |

## 4) `outputs/Participant_X/Qk.csv` — لاگ خام آی‌ترکر برای سؤال k
**ستون‌های کلیدی (نام‌ها ممکن است دقیقاً به حروف داده‌های دستگاه بستگی داشته باشد):**

- `FPOGX`, `FPOGY` : مختصات نقطهٔ نگاه «بهترین/ترکیبی» (۰..۱)  
- `BPOGX`, `BPOGY` : مختصات جایگزین (Best POG) — استفادهٔ اختیاری
- `FPOGV` : اعتبار نگاه (۱ = معتبر)  
- `LPOGX`, `LPOGY` / `RPOGX`, `RPOGY` : مختصات نگاه چشم چپ/راست (اختیاری)  
- `LPD`, `RPD` : قطر مردمک چپ/راست (واحد طبق مستندات سازنده)  
- `BKID`, `BKDUR`, `BKPMIN` : اطلاعات پلک‌زدن (برای ساخت پرچم Blink)  
- سایر ستون‌ها مانند `FPOGD`, `FPOGS`, `LPS`, `RPS` ... در صورت نیاز در QC بررسی می‌شوند.

## 5) متادیتا/ویژگی‌های مشتق‌شده (در مرحلهٔ پردازش)
- `area_id` (int): ۰=خارج از AOIها، ۱=question، ۲=answer_tl، ۳=answer_tr، ۴=answer_bl، ۵=answer_br، ۶=right_timer، ۷=right_submit
- `phase` (int): ۱=خواندن سؤال، ۲=بررسی گزینه‌ها
- `is_correct_area` (0/1): فقط وقتی `has_correct_answer=true` و AOI گزینهٔ صحیح باشد → ۱
- `ΔFPOGX`, `ΔFPOGY`, `ΔBPOGX`, `ΔBPOGY`, `ΔLPD`, `ΔRPD` : تفاضل با ردیف قبلی
- `blink` (0/1): پرچم پلک‌زدن
- برچسب‌ها: `label_effort` (0/1)، `label_random` (0/1)