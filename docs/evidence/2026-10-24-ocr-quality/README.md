# OCR quality on fictional documents - 24 October 2026

Status: **executed. Threshold not fully met:** all 16 clean documents pass; 34 of the 36 difficult scans pass, 2 fail.

## Threshold

The article fixes no OCR threshold, so the laboratory proposed one and wrote it in the fixture before the run. The business owner has not agreed it yet.

- Clean typed documents (150 dpi or better, no degradation): **character error rate of at most 2%** per document.
- Difficult scans: **character error rate of at most 10%** per document.
- The rate is the edit distance between the typed ground truth and the OCR output, divided by the ground-truth length, after Unicode NFC normalisation and whitespace collapsing. Case, accents and punctuation count.

## Method

- Four fictional typed documents in [`tests/fixtures/measurements/ocr_documents.json`](../../../tests/fixtures/measurements/ocr_documents.json): an English witness statement (Arial 11 pt), a French statement (Times New Roman 12 pt), a Dutch statement (Calibri 11 pt) and a trilingual evidence inventory laid out as a table (Courier New 11 pt).
- [`scripts/measure_ocr_quality.py`](../../../scripts/measure_ocr_quality.py) `generate` renders each document under 13 scan conditions with a fixed seed: 52 files in [`inputs/`](inputs/). Four conditions are clean (300, 200 and 150 dpi PNG, and a 300 dpi image-only PDF); nine are difficult (100 dpi, 3 degree skew, blur, 3% salt-and-pepper noise, JPEG quality 15, low contrast, uneven lighting, a fax-like binarised page, and a noisy, skewed 200 dpi image-only PDF). [`ocr-manifest.json`](ocr-manifest.json) records seeds, sizes, file digests, the generator versions and the font digests.
- `run` sends the 52 files to the deployed `lab-app` pod and extracts each one with the **service function** `lab.documents.extract_document`: pypdf for PDFs, PDFium rendering at scale 2.0 when a page has no text layer, then Tesseract 5.5.0. The service calls Tesseract without a language, which means **English data only**, although the image also contains French and Dutch data.
- A second pass, labelled diagnostic, repeats the same function with Tesseract set to `eng+fra+nld`. It is not the service configuration.

## Results

| Condition | Category | Documents within threshold | Median CER | Worst CER | Diagnostic `eng+fra+nld` median CER |
| --- | --- | --- | --- | --- | --- |
| 300 dpi PNG | clean | 4/4 | 0.63% | 1.81% | 0.48% |
| 200 dpi PNG | clean | 4/4 | 0.63% | 1.99% | 0.33% |
| 150 dpi PNG | clean | 4/4 | 0.93% | 1.99% | 0.33% |
| 300 dpi image-only PDF | clean | 4/4 | 0.63% | 1.81% | 0.24% |
| 100 dpi PNG | difficult | 3/4 | 1.60% | **14.13%** | 0.93% |
| 3 degree skew | difficult | 4/4 | 1.37% | 2.17% | 0.69% |
| Blur | difficult | 4/4 | 0.63% | 1.81% | 0.24% |
| Salt-and-pepper noise 3% | difficult | 3/4 | 8.87% | **18.75%** | 7.03% |
| JPEG quality 15 | difficult | 4/4 | 1.07% | 1.81% | 0.24% |
| Low contrast | difficult | 4/4 | 0.77% | 1.99% | 0.48% |
| Uneven lighting | difficult | 4/4 | 1.23% | 2.17% | 0.48% |
| Fax-like | difficult | 4/4 | 4.98% | 7.44% | 2.57% |
| Noisy, skewed 200 dpi PDF | difficult | 4/4 | 1.95% | 5.06% | 0.68% |

| Category | Within threshold | Median CER | 95th percentile | Diagnostic: within threshold, median |
| --- | --- | --- | --- | --- |
| Clean | 16/16 | 0.66% | 1.99% | 16/16, 0.33% |
| Difficult | 34/36 | 1.80% | 14.13% | 34/36, 0.66% |

The two failures are the French statement at 100 dpi (14.13%: Tesseract also read one line out of order) and the trilingual inventory with salt-and-pepper noise (18.75%). Extraction took 0.4 to 17.5 s per file in the pod, while M1 was running in parallel; OCR latency is measured separately in M3.

## Reading the result

- **Accents are the main error on clean pages.** With English data only, Tesseract turns `à` into `a`, `É` into `E`, `â` and `ô` into `a` and `o`. French documents therefore come close to the 2% limit even on clean 300 dpi pages (1.81%). The English statement loses points on `I` read as `|`. These errors change names, places and words in a statement; a reviewer must still compare the extraction with the original.
- **Setting the languages helps.** The diagnostic `eng+fra+nld` pass halves the median error on clean pages and keeps every clean page under 0.7%. It does not fix the two failures, which come from the image itself. Adding the language list to the service is a separate, tested change.
- **Effective resolution of PDFs:** the service renders image-only PDF pages at scale 2.0, about 144 dpi, whatever the scan resolution. The 300 dpi PDF still passed, but a lower-resolution scan inside a PDF loses detail twice.
- These documents are synthetic renderings of typed text. Handwriting, stamps, photographs of paper, multi-column layouts and real scanner artefacts remain unmeasured.

## Files

| File | Content |
| --- | --- |
| `inputs/` | The 52 generated fictional scans |
| `ocr-manifest.json` | Conditions, seeds, digests, generator versions and font digests |
| `runtime-context.json` | Cluster, images and source hashes of the running pod |
| `ocr-raw.json` | Extracted text and time per file, for the service and the diagnostic pass |
| `ocr-results.json` | Error rates per file, condition and category, the threshold verdicts and the diagnostic summary |
| `SHA256SUMS` | Digests of every file in this folder |
