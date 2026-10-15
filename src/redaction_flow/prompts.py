"""Prompts for original-page extraction, OCR-span mapping and crop triage."""
ORIGINAL_EXTRACTION_SYSTEM = """You identify sensitive information to redact from an original supplier compliance report page.
You receive only the original page image. Return JSON only:
{"redacted_information":[{"category":"...","field_label":"...","original_value":"...","confidence":0.0}]}
Protect supplier/customer/applicant identity, protected addresses, people/signatures, submitted sample names,
product/style/model/part identifiers, report/access identifiers, contact details, and proprietary references.
Do not redact laboratory branding, technical methods, chemicals, standards, units, results, procedures, or
generic technical content. A generic test-report number is not sensitive by itself; redact an identifier
only when page context marks it as a private client reference, order/reference number, PIN, QR/access code,
or other confidential submission identifier. Do not redact routine testing-laboratory mastheads, logos,
footers, public lab addresses, or standard contact blocks. Preserve values exactly as visibly written.
Read Chinese, Japanese, German, and mixed-language identity/address blocks. Include complete compound
address/contact blocks when they are tied to a protected party. Return every separate visible occurrence.
Do not invent hidden values."""

MAPPING_SYSTEM = """Map known redaction values to OCR spans. Return JSON only:
{\"mappings\":[{\"value\":\"...\",\"category\":\"...\",\"span_indices\":[0],\"confidence\":0.0,\"reason\":\"...\"}],\"unmatched\":[{\"value\":\"...\",\"category\":\"...\",\"reason\":\"...\"}]}.
Use only supplied span indices. A value may span adjacent OCR spans. Map each physical occurrence separately.
Do not invent indices or map nearby technical text. If no defensible mapping exists, put the value in unmatched."""

CROP_CLASSIFICATION_SYSTEM = """Classify the visible crop as exactly one category: QR_BARCODE, MATERIAL_SAMPLE, or NOISE.
MATERIAL_SAMPLE means a physical product, material, specimen, component, or label attached to one.
QR_BARCODE means a machine-readable code. NOISE means a diagram, chart, generic logo, text-only passage,
or irrelevant region. Decide from the crop only. Return JSON only:
{\"category\":\"QR_BARCODE|MATERIAL_SAMPLE|NOISE\",\"confidence\":0.0,\"reason\":\"...\"}."""

MATERIAL_TEXT_SYSTEM = """This is a material-sample photo crop. Locate visible text and decide each region using policy material-identifiers-v1.
REDACT material/sample names or identifiers, product/part codes, batch/lot/serial numbers, specimen/test/report
references associated with the sample, and company/person/address/contact identifying the sample/source.
PRESERVE ruler/scale graduations and numbers, units, dimensions, generic measurement labels/headings, and
non-identifying technical information. A number is not sensitive merely because it is a number; use context.
Unreadable or ambiguous text whose role is uncertain must be REVIEW, not guessed safe. For mixed text use tight
separate boxes. Return JSON only:
{\"text_regions\":[{\"text\":\"\",\"x0\":0,\"y0\":0,\"x1\":0,\"y1\":0,\"decision\":\"redact|preserve|review\",\"category\":\"\",\"reason\":\"\"}]}.
Coordinates are original crop pixels. Include all visible text. Do not redact the material itself."""
