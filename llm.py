import openai
import json
from dataclasses import dataclass

from rich.console import Console

from config import settings


console = Console()


if not settings.bosch_gateway_token:
    console.print("Bosch Gateway token is missing", style="red")
    raise ValueError("settings.bosch_gateway_token must not be empty")


client = openai.OpenAI(
    base_url=settings.bosch_gateway_url + "/v1",
    api_key=settings.bosch_gateway_token
)


@dataclass
class RedactionTarget:
    """A sensitive text span identified for redaction."""

    text: str
    category: str
    confidence: float
    reason: str
    page_number: int
    span_index: int


def test_gateway_connection() -> bool:
    """Check whether the Bosch AI Gateway accepts a simple chat request."""
    try:
        response = client.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": "Reply with the single word: OK"}],
            max_tokens=10
        )
        content = response.choices[0].message.content or ""
        if "OK" in content:
            console.print("Bosch Gateway connection OK", style="green")
            return True
        raise ValueError(f"Unexpected gateway response: {content}")
    except Exception as error:
        console.print(f"Bosch Gateway connection FAILED: {error}", style="red")
        return False


def identify_sensitive_spans(
    page_texts: list[dict],
    document_languages: list[str]
) -> list[RedactionTarget]:
    """Identify sensitive spans across document pages using the Bosch Gateway."""
    targets: list[RedactionTarget] = []
    system_prompt = """You are a compliance document redaction specialist.
Your job is to identify sensitive information in supplier compliance
documents such as RoHS, REACH, and PFAS reports.

Sensitive categories:
- supplier_name: company names, supplier identifiers, brand names
- internal_ref: internal codes, document numbers, part numbers,
  SID codes, BOS codes, internal reference numbers
- material_data: proprietary material compositions, coating formulas,
  trade secret material names, proprietary CAS numbers
- contact: email addresses, phone numbers, employee full names

Return ONLY a valid JSON array. No explanation, no markdown,
no code blocks. Just the raw JSON array.

Each item must have exactly these fields:
  span_index: integer
  category: string
  confidence: float between 0.0 and 1.0
  reason: string (one sentence)

If nothing is sensitive return empty array: []"""

    for page in page_texts:
        spans = page.get("spans", [])
        numbered_span_string = "\n".join(
            f"{span.get('index', index)}. {span.get('text', '')}"
            for index, span in enumerate(spans)
        )
        try:
            response = client.chat.completions.create(
                model=settings.llm_model,
                max_tokens=2048,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": f"""Document languages: {", ".join(document_languages)}

Text spans from page {page["page_number"]}:
{numbered_span_string}

Identify which spans contain sensitive information."""
                    }
                ]
            )
            raw = (response.choices[0].message.content or "").strip()
            parsed = json.loads(raw)
            for item in parsed:
                confidence = float(item["confidence"])
                span_index = int(item["span_index"])
                if confidence >= settings.redaction_confidence_threshold:
                    matching_span = next(
                        (
                            span for index, span in enumerate(spans)
                            if span.get("index", index) == span_index
                        ),
                        None
                    )
                    if matching_span is not None:
                        targets.append(
                            RedactionTarget(
                                text=matching_span.get("text", ""),
                                category=item["category"],
                                confidence=confidence,
                                reason=item["reason"],
                                page_number=page["page_number"],
                                span_index=span_index,
                            )
                        )
        except json.JSONDecodeError:
            console.print(f"Invalid JSON response: {raw}", style="yellow")
        except Exception as error:
            console.print(f"Sensitive span identification failed: {error}", style="red")

    return targets


def format_targets_summary(targets: list[RedactionTarget]) -> str:
    """Format redaction targets as one concise audit line per target."""
    return "\n".join(
        f"Page {target.page_number} | {target.category} | "
        f"confidence: {target.confidence:.0%} | {target.text}"
        for target in targets
    )
