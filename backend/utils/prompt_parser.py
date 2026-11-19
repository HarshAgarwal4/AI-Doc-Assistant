import html
import re
import unicodedata

from fastapi import HTTPException, status

# 1. Dangerous executable/embed tags and their inner content
SCRIPT_TAGS_PATTERN = re.compile(
    r"<(script|style|iframe|object|embed|svg|applet)[^>]*>.*?</\1>",
    re.IGNORECASE | re.DOTALL,
)

# 2. All remaining HTML/XML tags (prevents prompt breakout like </user_question>)
ALL_TAGS_PATTERN = re.compile(r"<[^>]+>")

# 3. Special LLM structural tokens (ChatML, Llama, Mistral, Anthropic tokens)
SPECIAL_TOKENS_PATTERN = re.compile(
    r"(<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|\[/?INST\]|<<?/?SYS>>?|\[/?SYSTEM\]|\[/?USER\]|\[/?ASSISTANT\])",
    re.IGNORECASE,
)

# 4. Common prompt injection overrides, JSON/chat simulation, and jailbreaks
PROMPT_INJECTION_PATTERN = re.compile(
    r"\b(ignore|disregard|forget|override|bypass|reject|cancel|drop)\s+(all\s+)?(previous|prior|above|earlier|former|past)\s+(instructions|prompts|rules|commands|context|questions?|tasks?)\b|"
    r"\b(continue\s+(this\s+)?(json|log|conversation|dialogue|chat|script|prompt)|next\s+assistant\s+message)\b|"
    r"\[?\s*\{\s*[\"']role[\"']\s*:\s*[\"'](user|assistant|system)[\"'][^}]*\}\s*,?|"
    r"\{\s*[\"']role[\"']\s*:\s*[\"']assistant[\"']\s*,?\s*[\"']?content[\"']?\s*:\s*[\"']?|"
    r"\b(developer\s+mode|test\s+mode|unrestricted\s+mode|god\s+mode|jailbreak|dan\s+mode)\b|"
    r"\b(you\s+are\s+now|pretend\s+to\s+be|act\s+as|roleplay\s+as)\s+(a\s+)?(developer|unrestricted|jailbreak|dan|unfiltered|system)\b|"
    r"\[\s*system\s*(override|message|prompt|directive)[^\]]*\]|"
    r"^(system|assistant|system\s*prompt|new\s*system\s*message)\s*:\s*",
    re.IGNORECASE | re.MULTILINE,
)

# 5. Repeated delimiter bars used to simulate prompt boundaries (e.g., ---, ===, ___)
REPEATED_DELIMITERS_PATTERN = re.compile(r"[-=_~*#]{3,}")

# 6. Non-printable control characters (excluding standard whitespace: tab, newline, return)
CONTROL_CHARS_PATTERN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

# 7. Invisible zero-width unicode characters used to bypass text filters
ZERO_WIDTH_CHARS_PATTERN = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff]")

MAX_PROMPT_LENGTH = 2000


def parse_prompt(raw_prompt: str) -> str:
    """
    Strictly parse and sanitize a user prompt to extract clean text content:
    1. Validates that input is a non-empty string.
    2. Strips invisible zero-width and non-printable control characters.
    3. Normalizes Unicode representations into standard NFKC form.
    4. Decodes HTML entities and removes script/style/iframe blocks with content.
    5. Strips all HTML/XML tags to prevent structure breakout (e.g. </user_question>).
    6. Removes LLM control tokens ([INST], <|im_start|>, <<SYS>>, etc.).
    7. Neutralizes common prompt injection overrides.
    8. Collapses fake boundary delimiters and normalizes whitespace.
    9. Enforces maximum length constraints.
    """
    if not raw_prompt or not isinstance(raw_prompt, str):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Prompt must be a non-empty string.",
        )

    # Step 1: Unicode normalization & removal of zero-width / control characters
    text = unicodedata.normalize("NFKC", raw_prompt)
    text = ZERO_WIDTH_CHARS_PATTERN.sub("", text)
    text = CONTROL_CHARS_PATTERN.sub("", text)

    # Step 2: Unescape HTML entities so hidden tags like &lt;script&gt; are exposed
    text = html.unescape(text)

    # Step 3: Strip dangerous executable/embed tags and their content
    text = SCRIPT_TAGS_PATTERN.sub("", text)

    # Step 4: Strip all remaining HTML / XML markup tags
    text = ALL_TAGS_PATTERN.sub(" ", text)

    # Step 5: Remove LLM control tokens and chat template markers
    text = SPECIAL_TOKENS_PATTERN.sub(" ", text)

    # Step 6: Neutralize prompt injection phrases
    text = PROMPT_INJECTION_PATTERN.sub("", text)

    # Step 7: Normalize repeated fake boundary delimiters
    text = REPEATED_DELIMITERS_PATTERN.sub(" ", text)

    # Step 8: Collapse whitespace and trim
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    # Step 9: Validate final clean text
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid prompt: prompt contains only forbidden tags, scripts, or delimiters.",
        )

    # Step 10: Enforce length limit
    if len(text) > MAX_PROMPT_LENGTH:
        text = text[:MAX_PROMPT_LENGTH].rsplit(" ", 1)[0].strip()

    return text
