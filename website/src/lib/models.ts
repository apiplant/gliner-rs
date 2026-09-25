/** GLiNER2 checkpoints the browser demos can download from Hugging Face,
 * mirroring the `--model-variant` choices the four CLI binaries expose. */

export const CHECKPOINT_FILES = ["config.json", "encoder_config/config.json", "tokenizer.json", "model.safetensors"] as const;

export interface ModelDef {
  key: string;
  label: string;
  hfRepo: string;
  /** Rough `model.safetensors` size, for the picker — not fetched ahead of time. */
  approxSizeMb: number;
  description: string;
}

export const ENTITY_MODELS: ModelDef[] = [
  {
    key: "small",
    label: "gliner2.5-small",
    hfRepo: "fastino/gliner2.5-small-v1",
    approxSizeMb: 282,
    description: "Smallest general-purpose checkpoint — fastest in the browser.",
  },
  {
    key: "multi",
    label: "gliner2.5-multi",
    hfRepo: "fastino/gliner2.5-multi-v1",
    approxSizeMb: 1096,
    description: "Multilingual, best general accuracy. Larger download.",
  },
  {
    key: "base",
    label: "gliner2.5-base",
    hfRepo: "fastino/gliner2.5-base-v1",
    approxSizeMb: 738,
    description: "English-focused, base size.",
  },
  {
    key: "decide",
    label: "gliner2.5-Decide",
    hfRepo: "fastino/GLiNER2.5-Decide",
    approxSizeMb: 1856,
    description: "Decision/verification checkpoint (English), 0.5B params.",
  },
  {
    key: "decide-multi",
    label: "gliner2.5-multi-Decide",
    hfRepo: "fastino/GLiNER2.5-multi-Decide",
    approxSizeMb: 1097,
    description: "Multilingual decision/verification checkpoint, 0.3B params.",
  },
  // decide-1b (fastino/GLiNER2.5-Decide-1B) is deliberately not listed here:
  // its model.safetensors alone is ~4.5GB, which leaves no headroom under
  // wasm32's hard 4GiB linear-memory cap once the model's own runtime
  // buffers are allocated. It works fine on the native CLI (`gliner`,
  // `gliner-classify`), just not in the browser demo.
];

export const PII_MODELS: ModelDef[] = [
  {
    key: "privacy",
    label: "privacy-filter-PII",
    hfRepo: "fastino/gliner2-privacy-filter-PII-multi",
    approxSizeMb: 1172,
    description: "42-label PII taxonomy (names, contact info, IDs, financial, credentials).",
  },
];

export const GUARDRAIL_MODELS: ModelDef[] = [
  {
    key: "gliguard",
    label: "gliguard-LLMGuardrails",
    hfRepo: "fastino/gliguard-LLMGuardrails-300M",
    approxSizeMb: 795,
    description: "Dedicated jailbreak/toxicity/refusal moderation checkpoint.",
  },
  {
    key: "guardrails-pii",
    label: "Guardrails-PII-Multi",
    hfRepo: "fastino/GLiNER2-Guardrails-PII-Multi",
    approxSizeMb: 1172,
    description: "Combined guardrails + PII checkpoint.",
  },
];

export function hfFileUrl(repo: string, file: string): string {
  return `https://huggingface.co/${repo}/resolve/main/${file}`;
}

/** The default PII taxonomy `gliner-pii` uses. */
export const PII_DEFAULT_LABELS = [
  "person",
  "full_name",
  "first_name",
  "middle_name",
  "last_name",
  "date_of_birth",
  "email",
  "phone_number",
  "address",
  "street_address",
  "city",
  "state_or_region",
  "postal_code",
  "country",
  "government_id",
  "national_id_number",
  "passport_number",
  "drivers_license_number",
  "tax_id",
  "tax_number",
  "bank_account",
  "account_number",
  "routing_number",
  "iban",
  "payment_card",
  "card_number",
  "card_expiry",
  "card_cvv",
  "username",
  "ip_address",
  "account_id",
  "sensitive_account_id",
  "password",
  "secret",
  "api_key",
  "access_token",
  "recovery_code",
  "sensitive_date",
  "document_date",
  "expiration_date",
  "transaction_date",
];

export const GUARDRAIL_TOXICITY_LABELS = [
  "violence",
  "sexual_content",
  "hate_speech",
  "self_harm",
  "pii_exposure",
  "misinformation",
  "regulated_advice",
];

export const GUARDRAIL_JAILBREAK_LABELS = [
  "prompt_injection",
  "jailbreak_attempt",
  "roleplay_bypass",
  "obfuscated_attack",
];
