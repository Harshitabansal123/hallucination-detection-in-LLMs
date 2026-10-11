
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
import torch


MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

LANGUAGE_NAMES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
}

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print("Configuring 4-bit quantization...")
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=quantization_config,
    device_map="auto",
    torch_dtype=torch.float16,
)

model.eval()
print("Model loaded successfully!")


def generate_answer(
    prompt: str,
    max_new_tokens: int = 150,
    language: str = "en",
) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty.")

    language = language.lower().strip()

    if language not in LANGUAGE_NAMES:
        raise ValueError(
            f"Unsupported language '{language}'. "
            f"Supported languages: {', '.join(LANGUAGE_NAMES)}"
        )

    language_name = LANGUAGE_NAMES[language]

    messages = [
        {
            "role": "system",
            "content": (
                f"You are a factual assistant. "
                f"Respond entirely in {language_name}. "
                f"Use correct {language_name} spelling and grammar. "
                "Preserve names, dates, numbers, and factual details. "
                "Do not switch languages unless a proper noun requires it."
            ),
        },
        {
            "role": "user",
            "content": prompt.strip(),
        },
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
    ).to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_tokens = outputs[0, inputs["input_ids"].shape[1]:]

    answer = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    ).strip()

    if not answer:
        raise RuntimeError("The model returned an empty answer.")

    return answer
