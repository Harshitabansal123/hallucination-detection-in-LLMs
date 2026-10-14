import json
import shutil
from pathlib import Path
from datetime import datetime
from collections import Counter

# Run this script from D:\hallucination-detection
DATASET = Path("data/processed/claims.json")

# These are drafted examples, not yet verified gold-standard annotations.
NEW_SAMPLES = [
    {
        "language": "mr",
        "question": "पृथ्वीबद्दल तीन तथ्ये सांगा.",
        "answer": "पृथ्वी सूर्याभोवती फिरते. पृथ्वीचा एक नैसर्गिक उपग्रह आहे, ज्याला चंद्र म्हणतात. पृथ्वीच्या पृष्ठभागाचा मोठा भाग पाण्याने व्यापलेला आहे.",
        "claims": [
            "पृथ्वी सूर्याभोवती फिरते.",
            "पृथ्वीचा नैसर्गिक उपग्रह चंद्र आहे.",
            "पृथ्वीच्या पृष्ठभागाचा मोठा भाग पाण्याने व्यापलेला आहे."
        ]
    },
    {
        "language": "mr",
        "question": "भारताच्या राजधानीबद्दल आणि राष्ट्रध्वजाबद्दल सांगा.",
        "answer": "भारताची राजधानी नवी दिल्ली आहे. भारताच्या राष्ट्रध्वजात केशरी, पांढरा आणि हिरवा असे तीन आडवे रंगपट्टे आहेत. ध्वजाच्या मध्यभागी निळ्या रंगाचे अशोकचक्र आहे.",
        "claims": [
            "भारताची राजधानी नवी दिल्ली आहे.",
            "भारताच्या राष्ट्रध्वजात केशरी रंगाचा आडवा पट्टा आहे.",
            "भारताच्या राष्ट्रध्वजात पांढऱ्या रंगाचा आडवा पट्टा आहे.",
            "भारताच्या राष्ट्रध्वजात हिरव्या रंगाचा आडवा पट्टा आहे.",
            "भारताच्या राष्ट्रध्वजाच्या मध्यभागी निळ्या रंगाचे अशोकचक्र आहे."
        ]
    },
    {
        "language": "mr",
        "question": "प्रकाशसंश्लेषण म्हणजे काय?",
        "answer": "हिरव्या वनस्पती प्रकाशसंश्लेषणाद्वारे सूर्यप्रकाशाची ऊर्जा वापरतात. या प्रक्रियेत वनस्पती कार्बन डायऑक्साइड आणि पाणी वापरून साखर तयार करतात. प्रकाशसंश्लेषणादरम्यान ऑक्सिजनही बाहेर सोडला जातो.",
        "claims": [
            "हिरव्या वनस्पती प्रकाशसंश्लेषणाद्वारे सूर्यप्रकाशाची ऊर्जा वापरतात.",
            "प्रकाशसंश्लेषणात वनस्पती कार्बन डायऑक्साइड वापरतात.",
            "प्रकाशसंश्लेषणात वनस्पती पाणी वापरतात.",
            "प्रकाशसंश्लेषणाद्वारे वनस्पती साखर तयार करतात.",
            "प्रकाशसंश्लेषणादरम्यान ऑक्सिजन बाहेर सोडला जातो."
        ]
    },
    {
        "language": "mr",
        "question": "पाण्याच्या उकळण्याच्या तापमानाबद्दल सांगा.",
        "answer": "सामान्य वातावरणीय दाबावर शुद्ध पाणी साधारणपणे १०० अंश सेल्सिअस तापमानाला उकळते. वातावरणीय दाब बदलल्यास पाण्याचा उकळण्याचा बिंदू बदलू शकतो.",
        "claims": [
            "सामान्य वातावरणीय दाबावर शुद्ध पाणी साधारणपणे १०० अंश सेल्सिअस तापमानाला उकळते.",
            "वातावरणीय दाब बदलल्यास पाण्याचा उकळण्याचा बिंदू बदलू शकतो."
        ]
    },
    {
        "language": "mr",
        "question": "संगणकाच्या CPU चे कार्य काय आहे?",
        "answer": "CPU म्हणजे सेंट्रल प्रोसेसिंग युनिट. ते संगणकाच्या सूचना कार्यान्वित करते. CPU अंकगणितीय आणि तार्किक क्रिया करण्यासाठी वापरले जाते.",
        "claims": [
            "CPU चे पूर्ण नाव सेंट्रल प्रोसेसिंग युनिट आहे.",
            "CPU संगणकाच्या सूचना कार्यान्वित करते.",
            "CPU अंकगणितीय क्रिया करण्यासाठी वापरले जाते.",
            "CPU तार्किक क्रिया करण्यासाठी वापरले जाते."
        ]
    },
    {
        "language": "bn",
        "question": "পৃথিবী সম্পর্কে তিনটি তথ্য বলো।",
        "answer": "পৃথিবী সূর্যের চারদিকে ঘোরে। পৃথিবীর একটি প্রাকৃতিক উপগ্রহ আছে, যার নাম চাঁদ। পৃথিবীর পৃষ্ঠের বড় অংশ জল দিয়ে আবৃত।",
        "claims": [
            "পৃথিবী সূর্যের চারদিকে ঘোরে।",
            "পৃথিবীর প্রাকৃতিক উপগ্রহের নাম চাঁদ।",
            "পৃথিবীর পৃষ্ঠের বড় অংশ জল দিয়ে আবৃত।"
        ]
    },
    {
        "language": "bn",
        "question": "ভারতের রাজধানী ও জাতীয় পতাকা সম্পর্কে বলো।",
        "answer": "ভারতের রাজধানী নয়াদিল্লি। ভারতের জাতীয় পতাকায় গেরুয়া, সাদা ও সবুজ—এই তিনটি অনুভূমিক রঙের দাগ রয়েছে। পতাকার মাঝখানে নীল রঙের অশোকচক্র রয়েছে।",
        "claims": [
            "ভারতের রাজধানী নয়াদিল্লি।",
            "ভারতের জাতীয় পতাকায় গেরুয়া রঙের অনুভূমিক দাগ রয়েছে।",
            "ভারতের জাতীয় পতাকায় সাদা রঙের অনুভূমিক দাগ রয়েছে।",
            "ভারতের জাতীয় পতাকায় সবুজ রঙের অনুভূমিক দাগ রয়েছে।",
            "ভারতের জাতীয় পতাকার মাঝখানে নীল রঙের অশোকচক্র রয়েছে।"
        ]
    },
    {
        "language": "bn",
        "question": "সালোকসংশ্লেষ কী?",
        "answer": "সবুজ উদ্ভিদ সালোকসংশ্লেষের মাধ্যমে সূর্যালোকের শক্তি ব্যবহার করে। এই প্রক্রিয়ায় উদ্ভিদ কার্বন ডাই-অক্সাইড ও জল ব্যবহার করে শর্করা তৈরি করে। সালোকসংশ্লেষের সময় অক্সিজেনও নির্গত হয়।",
        "claims": [
            "সবুজ উদ্ভিদ সালোকসংশ্লেষের মাধ্যমে সূর্যালোকের শক্তি ব্যবহার করে।",
            "সালোকসংশ্লেষের সময় উদ্ভিদ কার্বন ডাই-অক্সাইড ব্যবহার করে।",
            "সালোকসংশ্লেষের সময় উদ্ভিদ জল ব্যবহার করে।",
            "সালোকসংশ্লেষের মাধ্যমে উদ্ভিদ শর্করা তৈরি করে।",
            "সালোকসংশ্লেষের সময় অক্সিজেন নির্গত হয়।"
        ]
    },
    {
        "language": "bn",
        "question": "জলের স্ফুটনাঙ্ক সম্পর্কে বলো।",
        "answer": "স্বাভাবিক বায়ুচাপে বিশুদ্ধ জল সাধারণত ১০০ ডিগ্রি সেলসিয়াসে ফোটে। বায়ুচাপ পরিবর্তিত হলে জলের স্ফুটনাঙ্কও পরিবর্তিত হতে পারে।",
        "claims": [
            "স্বাভাবিক বায়ুচাপে বিশুদ্ধ জল সাধারণত ১০০ ডিগ্রি সেলসিয়াসে ফোটে।",
            "বায়ুচাপ পরিবর্তিত হলে জলের স্ফুটনাঙ্ক পরিবর্তিত হতে পারে।"
        ]
    },
    {
        "language": "bn",
        "question": "কম্পিউটারের CPU কী কাজ করে?",
        "answer": "CPU-এর পূর্ণরূপ Central Processing Unit। এটি কম্পিউটারের নির্দেশ কার্যকর করে। CPU গাণিতিক ও যৌক্তিক কাজ সম্পাদন করে।",
        "claims": [
            "CPU-এর পূর্ণরূপ Central Processing Unit।",
            "CPU কম্পিউটারের নির্দেশ কার্যকর করে।",
            "CPU গাণিতিক কাজ সম্পাদন করে।",
            "CPU যৌক্তিক কাজ সম্পাদন করে।"
        ]
    },
    {
        "language": "bn",
        "question": "সৌরজগতের সূর্য সম্পর্কে বলো।",
        "answer": "সূর্য সৌরজগতের একটি নক্ষত্র। পৃথিবী সূর্য থেকে আলো ও তাপ পায়। পৃথিবী সূর্যকে একবার প্রদক্ষিণ করতে প্রায় এক বছর সময় নেয়।",
        "claims": [
            "সূর্য সৌরজগতের একটি নক্ষত্র।",
            "পৃথিবী সূর্য থেকে আলো পায়।",
            "পৃথিবী সূর্য থেকে তাপ পায়।",
            "পৃথিবী সূর্যকে একবার প্রদক্ষিণ করতে প্রায় এক বছর সময় নেয়।"
        ]
    }
]


def main():
    if not DATASET.exists():
        raise SystemExit(
            f"Dataset not found: {DATASET.resolve()}. "
            "Run this script from the repository root."
        )

    with DATASET.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if isinstance(data, dict) and isinstance(data.get("samples"), list):
        samples = data["samples"]
        wrapper = data
    elif isinstance(data, list):
        samples = data
        wrapper = None
    else:
        raise SystemExit(
            "Unsupported JSON structure. Expected a list or an object "
            "containing a 'samples' list."
        )

    existing_ids = {
        str(sample["sample_id"])
        for sample in samples
        if isinstance(sample, dict) and sample.get("sample_id") is not None
    }

    existing_questions = {
        (str(sample.get("language", "")), str(sample.get("question", "")).strip())
        for sample in samples
        if isinstance(sample, dict)
    }

    numeric_ids = []
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        sample_id = str(sample.get("sample_id", ""))
        digits = "".join(char for char in sample_id if char.isdigit())
        if digits:
            numeric_ids.append(int(digits))

    next_id = max(numeric_ids, default=0) + 1
    additions = []

    for item in NEW_SAMPLES:
        language = item["language"]
        question = item["question"]

        if (language, question.strip()) in existing_questions:
            print(f"Skipping existing question ({language}): {question}")
            continue

        sample_id = f"sample_{next_id:06d}"
        next_id += 1

        while sample_id in existing_ids:
            sample_id = f"sample_{next_id:06d}"
            next_id += 1

        claims = [
            {
                "claim_id": f"{sample_id}_claim_{index:03d}",
                "claim": claim_text
            }
            for index, claim_text in enumerate(item["claims"], start=1)
        ]

        additions.append({
            "sample_id": sample_id,
            "question": question,
            "language": language,
            "model": "manually_drafted_multilingual_example",
            "answer": item["answer"],
            "claims": claims,
            "review_status": "pending_language_and_fact_review",
            "data_origin": "manually_drafted_example_not_yet_verified"
        })

        existing_ids.add(sample_id)
        existing_questions.add((language, question.strip()))

    if not additions:
        print("No new samples added; all candidate questions already exist.")
        return

    backup = DATASET.with_name(
        f"claims.backup.{datetime.now():%Y%m%d_%H%M%S}.json"
    )
    shutil.copy2(DATASET, backup)

    samples.extend(additions)

    if wrapper is not None:
        wrapper["samples"] = samples
        output = wrapper
    else:
        output = samples

    temp_path = DATASET.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as file:
        json.dump(output, file, ensure_ascii=False, indent=2)
        file.write("\n")

    temp_path.replace(DATASET)

    language_counts = Counter(
        str(sample.get("language", "missing"))
        for sample in samples
        if isinstance(sample, dict)
    )

    added_claim_count = sum(len(sample["claims"]) for sample in additions)

    print(f"Backup created: {backup}")
    print(f"Added samples: {len(additions)}")
    print(f"Added claims: {added_claim_count}")
    print(f"Total samples now: {len(samples)}")
    print(f"Language counts: {dict(language_counts)}")
    print(
        "All added samples are pending language and factual review. "
        "They are not certified gold-standard annotations."
    )


if __name__ == "__main__":
    main()