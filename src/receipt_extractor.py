import json
import re
from pathlib import Path
import lmstudio as lms


ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "raw" / "nota-sample.png"
OUTPUT_PATH = ROOT / "reports" / "receipt.json"
MODEL_NAME = "qwen2.5-vl-7b-instruct"


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"File gambar tidak ditemukan di: {IMAGE_PATH}")

    image = lms.prepare_image(str(IMAGE_PATH))
    model = lms.llm(MODEL_NAME)

    chat = lms.Chat()
    chat.add_user_message(
        "Baca nota. Ekstrak merchant, tanggal, item (list of object), subtotal, pajak, dan total. "
        "Kembalikan HANYA format JSON valid tanpa kata pengantar, penjelasan, atau formatting markdown. "
        "Jika pajak tidak terlihat, isi 0. Jangan mengarang.",
        images=[image],
    )

    print("Sedang memproses nota dengan LM Studio...")
    prediction = model.respond(chat)
    raw_content = prediction.content.strip()


    json_match = re.search(r"\{.*\}", raw_content, re.DOTALL)
    
    if not json_match:
        print("❌ Model tidak mengembalikan format JSON yang valid. Output mentah:")
        print(raw_content)
        return

    try:
        result = json.loads(json_match.group(0))
    except json.JSONDecodeError as e:
        print(f"❌ Gagal memuat JSON: {e}")
        print("Output mentah dari model:")
        print(raw_content)
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    formatted_json = json.dumps(result, indent=2, ensure_ascii=False)
    OUTPUT_PATH.write_text(formatted_json, encoding="utf-8")

    print(f"\n✅ Hasil berhasil disimpan ke: {OUTPUT_PATH}\n")
    print(formatted_json)


if __name__ == "__main__":
    main()
