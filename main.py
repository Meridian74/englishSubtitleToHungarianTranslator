import os
import tempfile
import urllib.request
from pathlib import Path

import argostranslate.package
import argostranslate.translate
import nltk
from nltk.data import find
from nltk.tokenize import sent_tokenize

# ----------------------------
# NLTK punkt biztosítása
# ----------------------------
try:
    find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# ----------------------------
# Konfiguráció
# ----------------------------
EXCEPTIONS = [
    "Angular", "React", "React Native", "Python", "Vue", "Node.js", "Docker", "Kubernetes",
    "Git", "GitHub", "TypeScript", "JavaScript", "AWS", "Azure",
    "History API", "Zero to Mastery", "C++", "C#", "C-sharp", "Java", "Objective-C", "SQL"
]

MAX_CHARS_PER_LINE = 65
MAX_SENTENCES_PER_BLOCK = 5
MAX_CHARS_PER_BLOCK = 512


# ----------------------------
# Segédfüggvények
# ----------------------------
def ensure_argos_model():
    installed = argostranslate.package.get_installed_packages()
    if any(p.from_code == "en" and p.to_code == "hu" for p in installed):
        return
    model_url = "https://argos-net.com/v1/translate-en_hu-1_9.argosmodel"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".argosmodel")
    tmp_path = tmp.name
    tmp.close()
    urllib.request.urlretrieve(model_url, tmp_path)
    argostranslate.package.install_from_path(tmp_path)
    os.remove(tmp_path)


def read_srt(srt_path):
    raw = Path(srt_path).read_text(encoding="utf-8")
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    blocks = [b for b in raw.split("\n\n") if b.strip()]
    srt_blocks = []

    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if len(lines) < 3:
            continue

        # Index tisztítása - csak a BOM karakter eltávolítása az első sorból
        index_line = lines[0]
        if index_line.startswith('\ufeff'):
            index_line = index_line[1:]  # BOM eltávolítása

        index = int(index_line.strip())
        timestamp = lines[1]
        text = " ".join(lines[2:])
        srt_blocks.append({"index": index, "timestamp": timestamp, "text": text})

    return srt_blocks


def protect_terms(text):
    for term in EXCEPTIONS:
        text = text.replace(term, f"§{term}§")
    return text


def unprotect_terms(text):
    return text.replace("§", "")


def translate_text(text):
    protected = protect_terms(text)
    translated = argostranslate.translate.translate(protected, "en", "hu")
    return unprotect_terms(translated)


def format_srt_text(text, max_chars_per_line=MAX_CHARS_PER_LINE):
    """
    Formázza a szöveget SRT formátumra, szimmetrikusan két sorba tördelve.
    Az első sor legyen picit rövidebb, mint a második.
    """
    if not text.strip():
        return ""

    # Először távolítsuk el a felesleges whitespace-eket
    text = " ".join(text.split())

    # Ha a szöveg rövid, ne tördeld
    if len(text) <= max_chars_per_line:
        return text

    # Szavakra bontás
    words = text.split()

    # Keressük meg az optimális tördelési pontot
    # Próbáljuk a közepétől kezdeni
    optimal_break_index = len(words) // 2

    # Először keressünk egy jó pontot a közepétől lefelé
    for i in range(optimal_break_index, 0, -1):
        first_line = " ".join(words[:i])
        if len(first_line) <= max_chars_per_line:
            # Megnézzük, hogy a második sor sem túl hosszú-e
            second_line = " ".join(words[i:])
            if len(second_line) <= max_chars_per_line:
                # Találtunk jó tördelési pontot
                return f"{first_line}\n{second_line}"

    # Ha nem sikerült, próbáljuk a közepétől felfelé
    for i in range(optimal_break_index + 1, len(words)):
        first_line = " ".join(words[:i])
        if len(first_line) > max_chars_per_line:
            # Túl hosszú, menjünk egyet vissza
            if i > 1:
                first_line = " ".join(words[:i - 1])
                second_line = " ".join(words[i - 1:])
                return f"{first_line}\n{second_line}"

    # Ha minden else fails, osszuk felezve a szavakat
    mid = len(words) // 2
    first_line = " ".join(words[:mid])
    second_line = " ".join(words[mid:])

    return f"{first_line}\n{second_line}"


def write_srt(final_srt, out_srt_path):
    lines = []
    for blk in final_srt:
        lines.append(str(blk["index"]))
        lines.append(blk["timestamp"])
        lines.append(blk["text"])
        lines.append("")
    Path(out_srt_path).write_text("\n".join(lines), encoding="utf-8")


def process_and_generate_srt(en_srt_path, out_srt_path):
    ensure_argos_model()

    # 1. Beolvassuk az angol SRT-t
    eng_blocks = read_srt(en_srt_path)

    # 2. Kinyerjük az összes angol mondatot
    all_eng_sentences = []
    for block in eng_blocks:
        sentences = sent_tokenize(block["text"])
        all_eng_sentences.extend(sentences)

    print(f"=== ANGOL SRT FELDOLGOZÁSA ===")
    print(f"Angol blokkok száma: {len(eng_blocks)}")
    print(f"Angol mondatok száma: {len(all_eng_sentences)}")

    # 3. Fordítjuk blokkonként, figyelve a mondatszámot
    all_hun_sentences = []
    current_sentence_idx = 0

    while current_sentence_idx < len(all_eng_sentences):
        # Próbáljunk maximum 6 mondatot venni (vagy amennyi van hátra)
        max_sentences_to_take = min(MAX_SENTENCES_PER_BLOCK, len(all_eng_sentences) - current_sentence_idx)

        # Kezdjük a maximummal
        attempt_sentence_count = max_sentences_to_take

        while attempt_sentence_count > 0:
            # Kivesszük a mondatokat
            end_idx = current_sentence_idx + attempt_sentence_count
            block_sentences = all_eng_sentences[current_sentence_idx:end_idx]

            # Ellenőrizzük a karakterlimit-et
            block_chars = sum(len(sent) for sent in block_sentences)

            if block_chars > MAX_CHARS_PER_BLOCK:
                # Túl hosszú, csökkentsük a mondatszámot
                attempt_sentence_count -= 1
                continue

            # Blokk létrehozása
            eng_block_text = " ".join(block_sentences)

            print(f"\n--- Fordítási próba ({len(block_sentences)} mondat, {block_chars} karakter) ---")
            print(f"Angol: {eng_block_text[:150]}..." if len(eng_block_text) > 150 else f"Angol: {eng_block_text}")

            # Fordítás
            translated = translate_text(eng_block_text)

            # Mondatszám ellenőrzés
            hun_sentences = sent_tokenize(translated)

            if len(hun_sentences) == len(block_sentences):
                # SIKER! Ugyanannyi mondat
                all_hun_sentences.extend(hun_sentences)
                current_sentence_idx += len(block_sentences)

                print(f"✓ Sikeres fordítás: {len(block_sentences)} mondat -> {len(hun_sentences)} mondat")
                print(f"Magyar: {translated[:150]}..." if len(translated) > 150 else f"Magyar: {translated}")
                break
            else:
                # Nem ugyanannyi mondat
                print(f"✗ Sikertelen: angol={len(block_sentences)}, magyar={len(hun_sentences)}")

                if attempt_sentence_count > 1:
                    # Több mondatnál problémák vannak -> azonnal menjünk 1 mondatra
                    print("! Több mondat problémás -> azonnal 1 mondatra váltás")
                    attempt_sentence_count = 1
                else:
                    # 1 mondatnál is probléma -> elfogadjuk és továbblépünk
                    print("! Egy mondat eltéréssel, de elfogadjuk")
                    all_hun_sentences.extend(hun_sentences)
                    current_sentence_idx += 1
                    break
        else:
            # Ha a while loop végigment és nem talált megfelelő blokkot
            # (nagyon ritka eset, de le kell kezelni)
            if current_sentence_idx < len(all_eng_sentences):
                # Kényszerítjük az egy mondat fordítását
                single_sentence = all_eng_sentences[current_sentence_idx]
                print(f"\n--- Kényszer egy mondat ({len(single_sentence)} karakter) ---")
                print(f"Angol: {single_sentence}")

                translated_single = translate_text(single_sentence)
                hun_single_sentences = sent_tokenize(translated_single)

                all_hun_sentences.extend(hun_single_sentences)
                current_sentence_idx += 1

                print(f"Magyar: {translated_single}")

    # 4. Most meg kell feleltetnünk a magyar mondatokat az eredeti időblokkoknak
    print(f"\n=== MAGYAR MONDA TOK IDŐBLOKKHOZ RENDEZÉSE ===")
    print(f"Magyar mondatok száma: {len(all_hun_sentences)}")

    # Ellenőrizzük, hogy ugyanannyi magyar mondatunk van-e, mint angol
    if len(all_hun_sentences) != len(all_eng_sentences):
        print(f"!!! VÉGLEGES FIGYELEM: Mondatszám eltérés maradt!")
        print(f"!!! Angol: {len(all_eng_sentences)}, Magyar: {len(all_hun_sentences)}")
        print(f"!!! Különbség: {len(all_eng_sentences) - len(all_hun_sentences)} mondat")

    # Most el kell osztanunk a magyar mondatokat az angol időblokkok szerint
    hun_blocks = []
    hun_sentence_idx = 0

    for eng_block in eng_blocks:
        # Megnézzük, hány mondat van ebben az angol blokkban
        eng_sentences_in_block = sent_tokenize(eng_block["text"])
        num_sentences = len(eng_sentences_in_block)

        # Kiveszünk ugyanannyi magyar mondatot
        hun_sentences_for_block = []
        for _ in range(num_sentences):
            if hun_sentence_idx < len(all_hun_sentences):
                hun_sentences_for_block.append(all_hun_sentences[hun_sentence_idx])
                hun_sentence_idx += 1

        # Összefűzzük a mondatokat
        block_text = " ".join(hun_sentences_for_block)

        # Formázás
        formatted_text = format_srt_text(block_text)

        # Blokk létrehozása
        hun_blocks.append({
            "index": eng_block["index"],
            "timestamp": eng_block["timestamp"],
            "text": formatted_text
        })

        print(f"Blokk {eng_block['index']} ({eng_block['timestamp']}):")
        print(f"  Angol mondatok: {num_sentences}")
        print(f"  Magyar mondatok: {len(hun_sentences_for_block)}")
        if formatted_text:
            print(f"  Magyar szöveg: {formatted_text}")

    # 6. Kiírás
    write_srt(hun_blocks, out_srt_path)

    print(f"\n=== VÉGEREDMÉNY ===")
    print(f"Angol blokkok: {len(eng_blocks)}")
    print(f"Magyar blokkok: {len(hun_blocks)}")
    print(f"Angol mondatok: {len(all_eng_sentences)}")
    print(f"Magyar mondatok: {len(all_hun_sentences)}")
    print(f"Magyar SRT létrehozva: {out_srt_path}")

    # Összehasonlítás
    print(f"\n=== ÖSSZEHASONLÍTÁS (utolsó 4 blokk) ===")
    for i in range(max(0, len(eng_blocks) - 4), len(eng_blocks)):
        print(f"\nBlokk {i + 1} ({eng_blocks[i]['timestamp']}):")
        print(f"  Angol: {eng_blocks[i]['text']}")
        print(f"  Magyar: {hun_blocks[i]['text']}")

    return hun_blocks


if __name__ == "__main__":
    INPUT_SRT = ("1. Understanding Frameworks.srt")  # Írd át a fájlnevedre
    OUTPUT_SRT = "1. Understanding Frameworks.hun.srt"
    process_and_generate_srt(INPUT_SRT, OUTPUT_SRT)