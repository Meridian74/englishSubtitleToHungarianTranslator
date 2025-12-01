#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
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
    "Angular", "React", "Vue", "Node.js", "Docker", "Kubernetes",
    "Git", "GitHub", "TypeScript", "JavaScript", "AWS", "Azure"
]

MARKER_FMT = "[[{num:05d}]]"
MAX_CHARS_PER_LINE = 60  # ha egy sor <= ennél, egy sorban marad
MAX_LINES_PER_BLOCK = 2

CONJUNCTIONS = ["and","or","but","so","therefore","because","who","what","how",
                "which","when","where","while","although","if","though","as","until","unless"]

# ----------------------------
# Segédfüggvények
# ----------------------------
def protect_terms(text):
    for term in EXCEPTIONS:
        # szóközök biztosítása, hogy a fordító ne ragassza a környező szöveghez
        text = text.replace(term, f" §{term}§ ")
    return text

def unprotect_terms(text):
    text = text.replace("§", "")
    return re.sub(r'\s+', ' ', text).strip()

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

def translate_with_preserved_markers(full_text):
    pattern = r'(\[\[\d{5}\]\])'
    parts = re.split(pattern, full_text)
    out_parts = []
    for part in parts:
        if re.fullmatch(pattern, part):
            out_parts.append(part)
        elif part.strip() == "":
            out_parts.append(part)
        else:
            chunk = protect_terms(part)
            translated_chunk = argostranslate.translate.translate(chunk, "en", "hu")
            translated_chunk = unprotect_terms(translated_chunk)
            out_parts.append(translated_chunk)
    return "".join(out_parts)

def fix_protected_terms_and_markers(translated_text):
    # EXCEPTIONS szavak rendezése (ha szükséges)
    for term in EXCEPTIONS:
        translated_text = re.sub(rf'\s*{re.escape(term)}\s*', f' {term} ', translated_text)
    translated_text = translated_text.replace('#', '')
    translated_text = re.sub(r'\s+', ' ', translated_text).strip()
    return translated_text

# ------------------------------------------------
#  ÚJ FUNKCIÓ – sorhosszok kiegyensúlyozása flip-flop nélkül
# ------------------------------------------------
def balance_two_lines(line1, line2):
    """
    Ha a két sor hossza >10 karakterrel eltér, akkor a hosszabbból átpakolunk
    egy szót a rövidebbhez. Maximum 3 iteráció, hogy ne legyen végtelen ciklus.
    """
    for _ in range(3):
        len1, len2 = len(line1), len(line2)
        diff = abs(len1 - len2)
        if diff <= 10:
            break

        # Felső hosszabb → levágjuk a végét és áttesszük a másik elejére
        if len1 > len2:
            parts = line1.split()
            if len(parts) <= 1:
                break
            moved = parts.pop()  # utolsó szó
            line1 = " ".join(parts)
            line2 = moved + " " + line2

        # Alsó hosszabb → levágjuk az elejéről és hozzácsapjuk a felső végére
        else:
            parts = line2.split()
            if len(parts) <= 1:
                break
            moved = parts.pop(0)  # első szó
            line2 = " ".join(parts)
            line1 = line1 + " " + moved

    return line1.strip(), line2.strip()

# ----------------------------
# wrap_text_to_lines módosítva: hívja a balance_two_lines()-t
# - ha a teljes szöveg <= max_chars => 1 sor
# - különben max 2 sor, és ha nagyon egyenetlen, kicsit átrendezünk
# ----------------------------
def wrap_text_to_lines(text, max_chars=MAX_CHARS_PER_LINE, max_lines=MAX_LINES_PER_BLOCK):
    words = text.split()

    # --- ha a teljes szöveg rövidebb mint limit, tartsuk egy sorban ---
    if len(text) <= max_chars:
        return text

    lines = []
    cur_line = []
    cur_len = 0

    for i, word in enumerate(words):
        extra_space = 1 if cur_line else 0
        if cur_len + len(word) + extra_space <= max_chars:
            cur_line.append(word)
            cur_len += len(word) + extra_space
        else:
            lines.append(" ".join(cur_line))
            cur_line = [word]
            cur_len = len(word)
            if len(lines) >= max_lines:
                # maradékot az utolsó sorhoz fűzzük
                rest = " ".join(cur_line + words[i+1:])
                lines[-1] = lines[-1] + " " + rest
                break

    if cur_line:
        lines.append(" ".join(cur_line))

    # ha egy sor maradt, adjunk vissza egy sort
    if len(lines) == 1:
        return lines[0]

    # ha több mint 2 sor jött létre, csak az első kettőt használjuk
    if len(lines) > 2:
        lines = lines[:2]

    # kiegyensúlyozás
    line1, line2 = balance_two_lines(lines[0], lines[1])
    # ha a végeredmény rövidebb mindkét sor összege kisebb mint max_chars, lehet egy sorba tenni
    if len(line1) + 1 + len(line2) <= max_chars:
        return (line1 + " " + line2).strip()
    return line1 + "\n" + line2

# ----------------------------
# Tagmondat-split heurisztika
# ----------------------------
def split_into_clauses(sentence: str) -> list:
    parts = re.split(r'(?<=[,;:])\s+', sentence)
    final_parts = []
    for part in parts:
        subparts = re.split(r'\s+(?=(?:' + "|".join(CONJUNCTIONS) + r')\b)', part, flags=re.IGNORECASE)
        for sp in subparts:
            s = sp.strip()
            if s:
                final_parts.append(s)
    return final_parts

# ----------------------------
# SRT beolvasás -> teljes angol szöveg
# ----------------------------
def read_srt_full_text(srt_path):
    text = Path(srt_path).read_text(encoding="utf-8")
    text = text.replace("\r\n","\n").replace("\r","\n")
    blocks = [b for b in text.split("\n\n") if b.strip()]
    full_text = []
    for block in blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        if len(lines)<3:
            continue
        content = " ".join(lines[2:]).strip()
        full_text.append(content)
    full_text_str = " ".join(full_text)
    # log - ideiglenes, törölhető később
    print("\n--- Teljes angol szöveg ---\n")
    print(full_text_str)
    return full_text_str, blocks

# ----------------------------
# Markerelés + timestamp hozzárendelés (szólista-alapú)
# ----------------------------
def mark_text_and_assign_timestamps(full_text, srt_blocks):
    marker_id = 1
    marker_to_timestamp = {}
    marked_clauses = []

    full_words = full_text.split()  # teljes szólista

    block_word_indices = []
    word_idx = 0
    for block in srt_blocks:
        lines = [ln for ln in block.split("\n") if ln.strip()]
        content = " ".join(lines[2:]).strip()
        words = content.split()
        block_word_indices.append((word_idx, word_idx+len(words)-1, lines[1].strip()))
        word_idx += len(words)

    sentences = sent_tokenize(full_text)
    for sent in sentences:
        clauses = split_into_clauses(sent)
        for clause in clauses:
            marker = MARKER_FMT.format(num=marker_id)
            clause_words = clause.strip().split()
            found_idx = -1
            for i in range(len(full_words)-len(clause_words)+1):
                if full_words[i:i+len(clause_words)] == clause_words:
                    found_idx = i
                    break
            if found_idx == -1:
                ts_assigned = "00:00:00,000 --> 00:00:04,000"
            else:
                ts_assigned = "00:00:00,000 --> 00:00:04,000"
                for start_idx, end_idx, ts in block_word_indices:
                    if found_idx >= start_idx and found_idx <= end_idx:
                        ts_assigned = ts
                        # csúsztatási logika itt (ha szükséges a speciális esetekhez)
                        # *** Ha a clause utolsó szava kötőszó és az a blokk végén van,
                        #     akkor csúsztassuk a következő blokk timestamp-jére (ha létezik).
                        if clause_words[-1].lower() in CONJUNCTIONS:
                            idx_block = block_word_indices.index((start_idx, end_idx, ts))
                            if found_idx == end_idx and idx_block + 1 < len(block_word_indices):
                                ts_assigned = block_word_indices[idx_block+1][2]
                                print(f"[Marker csúsztatva] {marker} -> {ts_assigned}")
                        break
            marker_to_timestamp[marker] = ts_assigned
            marked_clauses.append((marker, clause.strip(), ts_assigned))
            print(f"[Marker] {marker} '{clause.strip()}' -> {ts_assigned}")
            marker_id +=1

    return marked_clauses, marker_to_timestamp

# ----------------------------
# Teljes angol mondatok készítése fordításhoz
# ----------------------------
def build_sentences_for_translation(marked_clauses):
    sentences = []
    cur_sentence = []
    for marker, clause, _ in marked_clauses:
        cur_sentence.append(f"{marker} {clause}")
        if clause.endswith(('.', '?', '!')):
            sentences.append(" ".join(cur_sentence))
            cur_sentence = []
    if cur_sentence:
        sentences.append(" ".join(cur_sentence))
    print("\n--- Fordítandó angol mondatok ---\n")
    for s in sentences:
        print(s)
    return sentences

# ----------------------------
# SRT generálás a lefordított mondatokból
# ----------------------------
def generate_srt_from_translated(translated_sentences, marker_to_timestamp, output_path):
    ts_to_texts = {}
    pattern = r'(\[\[\d{5}\]\])'

    for sent in translated_sentences:
        # biztosítjuk, hogy marker-ek külön tokenként legyenek
        sent = re.sub(r'(\[\[\d{5}\]\])', r' \1 ', sent)
        parts = re.split(pattern, sent)
        i=0
        while i<len(parts):
            if re.fullmatch(pattern, parts[i]):
                marker = parts[i]
                text_piece = parts[i+1] if i+1<len(parts) else ""
                ts = marker_to_timestamp.get(marker, "00:00:00,000 --> 00:00:04,000")
                ts_to_texts.setdefault(ts, []).append(text_piece.strip())
                i+=2
            else:
                i+=1

    final_blocks = []
    idx = 1
    for ts, texts in ts_to_texts.items():
        block_text = " ".join([t for t in texts if t])
        block_text = re.sub(pattern, '', block_text)
        block_text_wrapped = wrap_text_to_lines(block_text)
        final_blocks.append((idx, ts, block_text_wrapped))
        idx+=1

    srt_lines = []
    for idx, ts, text in final_blocks:
        srt_lines.append(str(idx))
        srt_lines.append(ts)
        srt_lines.append(text)
        srt_lines.append("")
    Path(output_path).write_text("\n".join(srt_lines).strip()+"\n", encoding="utf-8")

# ----------------------------
# Fő folyamat
# ----------------------------
def process_and_generate_srt(en_srt_path, out_srt_path):
    ensure_argos_model()
    full_text, srt_blocks = read_srt_full_text(en_srt_path)
    marked_clauses, marker_to_timestamp = mark_text_and_assign_timestamps(full_text, srt_blocks)
    sentences_for_translation = build_sentences_for_translation(marked_clauses)

    translated_sentences = []
    print("\n--- Fordítás megkezdése ---\n")
    for sent in sentences_for_translation:
        translated = translate_with_preserved_markers(sent)
        translated = fix_protected_terms_and_markers(translated)
        print(f"\n[Fordítás]\n{sent}\n→ {translated}")
        translated_sentences.append(translated)

    generate_srt_from_translated(translated_sentences, marker_to_timestamp, out_srt_path)
    print(f"\n✔ KÉSZ → {out_srt_path}")




# ----------------------------
if __name__=="__main__":
    INPUT_SRT = "2. What Are We Building.eng.srt"
    OUTPUT_SRT = "2. What Are We Building.hun.srt"
    process_and_generate_srt(INPUT_SRT, OUTPUT_SRT)
