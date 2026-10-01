import os
import re
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# "simple" = our own word-counting summarizer (no downloads, works on a phone)
# "ai"     = a real AI model from Hugging Face (needs a laptop or Colab, ~1 GB download)
MODE = "simple"

STOP = set("a an the and or but if so of to in on at by for with from as is are was were be been being it its this that these those he she they we you i his her their our your not no do does did have has had will would can could should may might than then there here which who whom what when where why how also into about over after before just more most very".split())


def split_sentences(text):
    clean = re.sub(r"\s+", " ", text).strip()
    parts = re.findall(r"[^.!?]+[.!?]+[\"')\]]*|[^.!?]+$", clean)
    return [p.strip() for p in parts if len(p.strip()) > 1]


def words_of(sentence):
    return [w for w in re.findall(r"[a-z0-9']+", sentence.lower())
            if w not in STOP and len(w) > 2]


def summarize_simple(text, ratio):
    sents = split_sentences(text)
    if len(sents) < 3:
        return None
    freq = {}
    for s in sents:
        for w in words_of(s):
            freq[w] = freq.get(w, 0) + 1
    scored = []
    for i, s in enumerate(sents):
        ws = words_of(s)
        score = sum(freq[w] for w in ws) / (len(ws) ** 0.7) if ws else 0
        if i == 0:
            score *= 1.25
        scored.append((score, i, s))
    n = max(1, min(len(sents) - 1, round(len(sents) * ratio)))
    top = sorted(scored, reverse=True)[:n]
    return " ".join(s for _, _, s in sorted(top, key=lambda x: x[1]))


_ai = None


def summarize_ai(text, ratio):
    global _ai
    if _ai is None:
        from transformers import pipeline  # imported only when needed
        _ai = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    max_len = 60 if ratio < 0.2 else 110 if ratio < 0.4 else 160
    out = _ai(text, max_length=max_len, min_length=20, truncation=True)
    return out[0]["summary_text"]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    ratio = float(data.get("ratio", 0.3))
    if len(split_sentences(text)) < 3:
        return jsonify(error="Paste an article or notes with at least 3 sentences."), 400
    summary = summarize_ai(text, ratio) if MODE == "ai" else summarize_simple(text, ratio)
    return jsonify(summary=summary, before=len(text.split()), after=len(summary.split()))


if __name__ == "__main__":
    # works on Render / Replit / Railway (they assign a PORT) and also locally
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)