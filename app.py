from flask import Flask, render_template, request, send_from_directory
from youtube_transcript_api import YouTubeTranscriptApi
import urllib.parse as urlparse
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
from string import punctuation
from heapq import nlargest
from transformers import pipeline
from deep_translator import GoogleTranslator
import gtts
import os

LANGUAGE_MAP = {
    "Afrikaans": "af", "Albanian": "sq", "Amharic": "am", "Arabic": "ar", "Armenian": "hy",
    "Azerbaijani": "az", "Basque": "eu", "Belarusian": "be", "Bengali": "bn", "Bosnian": "bs",
    "Bulgarian": "bg", "Catalan": "ca", "Cebuano": "ceb", "Chichewa": "ny", "Chinese (Simplified)": "zh-cn",
    "Chinese (Traditional)": "zh-tw", "Corsican": "co", "Croatian": "hr", "Czech": "cs", "Danish": "da",
    "Dutch": "nl", "English": "en", "Esperanto": "eo", "Estonian": "et", "Filipino": "tl", "Finnish": "fi",
    "French": "fr", "Frisian": "fy", "Galician": "gl", "Georgian": "ka", "German": "de", "Greek": "el",
    "Gujarati": "gu", "Haitian Creole": "ht", "Hausa": "ha", "Hawaiian": "haw", "Hebrew": "he",
    "Hindi": "hi", "Hmong": "hmn", "Hungarian": "hu", "Icelandic": "is", "Igbo": "ig", "Indonesian": "id",
    "Irish": "ga", "Italian": "it", "Japanese": "ja", "Javanese": "jw", "Kannada": "kn", "Kazakh": "kk",
    "Khmer": "km", "Korean": "ko", "Kurdish (Kurmanji)": "ku", "Kyrgyz": "ky", "Lao": "lo", "Latin": "la",
    "Latvian": "lv", "Lithuanian": "lt", "Luxembourgish": "lb", "Macedonian": "mk", "Malagasy": "mg",
    "Malay": "ms", "Malayalam": "ml", "Maltese": "mt", "Maori": "mi", "Marathi": "mr", "Mongolian": "mn",
    "Myanmar (Burmese)": "my", "Nepali": "ne", "Norwegian": "no", "Odia": "or", "Pashto": "ps",
    "Persian": "fa", "Polish": "pl", "Portuguese": "pt", "Punjabi": "pa", "Romanian": "ro", "Russian": "ru",
    "Samoan": "sm", "Scots Gaelic": "gd", "Serbian": "sr", "Sesotho": "st", "Shona": "sn", "Sindhi": "sd",
    "Sinhala": "si", "Slovak": "sk", "Slovenian": "sl", "Somali": "so", "Spanish": "es", "Sundanese": "su",
    "Swahili": "sw", "Swedish": "sv", "Tajik": "tg", "Tamil": "ta", "Telugu": "te", "Thai": "th",
    "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur", "Uyghur": "ug", "Uzbek": "uz", "Vietnamese": "vi",
    "Welsh": "cy", "Xhosa": "xh", "Yiddish": "yi", "Yoruba": "yo", "Zulu": "zu"
}


nltk.download('punkt')
nltk.download('stopwords')
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/audio"

if not os.path.exists(app.config["UPLOAD_FOLDER"]):
    os.makedirs(app.config["UPLOAD_FOLDER"])

@app.route("/", methods=["GET", "POST"])
def index():
    return render_template("index.html")

@app.route("/process", methods=["POST"])
def process():
    url = request.form["video_url"]
    percent = int(request.form["summary_percent"])
    language_name = request.form["lang_code"]
    lang_code = LANGUAGE_MAP.get(language_name, "en")

    summarizer_type = request.form.get("summarizer_type")
    tts_option = request.form.get("tts_option")

    parsed = urlparse.urlparse(url)
    video_id = urlparse.parse_qs(parsed.query).get("v", [None])[0]

    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        lines = [line['text'].replace("\n", " ") for line in transcript]
        paragraph = " ".join(lines)

        # --- NLTK summarization ---
        stops = set(stopwords.words("english"))
        words = word_tokenize(paragraph)
        wordfreq = {}
        for word in words:
            word = word.lower()
            if word not in stops and word not in punctuation:
                wordfreq[word] = wordfreq.get(word, 0) + 1

        sentences = sent_tokenize(paragraph)
        sentfreq = {}
        for sentence in sentences:
            for word, freq in wordfreq.items():
                if word in sentence.lower():
                    sentfreq[sentence] = sentfreq.get(sentence, 0) + freq

        average = sum(sentfreq.values()) / len(sentfreq)
        nltk_summary = " ".join([s for s in sentences if sentfreq.get(s, 0) > (1.5 * average)])

        # --- spaCy summarization ---
        doc = nlp(paragraph)
        word_freq = {}
        for word in doc:
            if word.text.lower() not in stopwords.words("english") and word.text.lower() not in punctuation:
                word_freq[word.text.lower()] = word_freq.get(word.text.lower(), 0) + 1
        max_freq = max(word_freq.values())
        for word in word_freq:
            word_freq[word] = word_freq[word] / max_freq

        spacy_scores = {}
        for sent in doc.sents:
            for word in sent:
                if word.text.lower() in word_freq:
                    spacy_scores[sent] = spacy_scores.get(sent, 0) + word_freq[word.text.lower()]

        spacy_summary = " ".join([sent.text for sent in nlargest(int(len(list(doc.sents)) * (percent / 100)), spacy_scores, key=spacy_scores.get)])

        # --- Transformer summarizer ---
        transformer_summary = ""
        if summarizer_type == "transformer":
            hf_summarizer = pipeline("summarization")
            chunks = [paragraph[i:i+1000] for i in range(0, len(paragraph), 1000)]
            transformer_summary = " ".join([hf_summarizer(chunk)[0]["summary_text"] for chunk in chunks])

        # --- Final summary selection ---
        final_summary = {
            "nltk": nltk_summary,
            "spacy": spacy_summary,
            "transformer": transformer_summary
        }.get(summarizer_type, nltk_summary)

        # --- Translation ---
        translated_summary = GoogleTranslator(source='auto', target=lang_code).translate(final_summary)

        # --- Text to Speech ---
        audio_files = []
        if tts_option in ['extracted', 'both']:
            speech = gtts.gTTS(final_summary)
            filename = "extracted_summary.mp3"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            speech.save(filepath)
            audio_files.append(filename)

        if tts_option in ['translated', 'both']:
            translated_speech = gtts.gTTS(translated_summary)
            filename = "translated_summary.mp3"
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            translated_speech.save(filepath)
            audio_files.append(filename)

        return render_template("result.html",
                               summary=final_summary,
                               translated=translated_summary,
                               audio_files=audio_files)

    except Exception as e:
        return f"<h3>Error: {str(e)}</h3>"

@app.route("/static/audio/<filename>")
def download_audio(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
