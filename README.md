📘 Multilingual Summarizer and Audio Generator for YouTube Videos
A web application that extracts transcripts from YouTube videos, summarizes them using NLP, translates the summary into different languages, and converts it into audio using text-to-speech.

🚀 Features
🎥 Extracts transcript from YouTube videos

✂️ Summarizes content using NLP libraries (spaCy, NLTK, or Transformers)

🌐 Translates summaries into multiple languages

🔊 Converts translated summaries to audio using gTTS

🌍 User-friendly web interface built with Flask

🛠️ Tech Stack
Frontend: HTML, CSS, JavaScript (optional if frontend is simple)

Backend: Python, Flask

NLP: NLTK, spaCy, or Hugging Face Transformers

Translation: googletrans or other translation APIs

TTS: gTTS (Google Text-to-Speech)

Others: pytube, youtube-transcript-api


🔧 Installation

git clone https://github.com/yourusername/yt-summarizer-translator.git
cd yt-summarizer-translator
pip install -r requirements.txt
python app.py


🧪 How It Works
Enter a YouTube video link.

The app fetches the transcript using youtube-transcript-api.

The text is summarized using your chosen NLP tool.

The summary is translated into your selected language.

The translated text is converted into audio using gTTS.



