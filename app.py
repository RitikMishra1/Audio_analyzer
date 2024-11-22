from flask import Flask, request, render_template_string, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from pydub import AudioSegment
import os
import requests
import logging
import json
from better_profanity import profanity  # Library for detecting foul language

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Flask app initialization
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Static folder for images
app.static_folder = 'static'

# HTML Template for the interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Call Sentiment Analysis</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background-image: url('{{ url_for('static', filename='background.jpg') }}');
            background-size: cover;
            background-attachment: fixed;
            background-repeat: no-repeat;
            background-position: center;
            color: #333;
            position: relative;
            font-family: 'Arial', sans-serif;
        }
        .overlay {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background-color: rgba(255, 255, 255, 0.5); /* Subtle light overlay */
            z-index: -1;
        }
        .header {
            position: fixed;
            top: 20px;
            left: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
            background-color: rgba(0, 0, 0, 0.6);
            color: white;
            padding: 10px 20px;
            border-radius: 15px;
            font-size: 1.5rem;
            box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
            z-index: 1;
        }
        .header img {
            height: 40px;
            border-radius: 50%;
        }
        .intro {
            margin-top: 100px;
            text-align: center;
            padding: 20px;
            color: white;
            text-shadow: 0 2px 4px rgba(0, 0, 0, 0.7);
        }
        .intro h1 {
            font-size: 2.5rem;
            font-weight: bold;
        }
        .intro p {
            font-size: 1.2rem;
            margin-top: 10px;
        }
        .card {
            border-radius: 15px;
            background-color: rgba(255, 255, 255, 0.9);
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-top: 20px;
        }
        .emoji {
            font-size: 5rem;
        }
        .details {
            display: none;
        }
    </style>
</head>
<body>
    <div class="overlay"></div>

    <!-- Fixed Header -->
    <div class="header">
        <img src="{{ url_for('static', filename='header.jpg') }}" alt="Logo">
        Call Sentiment Analysis
    </div>

    <div class="container mt-5">
        <!-- Intro Section -->
        <div class="intro">
            <h1>Welcome to Call Sentiment Analysis</h1>
            <p>Analyze customer interactions, detect sentiment, identify foul language, and gather actionable insights. Upload your call recording to begin!</p>
        </div>

        <div class="card shadow-lg mb-5">
            <div class="card-header bg-primary text-white text-center">
                <h3>Upload Your Audio for Analysis</h3>
            </div>
            <div class="card-body">
                <form action="/" method="post" enctype="multipart/form-data">
                    <div class="mb-3">
                        <label for="audiofile" class="form-label">Choose an Audio File</label>
                        <input type="file" name="audiofile" id="audiofile" accept="audio/*" class="form-control" required>
                    </div>
                    <div class="text-center">
                        <button type="submit" class="btn btn-success btn-lg">Analyze</button>
                    </div>
                </form>
            </div>
        </div>

        {% if analysis %}
        <div class="card shadow-lg mb-5">
            <div class="card-header bg-secondary text-white text-center">
                <h3>Analysis Summary</h3>
            </div>
            <div class="card-body text-center">
                <span class="emoji">{{ analysis['emoji'] }}</span>
                <p class="text-muted mt-3">Overall Sentiment: {{ analysis['overall_sentiment'] }}</p>
                <p class="text-muted mt-3">Detected Language: {{ analysis['language'] }}</p>
                <p class="text-muted mt-3">Foul Language Used: {{ 'Yes' if analysis['foul_language_detected'] else 'No' }}</p>
                <button class="btn btn-primary mt-3" onclick="toggleDetails()">Show More</button>
                <div class="details mt-4">
                    <h4>Speaker Identification:</h4>
                    <ul>
                        <li><strong>Speaker 1:</strong> {{ analysis['speaker_1'] }}</li>
                        <li><strong>Speaker 2:</strong> {{ analysis['speaker_2'] }}</li>
                    </ul>
                    <h4>Sentiment Analysis:</h4>
                    <h5>Speaker 1:</h5>
                    <p>Initial Sentiment: {{ analysis['speaker_1_initial_sentiment'] }}</p>
                    <p>Later Sentiment: {{ analysis['speaker_1_later_sentiment'] }}</p>
                    <h5>Speaker 2:</h5>
                    <p>Consistent Sentiment: {{ analysis['speaker_2_sentiment'] }}</p>
                    <h4>Summary:</h4>
                    <p>{{ analysis['summary'] }}</p>
                    <h4>Recommendations:</h4>
                    <p>{{ analysis['recommendations'] }}</p>
                    <h4>Call Rating:</h4>
                    <p>{{ analysis['rating'] }}/10</p>
                </div>
            </div>
        </div>
        {% endif %}
    </div>
    <script>
        function toggleDetails() {
            const details = document.querySelector('.details');
            details.style.display = details.style.display === 'block' ? 'none' : 'block';
        }
    </script>
</body>
</html>
"""



# Helper function: Compress and convert audio
def compress_audio(audio_path, compressed_path):
    try:
        audio = AudioSegment.from_file(audio_path)
        audio = audio.set_frame_rate(16000).set_channels(1)  # Mono audio, 16 kHz
        audio.export(compressed_path, format="wav")
        return compressed_path
    except Exception as e:
        logging.error(f"Error compressing audio: {e}")
        raise

# Helper function: Transcribe audio using Whisper API
from langdetect import detect  # Import language detection library

# Helper function: Transcribe audio using Whisper API
def transcribe_audio(file_path):
    try:
        with open(file_path, "rb") as audio_file:
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                files={"file": audio_file},
                data={"model": "whisper-1"}
            )
        response.raise_for_status()
        result = response.json()
        transcript = result.get("text", "")
        # Check if Whisper provides language metadata
        detected_language = result.get("language", None)
        
        if not detected_language:
            # Fallback to langdetect if language is not provided
            detected_language = detect(transcript)
        
        return transcript, detected_language
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        raise


# Helper function: Detect foul language
def detect_foul_language(transcript):
    profanity.load_censor_words()  # Load default list of profane words
    return profanity.contains_profanity(transcript)

# Helper function: Analyze transcript with ChatGPT
def analyze_with_chatgpt(transcript):
    try:
        prompt = f"""
        Please analyze the following transcript in detail:
        Transcript:
        {transcript}

        1. Identify and name the speakers based on their roles in the conversation.
        2. Analyze the sentiment for each speaker (initial and later, if applicable) and explain the changes in tone.
        3. Provide an overall summary of the call, emphasizing customer service quality and areas of improvement.
        4. Offer actionable recommendations for improving similar calls.
        5. Assign a professional rating to this call on a scale of 1 to 10.
        6. Suggest a suitable emoji for the overall sentiment.

        Output the results as a JSON object with the following keys:
        - "speaker_1": A string describing the first speaker.
        - "speaker_2": A string describing the second speaker.
        - "speaker_1_initial_sentiment": Sentiment of speaker 1 initially.
        - "speaker_1_later_sentiment": Sentiment of speaker 1 later.
        - "speaker_2_sentiment": Sentiment of speaker 2.
        - "summary": Overall summary of the conversation.
        - "recommendations": Actionable recommendations.
        - "rating": Rating of the call on a scale of 1 to 10.
        - "emoji": A representative emoji for the overall sentiment.
        """
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-4",
                "messages": [
                    {"role": "system", "content": "You are an expert call center evaluator."},
                    {"role": "user", "content": prompt}
                ],
            }
        )
        response.raise_for_status()
        return json.loads(response.json()["choices"][0]["message"]["content"])
    except Exception as e:
        logging.error(f"Error with ChatGPT API: {e}")
        raise

# Main route for uploading and analyzing audio
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        audio_file = request.files.get("audiofile")
        if not audio_file or audio_file.filename == "":
            return render_template_string(HTML_TEMPLATE, analysis=None)

        try:
            # Save and process the audio file
            filename = secure_filename(audio_file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            audio_file.save(file_path)

            compressed_path = os.path.join(UPLOAD_FOLDER, f"compressed_{filename}")
            compress_audio(file_path, compressed_path)

            # Transcribe the audio
            transcript, language = transcribe_audio(compressed_path)

            # Detect foul language
            foul_language_detected = detect_foul_language(transcript)

            # Analyze the transcript with ChatGPT
            analysis = analyze_with_chatgpt(transcript)
            analysis['language'] = language
            analysis['foul_language_detected'] = foul_language_detected

            return render_template_string(HTML_TEMPLATE, analysis=analysis)
        except Exception as e:
            logging.error(f"Error processing audio: {e}")
            return render_template_string(HTML_TEMPLATE, analysis=None)

    return render_template_string(HTML_TEMPLATE, analysis=None)

if __name__ == "__main__":
    app.run(debug=True, port=5000)