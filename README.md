# ArgueMate — Contrarian Teaching Assistant

## Setup & Run

1. Install dependencies:
   pip install -r requirements.txt

2. Set your Groq API key:
   export GROQ_API_KEY=your_key_here   # Mac/Linux
   set GROQ_API_KEY=your_key_here      # Windows

3. Run:
   python app.py

4. Open http://localhost:5000

## Deploy to Render (free)
1. Push this folder to a GitHub repo
2. Go to render.com → New Web Service → connect repo
3. Build command: pip install -r requirements.txt
4. Start command: python app.py
5. Add environment variable: GROQ_API_KEY
