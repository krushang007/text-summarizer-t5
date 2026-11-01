#!/usr/bin/env python3
"""
Interactive Web Server for T5 Dialogue Summarizer
Serves an intuitive UI at http://localhost:8080
"""

import json
import re
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# 1. Load fine-tuned model
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Loading fine-tuned model from './saved_summary_model' onto {device}...")

model = AutoModelForSeq2SeqLM.from_pretrained("./saved_summary_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("./saved_summary_model")
model.eval()
print("Model loaded successfully and ready for inference!")

def clean_text(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def summarize(dialogue: str, max_length=128, num_beams=4) -> str:
    cleaned = clean_text(dialogue)
    prompt = "summarize: " + cleaned
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=max_length,
            num_beams=num_beams,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>T5 Dialogue Summarizer</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #090d16;
      --card: #111726;
      --border: #1e293b;
      --text: #f1f5f9;
      --text-muted: #94a3b8;
      --accent: #3b82f6;
      --accent-hover: #2563eb;
      --accent-glow: rgba(59, 130, 246, 0.15);
      --success: #10b981;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 32px 16px;
    }
    .container {
      width: 100%;
      max-width: 900px;
      display: flex;
      flex-direction: column;
      gap: 24px;
    }
    header {
      text-align: center;
      margin-bottom: 8px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      background: var(--accent-glow);
      border: 1px solid rgba(59, 130, 246, 0.3);
      color: #60a5fa;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 500;
      margin-bottom: 12px;
    }
    .status-dot {
      width: 8px;
      height: 8px;
      background: var(--success);
      border-radius: 50%;
      box-shadow: 0 0 8px var(--success);
    }
    h1 {
      font-size: 32px;
      font-weight: 700;
      letter-spacing: -0.02em;
      margin-bottom: 8px;
      background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    p.subtitle {
      color: var(--text-muted);
      font-size: 15px;
    }
    .samples-bar {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      align-items: center;
    }
    .samples-label {
      font-size: 13px;
      color: var(--text-muted);
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .chip {
      background: #1e293b;
      border: 1px solid #334155;
      color: #e2e8f0;
      font-size: 13px;
      padding: 6px 14px;
      border-radius: 8px;
      cursor: pointer;
      transition: all 0.15s ease;
    }
    .chip:hover {
      background: #334155;
      border-color: #475569;
      transform: translateY(-1px);
    }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.5);
    }
    label {
      display: block;
      font-size: 14px;
      font-weight: 600;
      margin-bottom: 8px;
      color: #e2e8f0;
    }
    textarea {
      width: 100%;
      height: 180px;
      background: #0b0f19;
      border: 1px solid #1e293b;
      border-radius: 12px;
      color: var(--text);
      padding: 14px;
      font-family: inherit;
      font-size: 14px;
      line-height: 1.5;
      resize: vertical;
      transition: border-color 0.2s ease;
    }
    textarea:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px var(--accent-glow);
    }
    .btn {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      margin-top: 14px;
      padding: 12px 24px;
      background: var(--accent);
      color: white;
      border: none;
      border-radius: 10px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.2s ease;
    }
    .btn:hover {
      background: var(--accent-hover);
      box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    .btn:disabled {
      opacity: 0.6;
      cursor: not-allowed;
    }
    .output-section {
      margin-top: 24px;
      display: none;
    }
    .output-box {
      background: #0b0f19;
      border: 1px solid #1e293b;
      border-left: 4px solid var(--accent);
      border-radius: 12px;
      padding: 18px;
      font-size: 15px;
      line-height: 1.6;
      color: #f8fafc;
    }
    .output-box.success {
      border-left-color: var(--success);
    }
    .spinner {
      width: 18px;
      height: 18px;
      border: 2px solid rgba(255,255,255,0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      display: none;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="badge">
        <div class="status-dot"></div>
        <span>Model Fine-Tuned & Ready</span>
      </div>
      <h1>Dialogue Text Summarizer</h1>
      <p class="subtitle">Fine-tuned Google T5 on SAMSum (Apple Silicon MPS Acceleration)</p>
    </header>

    <div class="samples-bar">
      <span class="samples-label">Quick Samples:</span>
      <button class="chip" onclick="loadSample(1)">Tech Meeting</button>
      <button class="chip" onclick="loadSample(2)">Weekend Trip</button>
      <button class="chip" onclick="loadSample(3)">Shopping Advice</button>
      <button class="chip" onclick="loadSample(4)">Customer Order</button>
    </div>

    <div class="card">
      <label for="dialogueInput">Dialogue Conversation</label>
      <textarea id="dialogueInput" placeholder="Enter or paste a multi-turn conversation here (e.g. Alice: Hello! Bob: Hey!)..."></textarea>
      
      <button id="submitBtn" class="btn" onclick="generateSummary()">
        <span class="spinner" id="btnSpinner"></span>
        <span id="btnText">Generate Summary</span>
      </button>

      <div class="output-section" id="outputSection">
        <label>Generated Summary</label>
        <div class="output-box success" id="summaryOutput"></div>
      </div>
    </div>
  </div>

  <script>
    const samples = {
      1: `Alex: Team, how is the Q4 backend deployment going?
Rachel: All database migrations finished smoothly on the replica.
Marcus: The new caching layer is deployed, API response times dropped by 40%.
Alex: That is great progress. Are there any blockers before tomorrow's general release?
Rachel: None from my side, test coverage is 94%.
Alex: Awesome, let's ship tomorrow at 10 AM.`,

      2: `Emma: Hey, are we still driving up to the cabin this Saturday?
Liam: Yes! I reserved the rental SUV for 8 AM.
Sophia: What should I bring?
Liam: Bring snacks and warm jackets, it will be chilly by the lake.
Emma: I'll make sandwiches and buy coffee.
Sophia: Sounds good! See you both on Saturday morning.`,

      3: `Hannah: Hey, do you have Betty's phone number?
Amanda: Lemme check my contact list.
Hannah: Thanks, appreciate it!
Amanda: Sorry, can't find it. You can ask Larry, he called her last weekend.
Hannah: I don't know him well though.
Amanda: Don't be shy, he's very friendly! Just send him a quick text.
Hannah: Alright, I'll text him. Thanks!`,

      4: `Customer: Hi, I ordered a mechanical keyboard three days ago (Order #8492) and haven't received tracking yet.
Support: Hello! Let me look up order #8492 in our system right now.
Support: Thanks for waiting. The package was dispatched yesterday via FedEx.
Customer: Do you have the tracking number?
Support: Yes, it is FX-992144-US. Delivery is scheduled for tomorrow afternoon.
Customer: Perfect, thank you for your quick help!`
    };

    function loadSample(num) {
      document.getElementById('dialogueInput').value = samples[num];
      document.getElementById('outputSection').style.display = 'none';
    }

    async function generateSummary() {
      const dialogue = document.getElementById('dialogueInput').value.trim();
      if (!dialogue) {
        alert('Please enter or select a dialogue first!');
        return;
      }

      const btn = document.getElementById('submitBtn');
      const btnText = document.getElementById('btnText');
      const btnSpinner = document.getElementById('btnSpinner');
      const outputSection = document.getElementById('outputSection');
      const summaryOutput = document.getElementById('summaryOutput');

      btn.disabled = true;
      btnSpinner.style.display = 'inline-block';
      btnText.textContent = 'Summarizing...';

      try {
        const response = await fetch('/api/summarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dialogue })
        });

        const data = await response.json();
        if (data.summary) {
          summaryOutput.textContent = data.summary;
          outputSection.style.display = 'block';
          outputSection.scrollIntoView({ behavior: 'smooth' });
        } else {
          summaryOutput.textContent = 'Error: ' + (data.error || 'Failed to generate summary.');
          outputSection.style.display = 'block';
        }
      } catch (err) {
        summaryOutput.textContent = 'Error communicating with server: ' + err.message;
        outputSection.style.display = 'block';
      } finally {
        btn.disabled = false;
        btnSpinner.style.display = 'none';
        btnText.textContent = 'Generate Summary';
      }
    }

    // Auto-load sample 1 on initial visit
    window.onload = () => loadSample(1);
  </script>
</body>
</html>
"""

class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/summarize":
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode("utf-8"))
                dialogue = data.get("dialogue", "")
                if not dialogue:
                    self.send_response(400)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps({"error": "Empty dialogue"}).encode("utf-8"))
                    return

                res_summary = summarize(dialogue)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"summary": res_summary}).encode("utf-8"))
            except Exception as e:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

def run_server(port=8080):
    server = HTTPServer(("0.0.0.0", port), RequestHandler)
    print(f"\n=======================================================")
    print(f" Web UI running at: http://localhost:{port}")
    print(f" Press Ctrl+C to stop the server.")
    print(f"=======================================================\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server...")
        server.server_close()

if __name__ == "__main__":
    run_server()

