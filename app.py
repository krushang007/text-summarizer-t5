#!/usr/bin/env python3
"""
T5 Dialogue Summarizer - Web Interface
"""

import json
import re
import torch
from http.server import HTTPServer, BaseHTTPRequestHandler
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Loading model from './saved_summary_model' onto {device}...")

model = AutoModelForSeq2SeqLM.from_pretrained("./saved_summary_model").to(device)
tokenizer = AutoTokenizer.from_pretrained("./saved_summary_model")
model.eval()
print("Model ready.")

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
  <title>Dialogue Summarizer</title>
  <style>
    * {
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background: #f8fafc;
      color: #0f172a;
      line-height: 1.5;
      padding: 40px 16px;
    }
    .container {
      max-width: 680px;
      margin: 0 auto;
    }
    h1 {
      font-size: 22px;
      font-weight: 600;
      color: #0f172a;
      margin-bottom: 6px;
    }
    p.desc {
      font-size: 14px;
      color: #64748b;
      margin-bottom: 24px;
    }
    .panel {
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 20px;
    }
    label {
      display: block;
      font-size: 13px;
      font-weight: 600;
      color: #334155;
      margin-bottom: 8px;
    }
    textarea {
      width: 100%;
      height: 160px;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 10px 12px;
      font-family: inherit;
      font-size: 14px;
      line-height: 1.5;
      color: #0f172a;
      resize: vertical;
    }
    textarea:focus {
      outline: none;
      border-color: #2563eb;
    }
    .actions {
      display: flex;
      justify-content: flex-end;
      gap: 10px;
      margin-top: 14px;
    }
    button {
      background: #0f172a;
      color: #ffffff;
      border: none;
      border-radius: 6px;
      padding: 8px 18px;
      font-size: 14px;
      font-weight: 500;
      cursor: pointer;
    }
    button:hover {
      background: #1e293b;
    }
    button:disabled {
      background: #94a3b8;
      cursor: not-allowed;
    }
    button.clear-btn {
      background: #ffffff;
      color: #64748b;
      border: 1px solid #cbd5e1;
    }
    button.clear-btn:hover {
      background: #f1f5f9;
      color: #0f172a;
    }
    .result-panel {
      margin-top: 20px;
      display: none;
    }
    .result-box {
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 12px 14px;
      font-size: 14px;
      color: #0f172a;
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Dialogue Summarizer</h1>
    <p class="desc">Enter a dialogue conversation to generate a concise summary.</p>

    <div class="panel">
      <label for="dialogue">Dialogue</label>
      <textarea id="dialogue" placeholder="Paste conversation here..."></textarea>

      <div class="actions">
        <button type="button" class="clear-btn" onclick="clearInput()">Clear</button>
        <button type="button" id="submitBtn" onclick="runSummarize()">Summarize</button>
      </div>

      <div class="result-panel" id="resultPanel">
        <label>Summary</label>
        <div class="result-box" id="resultText"></div>
      </div>
    </div>
  </div>

  <script>
    async function runSummarize() {
      const dialogue = document.getElementById('dialogue').value.trim();
      if (!dialogue) return;

      const btn = document.getElementById('submitBtn');
      const panel = document.getElementById('resultPanel');
      const text = document.getElementById('resultText');

      btn.disabled = true;
      btn.textContent = 'Summarizing...';

      try {
        const res = await fetch('/api/summarize', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ dialogue })
        });
        const data = await res.json();
        text.textContent = data.summary || data.error || 'No summary returned.';
        panel.style.display = 'block';
      } catch (err) {
        text.textContent = 'Error: ' + err.message;
        panel.style.display = 'block';
      } finally {
        btn.disabled = false;
        btn.textContent = 'Summarize';
      }
    }

    function clearInput() {
      document.getElementById('dialogue').value = '';
      document.getElementById('resultPanel').style.display = 'none';
      document.getElementById('dialogue').focus();
    }
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
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
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
    print(f"Serving at http://localhost:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()

if __name__ == "__main__":
    run_server()
