import os
import gradio as gr
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key) if api_key else None

theme = gr.themes.Soft(
    primary_hue="gray",
    secondary_hue="slate",
    neutral_hue="slate",
    spacing_size="md",
    radius_size="lg",
)

custom_css = """
:root{
  --bg:#0e0f12;
  --panel:#14161b;
  --panel2:#181b21;
  --text:#e6e7ea;
  --muted:rgba(230,231,234,.55);
  --line:rgba(255,255,255,.08);
  --shadow:0 16px 40px rgba(0,0,0,.55);
}
.gradio-container{
  background: linear-gradient(180deg,#0c0d10,var(--bg)) !important;
  color:var(--text) !important;
  font-family:-apple-system,BlinkMacSystemFont,"SF Pro Text","SF Pro Display",Inter,system-ui !important;
}
.main-container{
  max-width:920px !important;
  margin:0 auto !important;
  padding:22px 16px 16px !important;
}
.brand-title{
  font-size:22px; font-weight:700; letter-spacing:-0.02em; color:var(--text);
}
.brand-subtitle{
  font-size:12.5px; color:var(--muted); margin-top:-6px;
}
.main-chat{
  background:var(--panel) !important;
  border:1px solid var(--line) !important;
  border-radius:16px !important;
  box-shadow:var(--shadow) !important;
}
.input-row{
  background:var(--panel2) !important;
  border:1px solid var(--line) !important;
  border-radius:14px !important;
  padding:10px !important;
  margin-top:12px !important;
}
textarea{
  background:transparent !important;
  color:var(--text) !important;
  font-size:15px !important;
}
textarea::placeholder{ color:rgba(230,231,234,.4) !important; }
button.primary, .gr-button-primary{
  background:#1f2229 !important;
  color:var(--text) !important;
  border-radius:12px !important;
  border:1px solid var(--line) !important;
  font-weight:600 !important;
}
.secondary-btn, .gr-button-secondary{
  background:transparent !important;
  border:1px solid var(--line) !important;
  color:var(--text) !important;
  border-radius:12px !important;
}
button:hover{ background:#262a32 !important; }
.footer-note{ color:var(--muted) !important; font-size:12px !important; }
/* Scrollbar */
*::-webkit-scrollbar{ width:10px; }
*::-webkit-scrollbar-thumb{ background:rgba(255,255,255,.12); border-radius:10px; }
*::-webkit-scrollbar-thumb:hover{ background:rgba(255,255,255,.18); }
"""

def to_plain_text(content) -> str:
    """
    Gradio 6.x may represent rich text blocks as:
      [{"type":"text","text":"..."}, ...]
    We flatten to a single string so the UI never prints raw dicts.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text", "")))
            else:
                parts.append(str(block))
        return "".join(parts)
    if isinstance(content, dict):
        # if something unexpected slips through, salvage anything readable
        return str(content.get("text", content))
    return str(content)


def normalize_messages(history):
    """
    Ensure Chatbot always receives:
      [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]
    and that all content is plain text.
    """
    if not history:
        return []

    cleaned = []

    # Already messages?
    if isinstance(history, list) and len(history) > 0 and isinstance(history[0], dict) and "role" in history[0]:
        for m in history:
            cleaned.append({
                "role": m.get("role", "assistant"),
                "content": to_plain_text(m.get("content", "")),
            })
        return cleaned

    # Pairs -> messages
    for t in history:
        if isinstance(t, (list, tuple)) and len(t) == 2:
            cleaned.append({"role": "user", "content": to_plain_text(t[0])})
            cleaned.append({"role": "assistant", "content": to_plain_text(t[1])})
    return cleaned


def chat_response(message, history):
    history = normalize_messages(history)
    message = to_plain_text(message).strip()

    if not message:
        return "", history

    if not client:
        return "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "⚠️ GROQ_API_KEY is missing. Add it in Space → Settings → Secrets."}
        ]

    system_prompt = (
        "You are an Advanced Math Tutor.\n"
        "Rules:\n"
        "1) NEVER use dollar signs ($) or LaTeX.\n"
        "2) Use plain text only (x^2, sqrt(y), (a+b)/c).\n"
        "3) Explain step by step clearly.\n"
        "4) Keep formatting readable with short steps.\n"
    )

    groq_messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        role = m.get("role")
        if role in ("user", "assistant"):
            groq_messages.append({"role": role, "content": to_plain_text(m.get("content", ""))})

    groq_messages.append({"role": "user", "content": message})

    # Immediately show Thinking...
    yield "", history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "Thinking…"}
    ]

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=groq_messages,
            stream=True,
        )

        partial = ""
        for chunk in completion:
            delta = chunk.choices[0].delta
            content = getattr(delta, "content", None)
            if not content:
                continue

            # keep everything plain text + remove any stray $
            partial += to_plain_text(content).replace("$", "")

            yield "", history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": partial}
            ]

    except Exception as e:
        yield "", history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"⚠️ Error: {to_plain_text(e)}"}
        ]


def clear_chat():
    return []


def export_history(history):
    history = normalize_messages(history)
    text = "MATHWHIZ SESSION LOG\n" + "=" * 40 + "\n\n"
    for m in history:
        role = (m.get("role") or "").upper()
        content = to_plain_text(m.get("content", ""))
        text += f"[{role}]\n{content}\n\n" + "-" * 30 + "\n\n"

    path = "math_session_log.txt"
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path



with gr.Blocks() as demo:
    with gr.Column(elem_classes="main-container"):

        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown('<div class="brand-title">MathWhiz</div>')
                gr.Markdown('<div class="brand-subtitle">Standardized Mathematical Tutor Interface</div>')
            with gr.Column(scale=2):
                export_btn = gr.DownloadButton("Export Data", variant="secondary", elem_classes="secondary-btn")

       
        chatbot = gr.Chatbot(
            height=520,
            show_label=False,
            elem_classes="main-chat",
            value=[],
        )

        with gr.Row(elem_classes="input-row"):
            msg = gr.Textbox(
                placeholder="Enter a math question… (example: solve 2x+5=17)",
                scale=10,
                container=False,
                autofocus=True,
            )
            send = gr.Button("Execute", variant="primary", scale=2)

        with gr.Row():
            gr.Markdown('<div class="footer-note">Status: Ready • Engine: Llama-3.3-70b • Gradio 6.0</div>')
            clear_btn = gr.Button("Reset Session", variant="secondary", size="sm", elem_classes="secondary-btn")

    send.click(chat_response, [msg, chatbot], [msg, chatbot])
    msg.submit(chat_response, [msg, chatbot], [msg, chatbot])
    clear_btn.click(clear_chat, None, chatbot, queue=False)
    export_btn.click(export_history, [chatbot], [export_btn])

if __name__ == "__main__":
    demo.launch(theme=theme, css=custom_css)
