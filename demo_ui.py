import gradio as gr
import sys
from pathlib import Path

# Ensure the repository root is in the path
repo_root = Path(__file__).resolve().parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from model_b_generation.gating import moderate

def analyze_text(text: str, language: str):
    """
    Runs the full Civitas AI dual-node pipeline on the input text.
    """
    if not text.strip():
        return "Please enter some text.", ""

    # Map Gradio language choices to internal codes
    lang_map = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta"
    }
    lang_code = lang_map.get(language, "en")
    
    try:
        # Run gating logic which coordinates Model A and Model B
        result = moderate(text, lang_code)
        
        status = result.get("status", "unknown")
        
        if status == "safe":
            return "✅ **Model A classified this text as SAFE.**\n\nNo counter-narratives required.", ""
            
        elif status == "flagged":
            target = result.get("target", "Unknown")
            severity = result.get("severity", "Unknown")
            prob = result.get("p_abuse", 0.0)
            
            output_md = f"🚨 **Model A classified this text as ABUSIVE.**\n\n"
            output_md += f"- **Target Group:** {target}\n"
            output_md += f"- **Severity Score:** {severity:.2f}\n"
            output_md += f"- **Abuse Probability:** {prob:.2f}\n"
            
            alternatives = result.get("alternatives", [])
            
            if alternatives:
                alt_md = "### Model B Counter-Narrative Suggestions:\n"
                for i, alt in enumerate(alternatives, 1):
                    alt_md += f"{i}. {alt}\n"
            else:
                alt_md = "No counter-narratives could be generated."
                
            return output_md, alt_md
            
        else:
            return f"Unexpected status: {status}", ""
            
    except Exception as e:
        import traceback
        return f"❌ **Error during processing:**\n\n```\n{traceback.format_exc()}\n```", ""

with gr.Blocks(title="Civitas AI Dual-Node Moderation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🛡️ Civitas AI: Dual-Node Moderation System")
    gr.Markdown("This demo runs the **Model A** (XLM-R Detection) & **Model B** (mT5 Counter-Narrative Generation) pipeline.")
    
    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                lines=5,
                label="Input Text",
                placeholder="Enter text to moderate (e.g. 'You people are ruining this country')"
            )
            lang_dropdown = gr.Dropdown(
                choices=["English", "Hindi", "Tamil"],
                value="English",
                label="Language"
            )
            submit_btn = gr.Button("Analyze Text", variant="primary")
            
        with gr.Column(scale=3):
            model_a_output = gr.Markdown(label="Model A Output")
            model_b_output = gr.Markdown(label="Model B Output (Counter-Narratives)")

    submit_btn.click(
        fn=analyze_text,
        inputs=[input_text, lang_dropdown],
        outputs=[model_a_output, model_b_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)
