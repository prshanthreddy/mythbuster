"""Improved Gradio UI for MythBuster AI."""

import gradio as gr
from typing import Tuple, Optional, List
from models import ImageGenerationRequest
from services import MythBusterService
from logger import logger
from config import config

class MythBusterUI:
    """Enhanced UI for MythBuster AI."""
    
    def __init__(self):
        self.service = MythBusterService()
        self.setup_css()
    
    def setup_css(self) -> str:
        """Custom CSS for better styling."""
        return """
        <style>
        .gradio-container {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        }
        .main-header {
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .myth-input {
            font-size: 1.1em !important;
            padding: 1rem !important;
            border-radius: 10px !important;
        }
        .verdict-display {
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        .loading-indicator {
            color: #667eea;
            font-weight: bold;
            text-align: center;
        }
        .error-message {
            background-color: #ffe6e6;
            border: 1px solid #ff9999;
            color: #cc0000;
            padding: 1rem;
            border-radius: 5px;
        }
        .success-message {
            background-color: #e6ffe6;
            border: 1px solid #99ff99;
            color: #009900;
            padding: 1rem;
            border-radius: 5px;
        }
        </style>
        """
    
    def create_interface(self) -> gr.Blocks:
        """Create the main Gradio interface."""
        with gr.Blocks(
            title="🕵️ MythBuster AI - Enhanced", 
            theme=gr.themes.Soft(),
            css=self.setup_css()
        ) as interface:
            
            # Header
            gr.HTML("""
                <div class="main-header">
                    <h1>🕵️ MythBuster AI</h1>
                    <p><strong>Advanced Myth Detection & Fact-Checking Assistant</strong></p>
                    <p>Ask me about any myth, rumor, or common belief — I'll investigate it using AI and web search!</p>
                </div>
            """)
            
            # Status indicator
            status_display = gr.HTML(value="", visible=False)
            
            # Main content area
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("## 🧠 Myth Analysis")
                    chatbot = gr.Chatbot(
                        label="Conversation History",
                        height=400,
                        type="messages",
                        show_label=True,
                        container=True,
                        show_copy_button=True
                    )
                
                with gr.Column(scale=1):
                    gr.Markdown("## 🎨 Visual Results")
                    funny_image = gr.Image(
                        label="Generated Image",
                        height=300,
                        show_label=True,
                        container=True
                    )
                    
                    # Statistics panel
                    gr.Markdown("## 📊 Quick Stats")
                    stats_display = gr.HTML(value=self.get_stats_html())
            
            # Input area
            with gr.Row():
                with gr.Column(scale=4):
                    msg_input = gr.Textbox(
                        label="",
                        placeholder="e.g., 'Drinking cold water causes a sore throat' or 'Humans only use 10% of their brain'",
                        show_label=False,
                        lines=2,
                        elem_classes=["myth-input"]
                    )
                
                with gr.Column(scale=1):
                    with gr.Row():
                        submit_btn = gr.Button(
                            "🚀 Analyze Myth", 
                            variant="primary",
                            size="lg"
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear",
                            variant="secondary"
                        )
            
            # Options
            with gr.Row():
                image_enabled = gr.Checkbox(
                    label="🎨 Generate funny image",
                    value=config.has_image_generation,
                    interactive=config.has_image_generation
                )
                
                if not config.has_image_generation:
                    gr.HTML("<em>Image generation disabled (no HF token provided)</em>")
            
            # Examples
            with gr.Accordion("💡 Example Myths to Try", open=False):
                examples = gr.Examples(
                    examples=self.get_example_myths(),
                    inputs=msg_input,
                    label="Click any example to try it:"
                )
            
            # Help section
            with gr.Accordion("ℹ️ How It Works", open=False):
                gr.Markdown(self.get_help_text())
            
            # Event handlers
            def process_myth(message: str, history: List, generate_image: bool):
                return self.handle_user_message(message, history, generate_image, status_display)
            
            submit_btn.click(
                process_myth,
                inputs=[msg_input, chatbot, image_enabled],
                outputs=[msg_input, chatbot, funny_image, status_display]
            )
            
            msg_input.submit(
                process_myth,
                inputs=[msg_input, chatbot, image_enabled],
                outputs=[msg_input, chatbot, funny_image, status_display]
            )
            
            clear_btn.click(
                lambda: ([], None, ""),
                outputs=[chatbot, funny_image, status_display]
            )
        
        return interface
    
    def handle_user_message(
        self, 
        message: str, 
        history: List, 
        generate_image: bool,
        status_display
    ) -> Tuple[str, List, Optional[str], str]:
        """Handle user message and return updated components."""
        
        if not message.strip():
            return "", history, None, ""
        
        logger.info(f"Processing user message: {message}")
        
        # Initialize history if None
        if history is None:
            history = []
        
        # Show loading status
        status_html = '<div class="loading-indicator">🔍 Analyzing myth... Please wait.</div>'
        
        try:
            # Add user message to history
            history.append({"role": "user", "content": message})
            
            # Analyze the myth
            result = self.service.analyze_claim(message)
            
            # Format response
            response = result.format_response()
            history.append({"role": "assistant", "content": response})
            
            # Generate image if requested
            image_path = None
            if generate_image and config.has_image_generation:
                try:
                    image_request = ImageGenerationRequest(myth=message, enabled=True)
                    image_path = self.service.generate_funny_image(image_request)
                except Exception as e:
                    logger.error(f"Image generation failed: {e}")
            
            # Success status
            status_html = '<div class="success-message">✅ Analysis complete!</div>'
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            error_response = f"❌ **Error**: {str(e)}\n\nPlease try again or rephrase your question."
            history.append({"role": "assistant", "content": error_response})
            status_html = '<div class="error-message">❌ An error occurred. Please try again.</div>'
            image_path = None
        
        return "", history, image_path, status_html
    
    def get_example_myths(self) -> List[List[str]]:
        """Get example myths for the interface."""
        return [
            ["Drinking cold water causes a sore throat"],
            ["Humans only use 10% of their brain"],
            ["Goldfish have a 3-second memory"],
            ["You can see the Great Wall of China from space"],
            ["Eating carrots improves your eyesight"],
            ["Lightning never strikes the same place twice"],
            ["Cracking your knuckles causes arthritis"],
            ["Bulls are enraged by the color red"],
            ["We only have five senses"],
            ["Hair and nails continue growing after death"]
        ]
    
    def get_help_text(self) -> str:
        """Get help text explaining how the system works."""
        return """
        ### 🔍 How MythBuster AI Works:
        
        1. **📝 Enter Your Claim**: Type any myth, rumor, or belief you want fact-checked
        
        2. **🧠 Memory Search**: I first check my knowledge base for similar claims I've analyzed before
        
        3. **🌐 Web Research**: If needed, I search the web for current information and evidence
        
        4. **🤖 AI Analysis**: Using advanced language models, I analyze the evidence and provide a verdict
        
        5. **📊 Verdict Categories**:
           - ✅ **CONFIRMED**: The claim is supported by evidence
           - ❓ **PLAUSIBLE**: The claim might be true but lacks definitive proof
           - ❌ **BUSTED**: The claim is false or misleading
        
        6. **🎨 Visual Fun**: Optionally generate humorous images related to the myth
        
        ### 💡 Tips for Best Results:
        - Be specific and clear in your claims
        - Avoid overly complex or multi-part questions
        - Try rephrasing if you get an unclear answer
        
        ### 🔒 Privacy & Safety:
        - Your queries are processed securely
        - No personal data is stored permanently
        - All interactions are logged for quality improvement
        """
    
    def get_stats_html(self) -> str:
        """Get HTML for statistics display."""
        return """
        <div style="text-align: center; padding: 1rem; background: #f8f9fa; border-radius: 8px;">
            <p><strong>🎯 Accuracy Focus</strong></p>
            <p>Powered by LLaMA 3.1</p>
            <p>📚 Vector Knowledge Base</p>
            <p>🌐 Real-time Web Search</p>
        </div>
        """
    
    def launch(self):
        """Launch the Gradio interface."""
        interface = self.create_interface()
        
        logger.info("Launching MythBuster AI interface...")
        interface.launch(
            share=config.share_gradio,
            server_port=config.server_port,
            show_error=True
        )