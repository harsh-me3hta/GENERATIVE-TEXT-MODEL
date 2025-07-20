# app.py - Flask Backend for Ollama Text Generator

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import requests
import json
import time
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

class OllamaTextGenerator:
    def __init__(self, model_name="llama2"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"
        
    def check_ollama_status(self):
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def get_available_models(self):
        """Get list of available Ollama models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model['name'].split(':')[0] for model in models_data.get('models', [])]
            return []
        except requests.exceptions.RequestException:
            return []
    
    def pull_model(self, model_name):
        """Pull a model if it's not available"""
        try:
            response = requests.post(
                f"{self.base_url}/api/pull",
                json={"name": model_name},
                timeout=300  # 5 minutes timeout for model download
            )
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False
    
    def generate_paragraph(self, topic, style="informative", length="medium"):
        """Generate a paragraph on the given topic"""
        if not self.check_ollama_status():
            return {
                "success": False,
                "error": "Ollama service is not running. Please start Ollama first.",
                "text": ""
            }
        
        # Customize prompt based on style and length
        style_prompts = {
            "informative": "Write an informative and educational paragraph about",
            "creative": "Write a creative and engaging paragraph about", 
            "technical": "Write a detailed technical explanation about",
            "simple": "Write a simple, easy-to-understand paragraph about"
        }
        
        length_instructions = {
            "short": "Keep it concise, around 3-4 sentences.",
            "medium": "Write a well-developed paragraph of 5-7 sentences.",
            "long": "Write a comprehensive paragraph of 8-10 sentences with detailed information."
        }
        
        style_instruction = style_prompts.get(style, style_prompts["informative"])
        length_instruction = length_instructions.get(length, length_instructions["medium"])
        
        prompt = f"""{style_instruction} "{topic}". 

{length_instruction}

Make sure the paragraph:
- Is factually accurate and well-researched
- Flows naturally from one sentence to the next
- Provides valuable insights about the topic
- Maintains a consistent tone throughout
- Ends with a strong concluding thought

Topic: {topic}

Paragraph:"""

        try:
            # Set token limits based on length
            token_limits = {"short": 100, "medium": 200, "long": 300}
            max_tokens = token_limits.get(length, 200)
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "top_k": 40,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=60  # 1 minute timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', '').strip()
                
                # Clean up the generated text
                if generated_text:
                    # Remove any prompt repetition
                    if topic.lower() in generated_text.lower()[:50]:
                        # Find where the actual content starts
                        sentences = generated_text.split('.')
                        cleaned_sentences = []
                        for sentence in sentences:
                            if len(sentence.strip()) > 10:  # Avoid very short fragments
                                cleaned_sentences.append(sentence.strip())
                        generated_text = '. '.join(cleaned_sentences)
                        if not generated_text.endswith('.'):
                            generated_text += '.'
                
                return {
                    "success": True,
                    "text": generated_text,
                    "model": self.model_name,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                return {
                    "success": False,
                    "error": f"Ollama API error: {response.status_code}",
                    "text": ""
                }
                
        except requests.exceptions.Timeout:
            return {
                "success": False,
                "error": "Request timed out. The model might be taking too long to respond.",
                "text": ""
            }
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Connection error: {str(e)}",
                "text": ""
            }

# Initialize the generator
ollama_gen = OllamaTextGenerator()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/status')
def check_status():
    """Check Ollama status and available models"""
    status = ollama_gen.check_ollama_status()
    models = ollama_gen.get_available_models() if status else []
    
    return jsonify({
        "ollama_running": status,
        "available_models": models,
        "current_model": ollama_gen.model_name
    })

@app.route('/api/generate', methods=['POST'])
def generate_text():
    """Generate text based on topic and parameters"""
    data = request.get_json()
    
    topic = data.get('topic', '').strip()
    style = data.get('style', 'informative')
    length = data.get('length', 'medium')
    model = data.get('model', 'llama2')
    
    if not topic:
        return jsonify({
            "success": False,
            "error": "Please provide a topic"
        })
    
    # Update model if different
    if model != ollama_gen.model_name:
        ollama_gen.model_name = model
    
    # Generate the paragraph
    result = ollama_gen.generate_paragraph(topic, style, length)
    
    return jsonify(result)

@app.route('/api/pull-model', methods=['POST'])
def pull_model():
    """Pull a new model"""
    data = request.get_json()
    model_name = data.get('model_name', '')
    
    if not model_name:
        return jsonify({"success": False, "error": "Model name required"})
    
    success = ollama_gen.pull_model(model_name)
    return jsonify({"success": success})

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
        print("📁 Created 'templates' directory")
    
    print("🚀 Starting Ollama Text Generator Web Interface...")
    print("📍 Server will be available at: http://localhost:5000")
    print("🔧 Make sure Ollama is running on port 11434")
    print("📄 Make sure index.html is in the 'templates' folder")
    print("-" * 50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
