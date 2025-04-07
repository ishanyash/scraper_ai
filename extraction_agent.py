import requests
import logging
import json
from typing import Dict, List, Any, Optional
from multi_agent_core import Agent
import time
import re

class ExtractionAgent(Agent):
    """Agent responsible for extracting data from public data sources"""
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__("ExtractionAgent", model)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "application/json"
        }
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from public data sources"""
        query = input_data.get("interpreted_query", input_data.get("query", ""))
        instructions = input_data.get("instructions", "")
        required_fields = input_data.get("required_fields", [])
        
        self.logger.info(f"Starting data extraction for query: {query}")
        
        # Get aggregated data from multiple public sources
        extracted_data = []
        
        # Method 1: Public dataset repositories
        public_dataset_tools = self._get_data_from_public_datasets(query, required_fields)
        if public_dataset_tools:
            self.logger.info(f"Found {len(public_dataset_tools)} tools from public datasets")
            extracted_data.extend(public_dataset_tools)
        
        # Method 2: GitHub repositories with AI tool listings
        github_tools = self._get_data_from_github(query, required_fields)
        if github_tools:
            self.logger.info(f"Found {len(github_tools)} tools from GitHub repos")
            extracted_data.extend(github_tools)
        
        # Method 3: Open APIs with AI tool data
        api_tools = self._get_data_from_open_apis(query, required_fields)
        if api_tools:
            self.logger.info(f"Found {len(api_tools)} tools from open APIs")
            extracted_data.extend(api_tools)
        
        # If no data could be found through public sources, use AI model
        if not extracted_data:
            self.logger.info("No data found from public sources, using AI knowledge")
            ai_generated_tools = self._generate_ai_tool_data(query, required_fields)
            extracted_data.extend(ai_generated_tools)
            self.logger.info(f"Generated {len(ai_generated_tools)} tools using AI knowledge")
        
        # Remove duplicates based on tool names
        unique_data = []
        seen_names = set()
        for item in extracted_data:
            name = item.get("name", "").lower()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_data.append(item)
        
        result = {
            "extracted_data": unique_data,
            "total_items": len(unique_data),
            "sources_processed": 3,  # Public datasets, GitHub, APIs
            "extraction_success_rate": 100 if unique_data else 0
        }
        
        self.logger.info(f"Extraction completed: {len(unique_data)} unique AI tools extracted")
        return result
    
    def _get_data_from_public_datasets(self, query: str, required_fields: List[str]) -> List[Dict[str, Any]]:
        """Get AI tool data from public datasets (e.g., Kaggle)"""
        tools = []
        
        # Option 1: We could use the Kaggle API, but it requires authentication
        # Instead, let's use a simplified approach with pre-defined data URLs
        
        # Example: Access a public GitHub Gist with AI tool data
        try:
            # This is a sample URL - replace with an actual public dataset URL
            url = "https://raw.githubusercontent.com/public-apis/public-apis/master/categories/ai.json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # Parse the API list and convert to our format
                for api in data.get("entries", []):
                    tool = {
                        "name": api.get("API", ""),
                        "description": api.get("Description", ""),
                        "url": api.get("Link", ""),
                        "features": [api.get("Description", "")],
                        "pricing": "Free" if api.get("Auth") == "" else "Requires API Key",
                        "category": "AI API",
                        "source": "Public APIs GitHub Repository"
                    }
                    tools.append(tool)
            
        except Exception as e:
            self.logger.error(f"Error accessing public dataset: {str(e)}")
        
        # Since many public datasets require authentication, we'll create a backup
        # that contains common AI tools if we couldn't get any data
        if not tools:
            tools = self._get_fallback_ai_tools_data()
            
        return tools
    
    def _get_data_from_github(self, query: str, required_fields: List[str]) -> List[Dict[str, Any]]:
        """Get AI tool data from GitHub repositories with curated lists"""
        tools = []
        
        # Note: Using GitHub API would require authentication
        # For simplicity, we'll use pre-processed data from common awesome-lists

        # Example: Awesome AI Tools List
        try:
            # This URL would be a raw content URL from a GitHub repo with AI tools data
            url = "https://raw.githubusercontent.com/steven2358/awesome-generative-ai/main/README.md"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                # Parse markdown content to extract tool information
                content = response.text
                
                # Simple pattern matching for markdown links in sections
                # This is a simplified approach - a proper parser would be better
                tool_pattern = r'\* \[(.*?)\]\((https?://[^\s)]+)\) - (.*?)(?:\.|$)'
                matches = re.findall(tool_pattern, content)
                
                for match in matches:
                    if len(match) >= 3:
                        tool_name, tool_url, tool_description = match
                        
                        # Only include relevant AI tools based on query
                        if any(keyword in tool_name.lower() or keyword in tool_description.lower() 
                               for keyword in ["ai", "artificial intelligence", "machine learning", "ml"]):
                            tool = {
                                "name": tool_name,
                                "url": tool_url,
                                "description": tool_description,
                                "features": [tool_description],
                                "pricing": "Unknown",  # Pricing usually not specified in awesome lists
                                "category": "AI Tool",
                                "source": "GitHub Awesome List"
                            }
                            tools.append(tool)
            
        except Exception as e:
            self.logger.error(f"Error accessing GitHub data: {str(e)}")
        
        return tools
    
    def _get_data_from_open_apis(self, query: str, required_fields: List[str]) -> List[Dict[str, Any]]:
        """Get AI tool data from open APIs"""
        tools = []
        
        # Option: HuggingFace Models API
        try:
            # HuggingFace API for models
            url = "https://huggingface.co/api/models?sort=downloads&direction=-1&limit=100"
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                for model in data:
                    if any(tag in model.get("tags", []) for tag in ["ai", "nlp", "vision", "ml", "text-generation"]):
                        tool = {
                            "name": model.get("modelId", ""),
                            "url": f"https://huggingface.co/{model.get('modelId', '')}",
                            "description": model.get("description", ""),
                            "features": model.get("tags", []),
                            "pricing": "Open Source",
                            "category": "AI Model",
                            "downloads": model.get("downloads", 0),
                            "source": "HuggingFace Hub"
                        }
                        tools.append(tool)
            
        except Exception as e:
            self.logger.error(f"Error accessing open APIs: {str(e)}")
        
        return tools
    
    def _get_fallback_ai_tools_data(self) -> List[Dict[str, Any]]:
        """Provide a fallback list of common AI tools"""
        return [
            {
                "name": "ChatGPT",
                "url": "https://chat.openai.com/",
                "description": "Conversational AI assistant that can understand and generate human-like text",
                "features": ["Text generation", "Conversational AI", "Content creation", "Programming assistance", "Knowledge QA"],
                "pricing": "Free tier with limitations, Plus subscription at $20/month, Team and Enterprise options available",
                "category": "Conversational AI",
                "source": "Fallback Data"
            },
            {
                "name": "DALL-E",
                "url": "https://openai.com/dall-e/",
                "description": "AI system that creates realistic images and art from natural language descriptions",
                "features": ["Image generation", "Text-to-image", "Creative design", "Variations", "Inpainting"],
                "pricing": "Available through OpenAI API with credit system pricing",
                "category": "Image Generation",
                "source": "Fallback Data"
            },
            {
                "name": "Midjourney",
                "url": "https://www.midjourney.com/",
                "description": "AI art generator that creates images from text descriptions",
                "features": ["Image generation", "Text-to-image", "Art creation", "High-quality renders"],
                "pricing": "Basic plan at $10/month, Standard at $30/month, Pro at $60/month",
                "category": "Image Generation",
                "source": "Fallback Data"
            },
            {
                "name": "Stable Diffusion",
                "url": "https://stability.ai/",
                "description": "Open-source text-to-image model that generates detailed images based on text descriptions",
                "features": ["Text-to-image", "Open source", "Local installation option", "Community extensions"],
                "pricing": "Free (self-hosted), Commercial API through DreamStudio",
                "category": "Image Generation",
                "source": "Fallback Data"
            },
            {
                "name": "GitHub Copilot",
                "url": "https://github.com/features/copilot",
                "description": "AI pair programmer that offers code suggestions in real-time",
                "features": ["Code completion", "Code generation", "IDE integration", "Multiple language support"],
                "pricing": "$10/month for individuals, $19/user/month for businesses",
                "category": "Coding Assistant",
                "source": "Fallback Data"
            },
            {
                "name": "Jasper",
                "url": "https://www.jasper.ai/",
                "description": "AI writing assistant that helps create content for marketing, blogs, and social media",
                "features": ["Content writing", "Marketing copy", "Template library", "Multi-language support"],
                "pricing": "Creator plan at $49/month, Teams at $125/month",
                "category": "Content Writing",
                "source": "Fallback Data"
            },
            {
                "name": "Claude",
                "url": "https://claude.ai/",
                "description": "Conversational AI assistant by Anthropic designed to be helpful, harmless, and honest",
                "features": ["Text generation", "Document analysis", "Conversational AI", "Content creation"],
                "pricing": "Free tier with limitations, Claude Pro at $20/month",
                "category": "Conversational AI",
                "source": "Fallback Data"
            },
            {
                "name": "Runway",
                "url": "https://runwayml.com/",
                "description": "Creative tools powered by AI for video editing and generation",
                "features": ["Video generation", "Video editing", "Text-to-video", "Green screen"],
                "pricing": "Standard at $12/month, Pro at $28/month, Unlimited at $76/month",
                "category": "Video Generation",
                "source": "Fallback Data"
            },
            {
                "name": "Otter.ai",
                "url": "https://otter.ai/",
                "description": "AI-powered transcription and note-taking service",
                "features": ["Voice transcription", "Meeting notes", "Audio recording", "Keyword search"],
                "pricing": "Free tier, Pro at $8.33/month, Business at $20/month",
                "category": "Transcription",
                "source": "Fallback Data"
            },
            {
                "name": "Grammarly",
                "url": "https://www.grammarly.com/",
                "description": "AI writing assistant that checks grammar, spelling, and style",
                "features": ["Grammar checking", "Style suggestions", "Tone detection", "Plagiarism detection"],
                "pricing": "Free tier, Premium at $12/month, Business at $15/user/month",
                "category": "Writing Assistant",
                "source": "Fallback Data"
            }
        ]
    
    def _generate_ai_tool_data(self, query: str, required_fields: List[str], count: int = 50) -> List[Dict[str, Any]]:
        """Generate AI tool information using the model's knowledge"""
        system_prompt = """
        You are an expert on AI tools and software. Provide accurate, factual information 
        about the most popular and useful AI tools currently available.
        
        For each tool, include:
        1. Name
        2. Features (main capabilities)
        3. Pricing information (if known)
        4. URL (official website)
        5. Category (e.g., text generation, image generation)
        
        Return your response as a JSON array of objects, with each object representing one AI tool.
        """
        
        user_prompt = f"""
        Please provide detailed information about the top {count} AI tools related to this query:
        "{query}"
        
        Required fields: {', '.join(required_fields)}
        
        For each tool, include at minimum: name, features, pricing, and URL.
        Ensure the information is accurate, up-to-date, and comprehensive.
        Format your response as a JSON array of tool objects.
        """
        
        response = self.call_ai(system_prompt, user_prompt, json_response=True)
        
        if isinstance(response, list):
            return response
        elif isinstance(response, dict) and "tools" in response:
            return response["tools"]
        elif isinstance(response, dict) and "data" in response:
            return response["data"]
        else:
            # Try to extract JSON array from string response
            try:
                if isinstance(response, str):
                    # Look for JSON array in the response
                    match = re.search(r'\[.*\]', response, re.DOTALL)
                    if match:
                        return json.loads(match.group(0))
            except:
                pass
            
            # If all parsing attempts fail, return empty list
            return []