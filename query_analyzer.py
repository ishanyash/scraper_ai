import json
import logging
from typing import Dict, List, Any
from multi_agent_core import Agent

class QueryAnalyzerAgent(Agent):
    """Agent responsible for analyzing and interpreting vague queries"""
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__("QueryAnalyzer", model)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process the input query and identify required data fields"""
        query = input_data.get("query", "")
        instructions = input_data.get("instructions", "")
        
        system_prompt = """
        You are an expert query analyzer for data collection tasks. Your role is to interpret user queries,
        especially vague ones, and identify what kind of data the user is looking for.
        
        For any query, you should:
        1. Interpret the query in a clear, specific way
        2. Identify the main entities and concepts
        3. Specify the data fields that should be collected
        4. Suggest potential data sources
        5. Identify any ambiguities or assumptions you're making
        
        Provide your analysis in a structured JSON format.
        """
        
        user_prompt = f"""
        Query: {query}
        
        Additional Instructions: {instructions}
        
        {self.get_memory_context()}
        
        Please analyze this query and provide a detailed interpretation of what data needs to be collected.
        Be thorough in identifying all relevant data fields that would make for a high-quality dataset.
        
        Format your response as a JSON object with the following structure:
        {{
            "interpreted_query": "The query interpreted in a clear, specific way",
            "main_entities": ["entity1", "entity2", ...],
            "required_data_fields": [
                {{"name": "field1", "description": "Description of field1", "importance": "high/medium/low"}},
                {{"name": "field2", "description": "Description of field2", "importance": "high/medium/low"}},
                ...
            ],
            "potential_sources": ["source1", "source2", ...],
            "ambiguities": ["ambiguity1", "ambiguity2", ...],
            "assumptions": ["assumption1", "assumption2", ...],
            "search_keywords": ["keyword1", "keyword2", ...]
        }}
        """
        
        response = self.call_ai(system_prompt, user_prompt, json_response=True)
        
        # Add to memory
        self.add_to_memory({
            "type": "Query Analysis",
            "content": f"Interpreted '{query}' as '{response.get('interpreted_query', '')}' with {len(response.get('required_data_fields', []))} fields"
        })
        
        return response