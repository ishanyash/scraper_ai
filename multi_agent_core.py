import os
import json
import time
import logging
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from openai import OpenAI
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class Agent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, name: str, model: str = "gpt-4"):
        self.name = name
        self.model = model
        self.logger = logging.getLogger(f"Agent.{name}")
        self.memory = []  # Simple memory to store past interactions
    
    def add_to_memory(self, entry: Dict[str, Any]):
        """Add an entry to the agent's memory"""
        self.memory.append(entry)
        # Keep memory size manageable
        if len(self.memory) > 10:
            self.memory.pop(0)
    
    def get_memory_context(self) -> str:
        """Get a string representation of the agent's memory for context"""
        if not self.memory:
            return ""
        
        context = "Previous insights and decisions:\n\n"
        for i, entry in enumerate(self.memory, 1):
            context += f"{i}. {entry.get('type', 'Note')}: {entry.get('content', '')}\n"
        
        return context
    
    def call_ai(self, system_prompt: str, user_prompt: str, json_response: bool = False) -> Union[str, Dict[str, Any]]:
        """Call the AI model with given prompts"""
        try:
            self.logger.info(f"Calling AI with prompt: {user_prompt[:100]}...")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON if required
            if json_response:
                try:
                    # Try to find JSON in the response
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        return json.loads(json_str)
                    
                    # If no JSON object found, try array
                    start_idx = content.find('[')
                    end_idx = content.rfind(']') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        return json.loads(json_str)
                    
                    # If we couldn't find JSON brackets, try parsing the whole thing
                    return json.loads(content)
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse JSON from response")
                    # Return the raw content if JSON parsing fails
                    return content
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error calling AI: {str(e)}")
            return f"Error: {str(e)}"
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Process the input data and return the result"""
        pass


class CommandCenter:
    """Central coordinator for all agents"""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.logger = logging.getLogger("CommandCenter")
        self.agents = {}
        self.execution_plan = {}
        self.results = {}
    
    def register_agent(self, agent: Agent):
        """Register an agent with the command center"""
        self.agents[agent.name] = agent
        self.logger.info(f"Registered agent: {agent.name}")
    
    def create_execution_plan(self, query: str) -> Dict[str, Any]:
        """Create an execution plan for the given query"""
        system_prompt = """
        You are the command center of a multi-agent web scraping system. Your job is to create an execution plan
        for the given user query. The plan should specify which agents to use in what order, and what specific
        instructions to give to each agent.
        
        Available agents:
        1. QueryAnalyzer - Interprets vague queries and identifies required data fields
        2. SearchAgent - Finds relevant sources on the web
        3. ExtractionAgent - Extracts data from web sources
        4. DataProcessorAgent - Processes and structures the extracted data
        5. ValidationAgent - Validates and enriches the dataset
        
        Return a JSON object with the execution plan, including the order of agents to run and the specific
        instructions for each agent.
        """
        
        user_prompt = f"""
        User Query: {query}
        
        Create a detailed execution plan that will result in the best possible dataset for this query.
        For each agent, provide specific instructions that will help them perform their task effectively.
        
        Format your response as a JSON object with the following structure:
        {{
            "interpreted_query": "The query interpreted in a clear, specific way",
            "required_fields": ["field1", "field2", ...],
            "execution_order": ["Agent1", "Agent2", ...],
            "agent_instructions": {{
                "Agent1": "Specific instructions for Agent1",
                "Agent2": "Specific instructions for Agent2",
                ...
            }}
        }}
        """
        
        try:
            plan = self.call_ai(system_prompt, user_prompt, json_response=True)
            self.execution_plan = plan
            self.logger.info(f"Created execution plan: {json.dumps(plan, indent=2)}")
            return plan
        except Exception as e:
            self.logger.error(f"Error creating execution plan: {str(e)}")
            return {"error": str(e)}
    
    def execute_plan(self, query: str) -> Dict[str, Any]:
        """Execute the plan for the given query"""
        # Create the execution plan
        plan = self.create_execution_plan(query)
        
        if "error" in plan:
            return plan
        
        # Execute each agent in order
        self.results = {"query": query, "plan": plan}
        
        for agent_name in plan.get("execution_order", []):
            if agent_name not in self.agents:
                self.logger.warning(f"Agent {agent_name} not found, skipping")
                continue
            
            agent = self.agents[agent_name]
            instructions = plan.get("agent_instructions", {}).get(agent_name, "")
            
            self.logger.info(f"Executing agent: {agent_name}")
            
            # Prepare input data for the agent
            input_data = {
                "query": query,
                "interpreted_query": plan.get("interpreted_query", query),
                "required_fields": plan.get("required_fields", []),
                "instructions": instructions,
                "previous_results": self.results
            }
            
            # Process with the agent
            result = agent.process(input_data)
            
            # Store the result
            self.results[agent_name] = result
            
            self.logger.info(f"Completed execution of {agent_name}")
        
        return self.results
    
    def call_ai(self, system_prompt: str, user_prompt: str, json_response: bool = False) -> Union[str, Dict[str, Any]]:
        """Call the AI model with given prompts"""
        try:
            self.logger.info(f"Command Center calling AI with prompt: {user_prompt[:100]}...")
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
            # Parse JSON if required
            if json_response:
                try:
                    # Try to find JSON in the response
                    start_idx = content.find('{')
                    end_idx = content.rfind('}') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        return json.loads(json_str)
                    
                    # If no JSON object found, try array
                    start_idx = content.find('[')
                    end_idx = content.rfind(']') + 1
                    
                    if start_idx >= 0 and end_idx > start_idx:
                        json_str = content[start_idx:end_idx]
                        return json.loads(json_str)
                    
                    # If we couldn't find JSON brackets, try parsing the whole thing
                    return json.loads(content)
                except json.JSONDecodeError:
                    self.logger.warning("Failed to parse JSON from response")
                    # Return the raw content if JSON parsing fails
                    return content
            
            return content
            
        except Exception as e:
            self.logger.error(f"Error calling AI: {str(e)}")
            return f"Error: {str(e)}"