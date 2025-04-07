import json
import logging
import re
from typing import Dict, List, Any
from multi_agent_core import Agent

class DataProcessorAgent(Agent):
    """Agent responsible for processing and cleaning the extracted data"""
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__("DataProcessorAgent", model)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and clean the extracted data"""
        query = input_data.get("interpreted_query", input_data.get("query", ""))
        instructions = input_data.get("instructions", "")
        
        # Get extraction results
        extraction_results = input_data.get("previous_results", {}).get("ExtractionAgent", {})
        extracted_data = extraction_results.get("extracted_data", [])
        
        self.logger.info(f"DataProcessorAgent received {len(extracted_data)} items to process")
        
        if not extracted_data:
            self.logger.error("No extracted data to process")
            return {"error": "No extracted data to process", "processed_data": []}
        
        # Get query analysis if available
        query_analysis = input_data.get("previous_results", {}).get("QueryAnalyzer", {})
        required_fields = query_analysis.get("required_data_fields", [])
        
        # If required_fields is empty, use the required_fields from the plan
        if not required_fields:
            required_fields = input_data.get("required_fields", [])
            self.logger.info(f"Using required fields from plan: {required_fields}")
        
        # Prepare field names for standardization
        field_names = []
        for field in required_fields:
            if isinstance(field, dict):
                field_names.append(field.get("name", ""))
            else:
                field_names.append(field)
        
        self.logger.info(f"Processing data for fields: {field_names}")
        
        system_prompt = f"""
        You are an expert data processing agent. Your task is to clean, standardize, and enrich the extracted data.
        
        For the query: {query}
        
        Your data processing tasks include:
        1. Removing duplicates
        2. Standardizing field names and formats
        3. Filling in missing values where possible
        4. Validating data types and formats
        5. Enriching the data with additional derived fields if valuable
        
        The desired fields for the final dataset are: {', '.join(field_names)}
        
        Return your processed data as a JSON array of objects with standardized fields.
        """
        
        # Convert extracted_data to JSON string for the prompt
        # Limit the size to avoid token limits
        MAX_ITEMS_FOR_PROMPT = 50
        prompt_data = extracted_data[:MAX_ITEMS_FOR_PROMPT]
        data_json = json.dumps(prompt_data, indent=2)
        
        self.logger.info(f"Sending {len(prompt_data)} items to AI for processing (out of {len(extracted_data)} total)")
        
        user_prompt = f"""
        Additional Instructions: {instructions}
        
        {self.get_memory_context()}
        
        Here is the extracted data to process:
        {data_json}
        
        Please clean, standardize, and enrich this data. Ensure consistent field names and data formats.
        Remove any duplicates and validate the data. Fill in missing values where possible through inference
        or aggregation from other entries.
        
        Return the processed data as a JSON array of objects with standardized fields.
        Also include a summary of the processing steps performed and any issues found.
        """
        
        # Process with AI
        self.logger.info("Calling AI to process data")
        response = self.call_ai(system_prompt, user_prompt, json_response=True)
        self.logger.info("AI processing response received")
        
        # Handle various response formats
        processed_data = []
        processing_summary = {}
        
        if isinstance(response, list):
            self.logger.info(f"Received list with {len(response)} items")
            processed_data = response
            processing_summary = {"items_processed": len(response)}
        elif isinstance(response, dict) and "data" in response:
            self.logger.info(f"Received dict with 'data' field containing {len(response['data'])} items")
            processed_data = response["data"]
            # Remove the data field from the response for the summary
            summary_dict = response.copy()
            del summary_dict["data"]
            processing_summary = summary_dict
        elif isinstance(response, dict) and "processed_data" in response:
            self.logger.info(f"Received dict with 'processed_data' field containing {len(response['processed_data'])} items")
            processed_data = response["processed_data"]
            # Remove the processed_data field from the response for the summary
            summary_dict = response.copy()
            del summary_dict["processed_data"]
            processing_summary = summary_dict
        elif isinstance(response, str):
            # Try to extract JSON array
            try:
                start_idx = response.find('[')
                end_idx = response.rfind(']') + 1
                
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response[start_idx:end_idx]
                    processed_data = json.loads(json_str)
                    self.logger.info(f"Parsed JSON array from string, found {len(processed_data)} items")
                    processing_summary = {"items_processed": len(processed_data)}
                else:
                    self.logger.warning("No JSON array found in response string")
            except Exception as e:
                self.logger.error(f"Error parsing JSON from string: {str(e)}")
        else:
            self.logger.warning(f"Unexpected response format: {type(response)}")
        
        # Handle case where we only processed a subset of the data
        if len(extracted_data) > MAX_ITEMS_FOR_PROMPT and processed_data:
            self.logger.info(f"Processing remaining {len(extracted_data) - MAX_ITEMS_FOR_PROMPT} items")
            # Use the first processed items as a template for processing the remaining data
            # This is a simple approach - in a production system, you might want to use 
            # a trained model to process the rest of the data more efficiently
            remaining_processed = self._process_remaining_data(
                extracted_data[MAX_ITEMS_FOR_PROMPT:], 
                processed_data, 
                field_names
            )
            processed_data.extend(remaining_processed)
            self.logger.info(f"Added {len(remaining_processed)} more processed items")
        
        # Additional standardization and cleaning
        standardized_data = self._standardize_data(processed_data, field_names)
        self.logger.info(f"Standardized {len(processed_data)} items to {len(standardized_data)} items")
        
        # Add to memory
        self.add_to_memory({
            "type": "Data Processing",
            "content": f"Processed {len(processed_data)} items, standardized to {len(standardized_data)} items"
        })
        
        result = {
            "processed_data": standardized_data,
            "original_count": len(extracted_data),
            "processed_count": len(standardized_data),
            "processing_summary": processing_summary,
            "standardized_fields": list(standardized_data[0].keys()) if standardized_data else []
        }
        
        return result
    
    def _process_remaining_data(self, remaining_data: List[Dict[str, Any]], 
                               processed_template: List[Dict[str, Any]],
                               target_fields: List[str]) -> List[Dict[str, Any]]:
        """Process remaining data using the template from already processed data"""
        if not remaining_data or not processed_template:
            return []
        
        # Create field mappings from the template
        field_mapping = {}
        target_structure = {}
        
        # Use the first processed item as a template
        template = processed_template[0]
        
        # For each field in the original data, find the corresponding field in the processed data
        original_sample = remaining_data[0]
        for orig_field in original_sample.keys():
            for proc_field in template.keys():
                if self._are_fields_similar(orig_field, proc_field):
                    field_mapping[orig_field] = proc_field
                    break
        
        # Process remaining data using the mapping
        processed_remaining = []
        for item in remaining_data:
            processed_item = {}
            
            # Map fields using the established mapping
            for orig_field, value in item.items():
                if orig_field in field_mapping:
                    processed_item[field_mapping[orig_field]] = value
                else:
                    # For unmapped fields, try to find a match in target fields
                    for target in target_fields:
                        if self._are_fields_similar(orig_field, target):
                            processed_item[target] = value
                            field_mapping[orig_field] = target  # Update mapping for future use
                            break
                    else:
                        # If no match found, use the original field name
                        processed_item[orig_field] = value
            
            # Ensure all target fields exist in the processed item
            for field in template.keys():
                if field not in processed_item:
                    processed_item[field] = None
            
            processed_remaining.append(processed_item)
        
        return processed_remaining
    
    def _are_fields_similar(self, field1: str, field2: str) -> bool:
        """Check if two field names are similar"""
        # Normalize both fields
        f1 = field1.lower().replace(' ', '_').replace('-', '_')
        f2 = field2.lower().replace(' ', '_').replace('-', '_')
        
        # Check for exact match
        if f1 == f2:
            return True
        
        # Check for substring
        if f1 in f2 or f2 in f1:
            return True
        
        # Check for high similarity (e.g., "price" and "pricing")
        if len(f1) > 3 and len(f2) > 3:
            common_prefix_len = 0
            for i in range(min(len(f1), len(f2))):
                if f1[i] == f2[i]:
                    common_prefix_len += 1
                else:
                    break
            
            # If common prefix is more than 70% of the shorter string
            shorter_len = min(len(f1), len(f2))
            if common_prefix_len > shorter_len * 0.7:
                return True
        
        return False
    
    def _standardize_data(self, data: List[Dict[str, Any]], target_fields: List[str]) -> List[Dict[str, Any]]:
        """Standardize the data structure and field names"""
        if not data:
            return []
        
        standardized = []
        
        # Create a mapping of similar field names
        field_mapping = {}
        
        for item in data:
            standardized_item = {}
            
            # Map existing fields to target fields
            for field_name, value in item.items():
                mapped_field = self._map_field_name(field_name, target_fields, field_mapping)
                
                # Clean and standardize the value
                standardized_value = self._clean_value(value, mapped_field)
                standardized_item[mapped_field] = standardized_value
            
            # Check for missing target fields
            for target_field in target_fields:
                if target_field not in standardized_item:
                    standardized_item[target_field] = None
            
            standardized.append(standardized_item)
        
        # Remove duplicates (based on all fields except source information)
        unique_data = []
        seen = set()
        
        for item in standardized:
            # Create a fingerprint excluding source fields
            fingerprint = json.dumps({k: v for k, v in item.items() 
                                     if k not in ['source_url', 'source_title']})
            
            if fingerprint not in seen:
                seen.add(fingerprint)
                unique_data.append(item)
        
        return unique_data
    
    def _map_field_name(self, field_name: str, target_fields: List[str], mapping: Dict[str, str]) -> str:
        """Map a field name to a target field name"""
        # Check if we already have this mapping
        if field_name in mapping:
            return mapping[field_name]
        
        # Normalize the field name
        normalized = field_name.lower().replace(' ', '_').replace('-', '_')
        
        # Check for exact match in target fields
        for target in target_fields:
            normalized_target = target.lower().replace(' ', '_').replace('-', '_')
            if normalized == normalized_target:
                mapping[field_name] = target
                return target
        
        # Check for partial match
        for target in target_fields:
            normalized_target = target.lower().replace(' ', '_').replace('-', '_')
            if normalized in normalized_target or normalized_target in normalized:
                mapping[field_name] = target
                return target
        
        # If no match found, keep the original
        mapping[field_name] = field_name
        return field_name
    
    def _clean_value(self, value: Any, field_name: str) -> Any:
        """Clean and standardize a value based on field name and value type"""
        if value is None:
            return None
        
        # Convert to string for cleaning
        if not isinstance(value, str):
            return value
        
        value = value.strip()
        
        # Empty string to None
        if not value:
            return None
        
        # Date standardization
        if any(date_keyword in field_name.lower() for date_keyword in ['date', 'time', 'year', 'month', 'day']):
            # Try to standardize date format
            try:
                # Look for date patterns
                date_pattern = r'(\d{1,4}[-/\.]\d{1,2}[-/\.]\d{1,4}|\d{1,2}[-/\.]\d{1,2}[-/\.]\d{2,4})'
                match = re.search(date_pattern, value)
                if match:
                    return match.group(1)
            except:
                pass
        
        # URL standardization
        if any(url_keyword in field_name.lower() for url_keyword in ['url', 'link', 'website']):
            if not value.startswith(('http://', 'https://')):
                value = 'https://' + value
        
        # Price standardization
        if any(price_keyword in field_name.lower() for price_keyword in ['price', 'cost', 'fee']):
            # Extract numeric value
            price_pattern = r'[\$£€]?\s*(\d+(?:,\d{3})*(?:\.\d{1,2})?)'
            match = re.search(price_pattern, value)
            if match:
                # Remove commas and convert to float
                numeric_value = match.group(1).replace(',', '')
                try:
                    return float(numeric_value)
                except:
                    pass
        
        return value