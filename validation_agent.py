import json
import logging
import re
from typing import Dict, List, Any
from multi_agent_core import Agent

class ValidationAgent(Agent):
    """Agent responsible for validating and enhancing the dataset"""
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__("ValidationAgent", model)
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and enhance the processed data"""
        query = input_data.get("interpreted_query", input_data.get("query", ""))
        instructions = input_data.get("instructions", "")
        
        # Get processing results
        processing_results = input_data.get("previous_results", {}).get("DataProcessorAgent", {})
        processed_data = processing_results.get("processed_data", [])
        
        self.logger.info(f"ValidationAgent received {len(processed_data)} items to validate")
        
        if not processed_data:
            self.logger.error("No processed data to validate")
            return {"error": "No processed data to validate", "validated_data": []}
        
        # Get query analysis if available
        query_analysis = input_data.get("previous_results", {}).get("QueryAnalyzer", {})
        
        system_prompt = f"""
        You are an expert data validation agent. Your task is to validate and enhance the dataset
        to ensure it meets high quality standards.
        
        For the query: {query}
        
        Your validation tasks include:
        1. Checking for data consistency
        2. Identifying outliers and anomalies
        3. Verifying data completeness
        4. Checking for logical consistency
        5. Suggesting improvements or enrichments
        
        Return your validation report and the validated dataset as a JSON object.
        """
        
        # Convert processed_data to JSON string for the prompt
        # Limit to a smaller sample for the AI analysis
        MAX_ITEMS_FOR_PROMPT = 20
        sample_data = processed_data[:MAX_ITEMS_FOR_PROMPT]
        data_json = json.dumps(sample_data, indent=2)
        
        self.logger.info(f"Sending sample of {len(sample_data)} items to AI for validation")
        
        user_prompt = f"""
        Additional Instructions: {instructions}
        
        {self.get_memory_context()}
        
        Here is a sample of the processed data (showing first {len(sample_data)} items out of {len(processed_data)} total):
        {data_json}
        
        Please validate this dataset and provide a detailed validation report.
        Identify any issues, inconsistencies, or anomalies.
        Suggest improvements or enrichments to enhance the dataset quality.
        
        Return a JSON object with:
        1. A validation report including statistics and issues found
        2. A quality score (0-100)
        3. Enhancement suggestions
        """
        
        # Process with AI
        self.logger.info("Calling AI to validate data")
        response = self.call_ai(system_prompt, user_prompt, json_response=True)
        self.logger.info("AI validation response received")
        
        # Extract validation report
        validation_report = {}
        
        if isinstance(response, dict):
            if "validation_report" in response:
                validation_report = response["validation_report"]
                self.logger.info("Found validation_report in response")
            else:
                validation_report = response
                self.logger.info("Using entire response as validation report")
        else:
            self.logger.warning(f"Unexpected response format: {type(response)}")
            validation_report = {"error": "Unexpected response format", "quality_score": 50}
        
        # Apply fixes and enhancements based on the validation
        enhanced_data = self._enhance_data(processed_data, validation_report)
        self.logger.info(f"Enhanced {len(processed_data)} items to {len(enhanced_data)} items")
        
        # Add quality metrics
        data_quality = self._calculate_quality_metrics(enhanced_data)
        
        # Add to memory
        quality_score = validation_report.get("quality_score", 0)
        if isinstance(quality_score, str):
            try:
                quality_score = float(quality_score.replace('%', ''))
            except:
                quality_score = 0
                
        self.add_to_memory({
            "type": "Data Validation",
            "content": f"Validated {len(enhanced_data)} items with quality score: {quality_score}/100"
        })
        
        result = {
            "validated_data": enhanced_data,
            "validation_report": validation_report,
            "data_quality": data_quality,
            "completeness_rate": self._calculate_completeness(enhanced_data),
            "enhancement_suggestions": validation_report.get("enhancement_suggestions", [])
        }
        
        return result
    
    def _enhance_data(self, data: List[Dict[str, Any]], validation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply enhancements and fixes based on validation report"""
        if not data:
            return []
        
        enhanced_data = data.copy()
        
        # Apply fixes if suggested in the validation report
        if "fixes" in validation_report:
            fixes = validation_report["fixes"]
            if isinstance(fixes, list):
                for fix in fixes:
                    if isinstance(fix, dict) and "field" in fix and "action" in fix:
                        field = fix["field"]
                        action = fix["action"]
                        
                        # Apply the fix action to the field
                        if action == "remove":
                            for item in enhanced_data:
                                if field in item:
                                    del item[field]
                        elif action == "standardize" and "format" in fix:
                            format_pattern = fix["format"]
                            for item in enhanced_data:
                                if field in item and item[field]:
                                    # Simple standardization based on format hint
                                    item[field] = self._standardize_value(item[field], format_pattern)
        
        return enhanced_data
    
    def _standardize_value(self, value: Any, format_hint: str) -> Any:
        """Standardize a value based on a format hint"""
        if value is None:
            return None
            
        if isinstance(value, str):
            value = value.strip()
            
            # Date standardization
            if format_hint.lower() in ["date", "yyyy-mm-dd", "mm/dd/yyyy", "date-iso"]:
                # Try to convert to YYYY-MM-DD
                try:
                    # Various date patterns
                    patterns = [
                        r'(\d{4})[-/\.](\d{1,2})[-/\.](\d{1,2})',  # YYYY-MM-DD
                        r'(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{4})',  # MM-DD-YYYY or DD-MM-YYYY
                        r'(\d{1,2})[-/\.](\d{1,2})[-/\.](\d{2})'   # MM-DD-YY or DD-MM-YY
                    ]
                    
                    for pattern in patterns:
                        match = re.search(pattern, value)
                        if match:
                            groups = match.groups()
                            if len(groups[0]) == 4:  # YYYY-MM-DD format
                                year, month, day = groups
                                # Simple validation
                                month = int(month)
                                day = int(day)
                                if 1 <= month <= 12 and 1 <= day <= 31:
                                    return f"{year}-{month:02d}-{day:02d}"
                            elif len(groups[2]) == 4:  # MM-DD-YYYY or DD-MM-YYYY format
                                if format_hint.lower() == "mm/dd/yyyy":
                                    month, day, year = groups
                                else:
                                    day, month, year = groups
                                # Simple validation
                                month = int(month)
                                day = int(day)
                                if 1 <= month <= 12 and 1 <= day <= 31:
                                    return f"{year}-{month:02d}-{day:02d}"
                except:
                    pass
            
            # URL standardization
            if format_hint.lower() in ["url", "website", "link"]:
                if not value.startswith(('http://', 'https://')):
                    return 'https://' + value
        
        return value
    
    def _calculate_quality_metrics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate quality metrics for the dataset"""
        if not data:
            return {"error": "No data to calculate metrics"}
        
        # Get all fields
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        # Calculate completeness per field
        field_completeness = {}
        for field in all_fields:
            non_empty = sum(1 for item in data if field in item and item[field] is not None and item[field] != "")
            field_completeness[field] = (non_empty / len(data)) * 100
        
        # Calculate overall completeness
        overall_completeness = sum(field_completeness.values()) / len(field_completeness)
        
        # Calculate consistency (% of items that have the same fields)
        field_counts = {}
        for item in data:
            fields_tuple = tuple(sorted(item.keys()))
            field_counts[fields_tuple] = field_counts.get(fields_tuple, 0) + 1
        
        most_common_structure = max(field_counts.items(), key=lambda x: x[1])
        structure_consistency = (most_common_structure[1] / len(data)) * 100
        
        # Calculate uniqueness (absence of duplicates)
        unique_items = set()
        for item in data:
            # Create a fingerprint excluding source fields
            fingerprint = json.dumps({k: v for k, v in item.items() if k not in ['source_url', 'source_title']})
            unique_items.add(fingerprint)
        
        uniqueness = (len(unique_items) / len(data)) * 100
        
        return {
            "overall_completeness": overall_completeness,
            "field_completeness": field_completeness,
            "structure_consistency": structure_consistency,
            "uniqueness": uniqueness,
            "total_items": len(data)
        }
    
    def _calculate_completeness(self, data: List[Dict[str, Any]]) -> float:
        """Calculate the overall completeness of the dataset"""
        if not data:
            return 0.0
        
        # Get all fields
        all_fields = set()
        for item in data:
            all_fields.update(item.keys())
        
        # Calculate completeness
        total_fields = len(all_fields) * len(data)
        filled_fields = 0
        
        for item in data:
            for field in all_fields:
                if field in item and item[field] is not None and item[field] != "":
                    filled_fields += 1
        
        return (filled_fields / total_fields) * 100