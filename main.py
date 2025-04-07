import os
import json
import time
import pandas as pd
from typing import Dict, List, Any, Optional
import argparse
import logging
from datetime import datetime

from multi_agent_core import CommandCenter
from query_analyzer import QueryAnalyzerAgent
from search_agent import SearchAgent
from extraction_agent import ExtractionAgent
from data_processor import DataProcessorAgent
from validation_agent import ValidationAgent

class DatasetGenerator:
    """Responsible for generating the final dataset from validated data"""
    
    def __init__(self):
        self.logger = logging.getLogger("DatasetGenerator")
    
    def generate(self, data: List[Dict[str, Any]], query: str, formats: List[str] = ["csv"]) -> Dict[str, Any]:
        """Generate dataset files in the specified formats"""
        if not data:
            return {"error": "No data to generate dataset"}
        
        # Create a clean filename from the query
        clean_query = "".join(c if c.isalnum() else "_" for c in query).lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"{clean_query}_{timestamp}"
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Save in requested formats
        output_paths = {}
        
        if "csv" in formats or "all" in formats:
            csv_path = f"{filename_base}.csv"
            df.to_csv(csv_path, index=False)
            output_paths["csv"] = csv_path
            self.logger.info(f"Generated CSV dataset: {csv_path}")
            
        if "json" in formats or "all" in formats:
            json_path = f"{filename_base}.json"
            df.to_json(json_path, orient="records", indent=2)
            output_paths["json"] = json_path
            self.logger.info(f"Generated JSON dataset: {json_path}")
            
        if "excel" in formats or "all" in formats:
            try:
                excel_path = f"{filename_base}.xlsx"
                df.to_excel(excel_path, index=False)
                output_paths["excel"] = excel_path
                self.logger.info(f"Generated Excel dataset: {excel_path}")
            except Exception as e:
                self.logger.error(f"Error generating Excel file: {str(e)}")
                output_paths["excel_error"] = str(e)
        
        # Generate dataset summary
        summary = {
            "query": query,
            "row_count": len(df),
            "column_count": len(df.columns),
            "columns": list(df.columns),
            "output_paths": output_paths,
            "formats_saved": list(output_paths.keys())
        }
        
        # Save metadata file
        metadata_path = f"{filename_base}_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        return summary


def run_multi_agent_system(query: str, model: str = "gpt-4", output_formats: List[str] = ["csv"]) -> Dict[str, Any]:
    """Run the complete multi-agent system workflow"""
    logger = logging.getLogger("MultiAgentSystem")
    logger.info(f"Starting multi-agent system for query: {query}")
    
    # Initialize the command center
    command_center = CommandCenter(model=model)
    
    # Initialize specialized agents
    query_analyzer = QueryAnalyzerAgent(model=model)
    search_agent = SearchAgent(model=model)
    extraction_agent = ExtractionAgent(model=model)
    data_processor = DataProcessorAgent(model=model)
    validation_agent = ValidationAgent(model=model)
    
    # Register agents with the command center
    command_center.register_agent(query_analyzer)
    command_center.register_agent(search_agent)
    command_center.register_agent(extraction_agent)
    command_center.register_agent(data_processor)
    command_center.register_agent(validation_agent)
    
    # Execute the plan
    start_time = time.time()
    results = command_center.execute_plan(query)
    execution_time = time.time() - start_time
    
    # Check if we have validated data
    validated_data = []
    if "ValidationAgent" in results and "validated_data" in results["ValidationAgent"]:
        validated_data = results["ValidationAgent"]["validated_data"]
        logger.info(f"Using validated data: {len(validated_data)} items")
    
    # If no validated data but we have processed data, use that
    if not validated_data and "DataProcessorAgent" in results and "processed_data" in results["DataProcessorAgent"]:
        validated_data = results["DataProcessorAgent"]["processed_data"]
        logger.info(f"Using processed data: {len(validated_data)} items")
    
    # If still no data but we have extracted data, use that
    if not validated_data and "ExtractionAgent" in results and "extracted_data" in results["ExtractionAgent"]:
        validated_data = results["ExtractionAgent"]["extracted_data"]
        logger.info(f"Using extracted data: {len(validated_data)} items")
    
    # Generate the dataset if we have data
    dataset_info = {}
    if validated_data:
        dataset_generator = DatasetGenerator()
        dataset_info = dataset_generator.generate(validated_data, query, output_formats)
    else:
        error_message = "No data collected. Check agent results for details."
        logger.error(error_message)
        
        # Debug information about agent results
        logger.error("Agent result keys: " + str(results.keys()))
        for agent_name, agent_results in results.items():
            if isinstance(agent_results, dict):
                logger.error(f"{agent_name} keys: {agent_results.keys()}")
        
        dataset_info = {"error": error_message}
    
    # Compile final results
    final_results = {
        "query": query,
        "execution_time": execution_time,
        "dataset_info": dataset_info,
        "agent_execution_plan": results.get("plan", {}),
        "data_collected": len(validated_data),
        "success": len(validated_data) > 0
    }
    
    # Save the execution report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"execution_report_{timestamp}.json"
    
    # Create a summarized report (without the full data for brevity)
    report = {
        "query": query,
        "execution_time": execution_time,
        "dataset_info": dataset_info,
        "agent_execution_plan": results.get("plan", {}),
        "execution_summary": {
            "query_analysis": results.get("QueryAnalyzer", {}).get("interpreted_query", ""),
            "sources_found": len(results.get("SearchAgent", {}).get("sources", [])),
            "data_extracted": results.get("ExtractionAgent", {}).get("total_items", 0),
            "data_processed": results.get("DataProcessorAgent", {}).get("processed_count", 0),
            "data_quality": results.get("ValidationAgent", {}).get("data_quality", {}).get("overall_completeness", 0)
        }
    }
    
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    
    logger.info(f"Execution report saved to: {report_file}")
    logger.info(f"Multi-agent system completed in {execution_time:.2f} seconds with {len(validated_data)} data items")
    
    return final_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Agent Web Scraping System")
    parser.add_argument("query", type=str, help="The query to process")
    parser.add_argument("--model", type=str, default="gpt-4", help="OpenAI model to use")
    parser.add_argument("--output", type=str, default="csv", choices=["csv", "json", "excel", "all"], 
                      help="Output format for the dataset")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set up logging to file and console
    log_file = f"multi_agent_system_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG if args.debug else logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    
    # Run the system
    output_formats = [args.output] if args.output != "all" else ["csv", "json", "excel"]
    results = run_multi_agent_system(args.query, args.model, output_formats)
    
    # Print summary to console
    print("\nMulti-Agent Web Scraping System - Results Summary")
    print("=" * 60)
    print(f"Query: {args.query}")
    print(f"Execution Time: {results['execution_time']:.2f} seconds")
    print(f"Data Items Collected: {results['data_collected']}")
    print(f"Success: {'Yes' if results['success'] else 'No'}")
    
    if results['success'] and "dataset_info" in results and "output_paths" in results["dataset_info"]:
        print("\nDataset Files Created:")
        for format_name, path in results["dataset_info"]["output_paths"].items():
            print(f"- {format_name.upper()}: {path}")
    
    print(f"\nDetailed execution log saved to: {log_file}")
    
    if "dataset_info" in results and "error" in results["dataset_info"]:
        print(f"\nError: {results['dataset_info']['error']}")