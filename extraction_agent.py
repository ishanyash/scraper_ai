import requests
import logging
import json
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional
from multi_agent_core import Agent
import time
import re

class ExtractionAgent(Agent):
    """Agent responsible for extracting data from web sources"""
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__("ExtractionAgent", model)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data from given sources"""
        self.logger.info("Starting data extraction from sources")
        
        # Get sources from previous results
        search_results = data.get("previous_results", {}).get("SearchAgent", {})
        sources = search_results.get("sources", [])
        required_fields = data.get("required_fields", [])
        
        if not sources:
            self.logger.warning("No sources provided for extraction")
            return {
                "extracted_data": [],
                "total_items": 0,
                "sources_processed": 0,
                "extraction_success_rate": 0
            }
        
        # Sort sources by reliability score
        sources.sort(key=lambda x: x.get("reliability", "low") == "high", reverse=True)
        
        extracted_data = []
        processed_sources = 0
        successful_extractions = 0
        
        # Process each source
        for source in sources:
            if not source.get("accessible", False):
                continue
                
            try:
                url = source.get("url", "Unknown URL")
                self.logger.info(f"Processing source: {url}")
                
                # Get the content
                content = source.get("content_preview", "")
                if not content:
                    continue
                
                # Try to extract data using different methods
                items = []
                
                # 1. Try structured data first
                if "structured_data" in source:
                    items.extend(self._extract_from_structured_data(source["structured_data"], required_fields))
                
                # 2. Try HTML parsing if no items found
                if not items:
                    items.extend(self._extract_from_html(content, required_fields))
                
                # 3. If still no items, try AI extraction
                if not items:
                    items.extend(self._extract_with_ai(content, url, required_fields))
                
                if items:
                    # Add source information to items
                    for item in items:
                        item["source_url"] = url
                        item["source_title"] = source.get("title", "")
                        item["source_reliability"] = source.get("reliability", "medium")
                    
                    extracted_data.extend(items)
                    successful_extractions += 1
                
                processed_sources += 1
                
            except Exception as e:
                self.logger.error(f"Error processing source {url}: {str(e)}")
                continue
        
        # Remove duplicates based on tool names
        unique_data = []
        seen_names = set()
        for item in extracted_data:
            name = item.get("name", "").lower()
            if name and name not in seen_names:
                seen_names.add(name)
                unique_data.append(item)
        
        # Calculate success rate
        success_rate = (successful_extractions / processed_sources * 100) if processed_sources > 0 else 0
        
        self.logger.info(f"Extraction completed: {len(unique_data)} unique items from {processed_sources} sources")
        
        return {
            "extracted_data": unique_data,
            "total_items": len(unique_data),
            "sources_processed": processed_sources,
            "extraction_success_rate": success_rate
        }
    
    def _extract_from_structured_data(self, structured_data: List[Dict], required_fields: List[str]) -> List[Dict]:
        """Extract data from structured data (JSON-LD)"""
        items = []
        
        for data in structured_data:
            try:
                # Handle different structured data types
                if isinstance(data, dict):
                    if "@type" in data:
                        item_type = data["@type"].lower()
                        
                        # Handle different types of structured data
                        if "softwareapplication" in item_type or "product" in item_type:
                            item = {
                                "name": data.get("name", ""),
                                "description": data.get("description", ""),
                                "url": data.get("url", ""),
                                "price": data.get("offers", {}).get("price", ""),
                                "currency": data.get("offers", {}).get("priceCurrency", ""),
                                "features": data.get("featureList", []),
                                "category": data.get("applicationCategory", ""),
                                "rating": data.get("aggregateRating", {}).get("ratingValue", ""),
                                "review_count": data.get("aggregateRating", {}).get("reviewCount", "")
                            }
                            items.append(item)
            except Exception as e:
                self.logger.debug(f"Error parsing structured data: {str(e)}")
                continue
        
        return items
    
    def _extract_from_html(self, html_content: str, required_fields: List[str]) -> List[Dict]:
        """Extract data from HTML content"""
        items = []
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Look for common patterns in AI tool listings
        # 1. Look for list items or cards that might contain tool information
        potential_items = soup.find_all(['div', 'article', 'section', 'li'], class_=lambda x: x and any(term in x.lower() for term in ['tool', 'card', 'item', 'product', 'listing', 'result']))
        
        for element in potential_items:
            try:
                # Try to find the tool name
                name_elem = element.find(['h1', 'h2', 'h3', 'h4', 'strong', 'b', 'a'], class_=lambda x: x and any(term in str(x).lower() for term in ['title', 'name', 'heading']))
                if not name_elem:
                    name_elem = element.find(['h1', 'h2', 'h3', 'h4', 'strong', 'b', 'a'])
                
                if not name_elem:
                    continue
                    
                name = name_elem.get_text().strip()
                if not name or len(name) < 2:
                    continue
                
                # Initialize item with name
                item = {"name": name}
                
                # Try to find description
                desc_elem = element.find(['p', 'div'], class_=lambda x: x and any(term in str(x).lower() for term in ['desc', 'text', 'content', 'summary']))
                if desc_elem:
                    item["description"] = desc_elem.get_text().strip()
                
                # Try to find pricing
                price_elems = element.find_all(text=re.compile(r'(\$|\€|\£)\s*\d+|\b(free|pricing|cost)\b', re.IGNORECASE))
                if price_elems:
                    item["price"] = ' '.join([p.strip() for p in price_elems])
                
                # Try to find features
                features = []
                # Look for feature lists
                feature_list = element.find(['ul', 'ol'], class_=lambda x: x and any(term in str(x).lower() for term in ['feature', 'benefit', 'capability']))
                if feature_list:
                    for feat in feature_list.find_all('li'):
                        features.append(feat.get_text().strip())
                
                # Look for feature sections
                feature_sections = element.find_all(['div', 'span', 'p'], class_=lambda x: x and any(term in str(x).lower() for term in ['feature', 'benefit', 'capability']))
                for section in feature_sections:
                    features.append(section.get_text().strip())
                
                if features:
                    item["features"] = features
                
                # Try to find URL
                url_elem = element.find('a', href=True)
                if url_elem:
                    item["url"] = url_elem['href']
                
                items.append(item)
                
            except Exception as e:
                self.logger.debug(f"Error extracting item from HTML: {str(e)}")
                continue
        
        return items
    
    def _extract_with_ai(self, content: str, url: str, required_fields: List[str]) -> List[Dict]:
        """Use AI to extract data from content when other methods fail"""
        try:
            # Clean up content
            soup = BeautifulSoup(content, 'html.parser')
            for script in soup(["script", "style"]):
                script.extract()
            
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            # Limit text length
            text = text[:15000]
            
            system_prompt = """
            You are an expert at extracting information about AI tools from web content.
            Your task is to identify and extract details about AI tools, including:
            1. Tool name
            2. Features and capabilities
            3. Pricing information
            4. Any other relevant details
            
            Return the data in a structured JSON format.
            """
            
            user_prompt = f"""
            Source URL: {url}
            Required fields: {required_fields}
            
            Content to analyze:
            {text}
            
            Extract information about any AI tools mentioned in the content.
            Format each tool as a JSON object with the required fields.
            If you can't find certain information, use null values.
            Return an array of tool objects.
            """
            
            response = self.call_ai(system_prompt, user_prompt, json_response=True)
            
            if isinstance(response, list):
                return response
            elif isinstance(response, dict) and "tools" in response:
                return response["tools"]
            elif isinstance(response, dict) and "data" in response:
                return response["data"]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error in AI extraction: {str(e)}")
            return []