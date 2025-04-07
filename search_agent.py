import json
import logging
import requests
import time
from typing import Dict, List, Any
from bs4 import BeautifulSoup
from multi_agent_core import Agent

class SearchAgent(Agent):
    """Agent responsible for finding relevant sources on the web"""
    
    def __init__(self, model: str = "gpt-4"):
        super().__init__("SearchAgent", model)
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5"
        }
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Find and validate sources for data collection"""
        query = data.get("interpreted_query", "")
        required_fields = data.get("required_fields", [])
        
        self.logger.info(f"Search agent received query: {query}")
        
        # Generate search strategies
        search_strategies = [
            "Search for comprehensive AI tool directories and rankings",
            "Look for technology review sites with AI tool comparisons",
            "Find official AI tool marketplaces and directories",
            "Search for recent blog posts and articles about top AI tools",
            "Look for AI tool comparison charts and lists"
        ]
        
        # Generate source suggestions using AI
        system_prompt = """
        You are an expert at finding reliable sources for data collection. Your task is to identify websites that:
        1. Contain lists or directories of AI tools
        2. Include detailed information about features and pricing
        3. Are regularly updated
        4. Have good reputation and reliability
        
        Focus on sources like:
        - Official AI tool directories
        - Technology review sites
        - Software comparison platforms
        - Industry blogs and news sites
        
        Return your response in JSON format with the following structure:
        {
            "sources": [
                {
                    "url": "URL of the source",
                    "type": "Type of source (directory/review/blog)",
                    "reliability": "high/medium/low",
                    "expected_fields": ["field1", "field2"]
                }
            ]
        }
        """
        
        user_prompt = f"""
        Query: {query}
        
        Required fields to find in sources:
        {required_fields}
        
        Search strategies to consider:
        {search_strategies}
        
        Identify 10-15 potential sources that would be good for collecting this data.
        For each source, provide:
        1. URL
        2. Type of source (e.g., directory, review site, marketplace)
        3. Expected data quality (high/medium/low)
        4. Likelihood of finding required fields
        """
        
        # Try to get source suggestions as JSON
        try:
            source_suggestions = self.call_ai(system_prompt, user_prompt, json_response=True)
            if isinstance(source_suggestions, str):
                # Try to parse JSON from string response
                source_suggestions = json.loads(source_suggestions)
        except Exception as e:
            self.logger.warning(f"Error parsing AI response as JSON: {str(e)}")
            # Use default sources if AI response fails
            source_suggestions = {
                "sources": [
                    {
                        "url": "https://www.g2.com/categories/artificial-intelligence-platforms",
                        "type": "directory",
                        "reliability": "high",
                        "expected_fields": ["name", "features", "pricing"]
                    },
                    {
                        "url": "https://www.capterra.com/artificial-intelligence-software/",
                        "type": "directory",
                        "reliability": "high",
                        "expected_fields": ["name", "features", "pricing"]
                    },
                    {
                        "url": "https://www.predictiveanalyticstoday.com/top-free-artificial-intelligence-software/",
                        "type": "review",
                        "reliability": "medium",
                        "expected_fields": ["name", "features", "pricing"]
                    },
                    {
                        "url": "https://www.softwaretestinghelp.com/ai-tools/",
                        "type": "review",
                        "reliability": "medium",
                        "expected_fields": ["name", "features"]
                    },
                    {
                        "url": "https://www.techradar.com/best/best-ai-tools",
                        "type": "review",
                        "reliability": "high",
                        "expected_fields": ["name", "features", "pricing"]
                    }
                ]
            }
        
        # Add to memory
        self.add_to_memory({
            "type": "Source Research",
            "content": f"Identified {len(source_suggestions.get('sources', []))} potential sources for data collection"
        })
        
        # Now try to validate some of the top sources
        validated_sources = []
        
        for source in source_suggestions.get("sources", [])[:10]:  # Validate top 10 sources
            url = source.get("url", "")
            if not url:
                continue
                
            try:
                self.logger.info(f"Validating source: {url}")
                
                # Enhanced headers to better mimic a browser
                headers = {
                    **self.headers,
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1",
                    "DNT": "1"
                }
                
                # Try to fetch the page with a longer timeout
                response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
                
                # More lenient status code checking
                if response.status_code < 400:  # Accept any successful response
                    content_type = response.headers.get('content-type', '').lower()
                    
                    # Accept any text-based content type
                    if any(t in content_type for t in ['text/html', 'text/plain', 'application/json', 'application/xml']):
                        content = response.text
                        
                        # More lenient content size check
                        if len(content) > 100:  # Reduced minimum size requirement
                            soup = BeautifulSoup(content, 'html.parser')
                            
                            # Extract useful information regardless of robots meta
                            source["accessible"] = True
                            source["title"] = self._extract_title(content)
                            source["content_preview"] = content[:50000]  # Store first 50K chars for extraction
                            source["html_size"] = len(content)
                            source["status_code"] = response.status_code
                            
                            # Try to extract structured data if available
                            try:
                                # Look for JSON-LD
                                json_ld = soup.find_all('script', type='application/ld+json')
                                if json_ld:
                                    source["structured_data"] = [json.loads(script.string) for script in json_ld if script.string]
                                
                                # Look for meta tags with useful information
                                meta_tags = {}
                                for meta in soup.find_all('meta'):
                                    name = meta.get('name', meta.get('property', '')).lower()
                                    content = meta.get('content', '')
                                    if name and content:
                                        meta_tags[name] = content
                                source["meta_tags"] = meta_tags
                            except Exception as e:
                                self.logger.debug(f"Error extracting structured data: {str(e)}")
                        else:
                            source["accessible"] = False
                            source["error"] = "Page content too small"
                    else:
                        source["accessible"] = False
                        source["error"] = f"Unsupported content type: {content_type}"
                else:
                    source["accessible"] = False
                    source["status_code"] = response.status_code
                    source["error"] = f"HTTP {response.status_code}"
            except requests.exceptions.RequestException as e:
                self.logger.warning(f"Error validating source {url}: {str(e)}")
                source["accessible"] = False
                source["error"] = str(e)
            except Exception as e:
                self.logger.warning(f"Unexpected error validating source {url}: {str(e)}")
                source["accessible"] = False
                source["error"] = str(e)
            
            validated_sources.append(source)
            
            # Reduced delay between requests
            time.sleep(1)
        
        # Calculate confidence in sources
        accessible_sources = [s for s in validated_sources if s.get("accessible", False)]
        source_confidence = len(accessible_sources) / len(validated_sources) if validated_sources else 0
        
        # Prepare backup sources
        backup_sources = [
            "https://www.g2.com/categories/artificial-intelligence-platforms",
            "https://www.capterra.com/artificial-intelligence-software/",
            "https://www.predictiveanalyticstoday.com/top-free-artificial-intelligence-software/",
            "https://www.softwaretestinghelp.com/ai-tools/",
            "https://www.techradar.com/best/best-ai-tools"
        ]
        
        result = {
            "sources": validated_sources,
            "search_strategies": search_strategies,
            "backup_sources": backup_sources,
            "accessible_sources_count": len(accessible_sources),
            "total_sources_suggested": len(validated_sources),
            "source_confidence": source_confidence
        }
        
        self.logger.info(f"Found {len(accessible_sources)} accessible sources out of {len(validated_sources)} total sources")
        
        return result
    
    def _extract_title(self, html: str) -> str:
        """Extract the title from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        title_tag = soup.find('title')
        return title_tag.text.strip() if title_tag else "No title found"
    
    def _extract_content_preview(self, html: str, max_length: int = 200) -> str:
        """Extract a preview of the content from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        paragraphs = soup.find_all('p')
        
        if paragraphs:
            text = ' '.join([p.text.strip() for p in paragraphs[:3]])
            return text[:max_length] + "..." if len(text) > max_length else text
        
        # If no paragraphs, get any text
        text = soup.get_text()
        return text[:max_length] + "..." if len(text) > max_length else text