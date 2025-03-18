"""
Tools for CrewAI agents using Ollama models.
"""

import json
import ollama
from typing import Dict, Any, List
from langchain.tools import BaseTool

class OllamaTravelAdvisorTool(BaseTool):
    """Tool that uses Ollama model for travel advice."""
    
    name = "Ollama Travel Advisor"
    description = "Generate travel advice, recommendations, and itineraries for Morocco"
    
    def __init__(self, model_name="llama2"):
        """Initialize with Ollama model."""
        super().__init__()
        self.model_name = model_name
        
        try:
            # Test connection to Ollama
            ollama.list()
            self.initialized = True
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print("Make sure Ollama is installed and running (https://ollama.ai)")
            self.initialized = False
    
    def _run(self, query: str) -> str:
        """Execute the Ollama model on the provided query."""
        if not self.initialized:
            return ("Error: Ollama not initialized. Make sure Ollama is installed and running. "
                   "Visit https://ollama.ai for installation instructions.")
        
        try:
            prompt = f"""You are a travel advisor specializing in Morocco tourism. 
            Provide detailed, helpful information about the following query:
            
            {query}
            
            Focus on accurate, practical advice that would help travelers have an authentic experience.
            Include specific locations, cultural insights, and practical tips where relevant.
            """
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 500  # Approximately 500 tokens
                }
            )
            
            # Extract just the response text
            return response['response'].strip()
        
        except Exception as e:
            return f"Error generating travel advice with Ollama: {str(e)}"

class OllamaDestinationMatcherTool(BaseTool):
    """Tool that uses Ollama to match traveler preferences to Morocco destinations."""
    
    name = "Ollama Destination Matcher"
    description = "Match traveler preferences to suitable destinations in Morocco"
    
    def __init__(self, model_name="llama2"):
        """Initialize with Ollama model."""
        super().__init__()
        self.model_name = model_name
        
        try:
            # Test connection to Ollama
            ollama.list()
            self.initialized = True
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            self.initialized = False
    
    def _run(self, preferences: str) -> str:
        """Match preferences to destinations using Ollama."""
        if not self.initialized:
            return "Error: Ollama not initialized. Make sure Ollama is installed and running."
        
        try:
            from data.morocco_info import MOROCCO_DESTINATIONS
            
            # Create a structured prompt for the model
            destinations_json = json.dumps(MOROCCO_DESTINATIONS, indent=2)
            
            prompt = f"""You are a travel advisor for Morocco. Based on the following traveler preferences:

"{preferences}"

And using this data about Moroccan destinations:
{destinations_json}

Select the top 3 destinations that best match the traveler's preferences. 
For each destination, explain why it matches their preferences in 2-3 sentences.
Format your response as a markdown list with destination names as headings.
"""
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.2,  # Lower temperature for more focused responses
                    "top_p": 0.9,
                    "num_predict": 500
                }
            )
            
            return response['response'].strip()
        
        except Exception as e:
            return f"Error matching destinations with Ollama: {str(e)}"

class OllamaItineraryGeneratorTool(BaseTool):
    """Tool that uses Ollama to generate travel itineraries for Morocco."""
    
    name = "Ollama Itinerary Generator"
    description = "Generate day-by-day itineraries for Morocco based on preferences and constraints"
    
    def __init__(self, model_name="llama2"):
        """Initialize with Ollama model."""
        super().__init__()
        self.model_name = model_name
        
        try:
            # Test connection to Ollama
            ollama.list()
            self.initialized = True
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            self.initialized = False
    
    def _run(self, query: str) -> str:
        """Generate an itinerary using Ollama."""
        if not self.initialized:
            return "Error: Ollama not initialized. Make sure Ollama is installed and running."
        
        try:
            from data.morocco_info import MOROCCO_DESTINATIONS, TRANSPORT_OPTIONS
            
            # Extract duration from query (fallback to 7 days)
            duration = 7
            query_lower = query.lower()
            for i in range(1, 31):  # Check for 1-30 day trips
                if f"{i} day" in query_lower or f"{i}-day" in query_lower:
                    duration = i
                    break
            
            # Extract destinations
            mentioned_destinations = []
            for dest in MOROCCO_DESTINATIONS.keys():
                if dest.lower() in query_lower:
                    mentioned_destinations.append(dest)
            
            # If no destinations specified, use placeholder text
            destinations_text = "the mentioned destinations" if mentioned_destinations else "major destinations in Morocco"
            
            # Create transport info
            transport_json = json.dumps(TRANSPORT_OPTIONS, indent=2)
            
            prompt = f"""Generate a {duration}-day Morocco travel itinerary based on this query:
"{query}"

When creating this itinerary:
1. Organize it by day with specific activities for morning, afternoon, and evening
2. Include specific attractions, sites, and experiences from {destinations_text}
3. Consider logistics and reasonable travel times between locations
4. Include practical advice for transportation using this information:
{transport_json}

Format the itinerary as Markdown with day-by-day headings and bullet points for activities.
Include a brief section at the end with practical tips for traveling in Morocco.
"""
            
            response = ollama.generate(
                model=self.model_name,
                prompt=prompt,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "num_predict": 1500  # Longer content for detailed itinerary
                }
            )
            
            return response['response'].strip()
        
        except Exception as e:
            return f"Error generating itinerary with Ollama: {str(e)}"
