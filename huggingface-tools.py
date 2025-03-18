"""
Tools for CrewAI agents using Hugging Face models.
"""

import json
from typing import List, Dict, Any
from langchain.tools import BaseTool
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

class DeepSeekTravelAdvisorTool(BaseTool):
    """Tool that uses DeepSeek model for travel recommendations."""
    
    name = "DeepSeek Travel Advisor"
    description = "Generate travel advice, recommendations, and itineraries for Morocco"
    
    def __init__(self, model_name="deepseek-ai/deepseek-coder-6.7b-base"):
        """Initialize with DeepSeek model."""
        super().__init__()
        # Check for GPU availability
        device = 0 if torch.cuda.is_available() else -1
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        try:
            # Load the model with optimal settings
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map="auto" if torch.cuda.is_available() else None
            )
            self.initialized = True
        except Exception as e:
            print(f"Error initializing DeepSeek model: {e}")
            # Fallback to text-generation pipeline with a smaller model
            try:
                self.pipeline = pipeline(
                    "text-generation",
                    model="gpt2",
                    device=device
                )
                self.initialized = False
                print("Falling back to GPT-2 model")
            except Exception as e2:
                print(f"Error initializing fallback model: {e2}")
                self.initialized = False
    
    def _run(self, query: str) -> str:
        """Execute the model on the provided query."""
        try:
            prompt = f"""You are a travel advisor specializing in Morocco tourism. 
            Provide detailed, helpful information about the following query:
            
            {query}
            
            Focus on accurate, practical advice that would help travelers have an authentic experience.
            Include specific locations, cultural insights, and practical tips where relevant.
            """
            
            if self.initialized:
                # Use the DeepSeek model directly
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs, 
                    max_new_tokens=500,
                    temperature=0.7,
                    do_sample=True
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Extract just the generated part (not the prompt)
                response = response[len(prompt):]
            else:
                # Use the fallback pipeline
                response = self.pipeline(
                    prompt,
                    max_length=len(prompt) + 500,
                    temperature=0.7,
                    num_return_sequences=1
                )[0]["generated_text"]
                response = response[len(prompt):]
                
            return response.strip()
        
        except Exception as e:
            return f"Error generating travel advice: {str(e)}"

class DestinationMatcherTool(BaseTool):
    """Tool that matches traveler preferences to Morocco destinations."""
    
    name = "Morocco Destination Matcher"
    description = "Match traveler preferences to suitable destinations in Morocco"
    
    def __init__(self):
        """Initialize with a text classification model."""
        super().__init__()
        # Use a smaller sentiment model for preference analysis
        try:
            self.classifier = pipeline(
                "text-classification",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
            self.initialized = True
        except Exception as e:
            print(f"Error initializing classifier: {e}")
            self.initialized = False
    
    def _run(self, preferences: str) -> str:
        """Match preferences to destinations."""
        try:
            from data.morocco_info import MOROCCO_DESTINATIONS
            
            # Extract key preferences
            preferences_lower = preferences.lower()
            
            # Define preference categories
            categories = {
                "beach": ["beach", "coast", "ocean", "sea", "surf", "wind"],
                "culture": ["culture", "history", "museum", "architecture", "tradition", "art"],
                "nature": ["nature", "mountain", "hike", "trek", "landscape", "outdoor"],
                "desert": ["desert", "sand", "dune", "sahara", "camel", "star"],
                "urban": ["city", "urban", "modern", "shopping", "restaurant", "cafe"],
                "relaxation": ["relax", "peaceful", "quiet", "calm", "retreat", "rest"]
            }
            
            # Score each category based on preferences
            category_scores = {}
            for category, keywords in categories.items():
                score = sum(10 if keyword in preferences_lower else 0 for keyword in keywords)
                if self.initialized:
                    # Add sentiment analysis for relevant sentences
                    for sentence in preferences.split('.'):
                        if any(keyword in sentence.lower() for keyword in keywords):
                            sentiment = self.classifier(sentence)[0]
                            sentiment_boost = 5 if sentiment['label'] == 'POSITIVE' else -5
                            score += sentiment_boost
                
                category_scores[category] = max(0, score)  # Ensure no negative scores
            
            # Match destinations to preference categories
            destination_matches = {}
            for dest, info in MOROCCO_DESTINATIONS.items():
                score = 0
                desc_lower = info["description"].lower()
                
                # Match based on description keywords
                for category, category_score in category_scores.items():
                    if category_score > 0:
                        keywords = categories[category]
                        if any(keyword in desc_lower for keyword in keywords):
                            score += category_score
                
                # Additional scoring based on highlights
                for highlight in info["highlights"]:
                    highlight_lower = highlight.lower()
                    for category, keywords in categories.items():
                        if any(keyword in highlight_lower for keyword in keywords):
                            score += category_scores.get(category, 0) * 0.5
                
                destination_matches[dest] = score
            
            # Sort destinations by score
            sorted_destinations = sorted(
                destination_matches.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            # Format the response
            response = "Based on your preferences, here are recommended destinations in Morocco:\n\n"
            
            for dest, score in sorted_destinations[:3]:  # Top 3 matches
                info = MOROCCO_DESTINATIONS[dest]
                response += f"🌟 **{dest}** (Match score: {score})\n"
                response += f"- {info['description']}\n"
                response += f"- Highlights: {', '.join(info['highlights'][:2])}\n"
                response += f"- Best time to visit: {info['best_time']}\n\n"
            
            return response
        
        except Exception as e:
            return f"Error matching destinations: {str(e)}"

class ItineraryGeneratorTool(BaseTool):
    """Tool that generates travel itineraries for Morocco."""
    
    name = "Morocco Itinerary Generator"
    description = "Generate day-by-day itineraries for Morocco based on preferences and constraints"
    
    def __init__(self, model_name="deepseek-ai/deepseek-coder-6.7b-base"):
        """Initialize with DeepSeek model."""
        super().__init__()
        try:
            # Use a smaller model for itinerary generation since it's structured
            self.generator = pipeline(
                "text-generation",
                model="gpt2-medium",
                device=0 if torch.cuda.is_available() else -1
            )
            self.initialized = True
        except Exception as e:
            print(f"Error initializing itinerary generator: {e}")
            self.initialized = False
    
    def _run(self, query: str) -> str:
        """Generate an itinerary based on query details."""
        try:
            # Parse the query for key parameters
            query_lower = query.lower()
            
            # Extract duration (fallback to 7 days)
            duration = 7
            for i in range(1, 31):  # Check for 1-30 day trips
                if f"{i} day" in query_lower or f"{i}-day" in query_lower:
                    duration = i
                    break
            
            # Extract destinations
            from data.morocco_info import MOROCCO_DESTINATIONS
            destinations = []
            for dest in MOROCCO_DESTINATIONS.keys():
                if dest.lower() in query_lower:
                    destinations.append(dest)
            
            # If no destinations specified, assume it's a general Morocco trip
            if not destinations:
                destinations = ["Marrakech", "Fes", "Chefchaouen"]
            
            # Create a structured itinerary
            itinerary = self._generate_structured_itinerary(destinations, duration, query)
            return itinerary
        
        except Exception as e:
            return f"Error generating itinerary: {str(e)}"
    
    def _generate_structured_itinerary(self, destinations, duration, preferences):
        """Generate a structured day-by-day itinerary."""
        from data.morocco_info import MOROCCO_DESTINATIONS, TRANSPORT_OPTIONS
        
        # Distribute days among destinations based on importance
        # Simple algorithm: allocate days proportionally with a minimum of 1
        total_destinations = len(destinations)
        days_per_destination = {}
        
        remaining_days = duration
        for dest in destinations:
            days_per_destination[dest] = 1
            remaining_days -= 1
        
        # Distribute remaining days
        while remaining_days > 0:
            for dest in destinations:
                if remaining_days > 0:
                    days_per_destination[dest] += 1
                    remaining_days -= 1
                else:
                    break
        
        # Generate the actual itinerary
        itinerary = "# Your Personalized Morocco Itinerary\n\n"
        
        day = 1
        current_location = None
        
        for dest in destinations:
            if current_location:
                # Add travel day between destinations
                transport = self._suggest_transport(current_location, dest)
                itinerary += f"## Day {day}: Travel from {current_location} to {dest}\n"
                itinerary += f"- Morning: Check out from your accommodation in {current_location}\n"
                itinerary += f"- Transportation: {transport}\n"
                itinerary += f"- Afternoon: Arrive in {dest}, check in to your accommodation\n"
                itinerary += f"- Evening: Relaxed exploration near your accommodation, enjoy local cuisine\n\n"
                day += 1
            
            # Add days for this destination
            days_here = days_per_destination[dest]
            dest_info = MOROCCO_DESTINATIONS[dest]
            highlights = dest_info["highlights"]
            
            # First day in the destination
            itinerary += f"## Day {day}: {dest} - Orientation\n"
            itinerary += f"- Morning: Breakfast and orientation walk around {dest}\n"
            
            if len(highlights) > 0:
                itinerary += f"- Afternoon: Visit {highlights[0]}\n"
            
            itinerary += f"- Evening: Dinner at a local restaurant, experience {dest} by night\n\n"
            day += 1
            
            # Middle days
            highlights_index = 1
            for d in range(1, days_here - 1):
                itinerary += f"## Day {day}: {dest} - Exploration\n"
                
                if highlights_index < len(highlights):
                    itinerary += f"- Morning: Visit {highlights[highlights_index]}\n"
                    highlights_index += 1
                else:
                    itinerary += f"- Morning: Free time to explore {dest} at your own pace\n"
                
                if highlights_index < len(highlights):
                    itinerary += f"- Afternoon: Visit {highlights[highlights_index]}\n"
                    highlights_index += 1
                else:
                    itinerary += f"- Afternoon: Optional activities or shopping in local markets\n"
                
                itinerary += "- Evening: Dinner and relaxation\n\n"
                day += 1
            
            # Last day in the destination
            if days_here > 1:
                itinerary += f"## Day {day}: {dest} - Final Day\n"
                itinerary += f"- Morning: Last-minute sightseeing or shopping in {dest}\n"
                itinerary += "- Afternoon: Relaxation or optional activities\n"
                itinerary += f"- Evening: Farewell dinner in {dest}\n\n"
                day += 1
            
            current_location = dest
        
        # Add practical information
        itinerary += "## Practical Information\n\n"
        itinerary += "### Accommodation Suggestions\n"
        for dest in destinations:
            itinerary += f"- **{dest}**: Traditional riad in the medina or modern hotel based on preference\n"
        
        itinerary += "\n### Transportation Tips\n"
        for mode, info in list(TRANSPORT_OPTIONS.items())[:3]:
            itinerary += f"- **{mode.replace('_', ' ').title()}**: {info['description']}\n"
        
        from data.morocco_info import CULTURAL_TIPS, SAFETY_TIPS
        itinerary += "\n### Cultural Tips\n"
        for tip in CULTURAL_TIPS[:4]:
            itinerary += f"- {tip}\n"
        
        itinerary += "\n### Safety Tips\n"
        for tip in SAFETY_TIPS[:3]:
            itinerary += f"- {tip}\n"
        
        return itinerary
    
    def _suggest_transport(self, origin, destination):
        """Suggest transport options between two destinations."""
        # This is a simplified version - in reality, you'd need more data
        long_distance_pairs = [
            ("Marrakech", "Tangier"),
            ("Casablanca", "Merzouga"),
            ("Fes", "Essaouira"),
            ("Rabat", "Merzouga")
        ]
        
        # Check if either combination exists (order doesn't matter)
        is_long_distance = (origin, destination) in long_distance_pairs or (destination, origin) in long_distance_pairs
        
        if is_long_distance:
            return "Domestic flight (recommended) or overnight train where available"
        else:
            return "Train or CTM bus (both comfortable options with regular departures)"
