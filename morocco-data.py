"""
Static data about Morocco destinations for the trip planner.
"""

MOROCCO_DESTINATIONS = {
    "Marrakech": {
        "description": "The Red City known for its vibrant souks, historic medina, and Jamaa el-Fna square.",
        "highlights": ["Jardin Majorelle", "Bahia Palace", "Koutoubia Mosque", "Medina of Marrakech"],
        "climate": "Hot desert climate with very hot summers and mild winters.",
        "best_time": "March to May and September to November",
        "avg_cost_per_day": "$50-100"
    },
    "Casablanca": {
        "description": "Morocco's largest city and economic capital with a mix of modern architecture and old medina.",
        "highlights": ["Hassan II Mosque", "Corniche", "Morocco Mall", "Old Medina"],
        "climate": "Mediterranean climate with mild, wet winters and warm, dry summers.",
        "best_time": "April to October",
        "avg_cost_per_day": "$60-120"
    },
    "Fes": {
        "description": "Cultural capital with the oldest university in the world and a well-preserved medieval medina.",
        "highlights": ["Fes El Bali (Old Medina)", "Al-Qarawiyyin Mosque", "Bou Inania Madrasa", "Tanneries"],
        "climate": "Mediterranean with continental influences, hot summers and cold winters.",
        "best_time": "March to May and September to November",
        "avg_cost_per_day": "$45-90"
    },
    "Chefchaouen": {
        "description": "The Blue Pearl of Morocco, known for its striking blue-washed buildings.",
        "highlights": ["Blue Medina", "Kasbah Museum", "Ras El Ma (Water Spring)", "Spanish Mosque viewpoint"],
        "climate": "Mediterranean mountain climate with mild winters and warm summers.",
        "best_time": "April to June and September to October",
        "avg_cost_per_day": "$40-80"
    },
    "Essaouira": {
        "description": "Coastal town known for its windy beaches, fishing port, and Portuguese fortifications.",
        "highlights": ["Medina of Essaouira", "Skala de la Ville", "Port of Essaouira", "Essaouira Beach"],
        "climate": "Mild Mediterranean climate tempered by ocean breezes.",
        "best_time": "April to October",
        "avg_cost_per_day": "$45-85"
    },
    "Merzouga": {
        "description": "Desert town at the edge of the Sahara, known for Erg Chebbi sand dunes.",
        "highlights": ["Erg Chebbi Dunes", "Camel Trekking", "Desert Camping", "Khamlia Village"],
        "climate": "Hot desert climate with extreme temperature variations between day and night.",
        "best_time": "October to April",
        "avg_cost_per_day": "$50-100"
    },
    "Rabat": {
        "description": "Morocco's capital city, known for its Islamic and French-colonial heritage.",
        "highlights": ["Kasbah of the Udayas", "Hassan Tower", "Chellah Necropolis", "Royal Palace"],
        "climate": "Mediterranean climate with mild winters and warm summers.",
        "best_time": "April to June and September to November",
        "avg_cost_per_day": "$55-110"
    },
    "Tangier": {
        "description": "Port city at the Strait of Gibraltar with a rich international history.",
        "highlights": ["Kasbah Museum", "Caves of Hercules", "Cap Spartel", "Tangier American Legation Museum"],
        "climate": "Mediterranean climate with oceanic influences.",
        "best_time": "May to October",
        "avg_cost_per_day": "$50-100"
    }
}

TRANSPORT_OPTIONS = {
    "domestic_flights": {
        "description": "Several domestic airports connect major cities",
        "pros": "Fast for long distances",
        "cons": "More expensive, limited schedules",
        "approximate_cost": "$60-120 one way"
    },
    "trains": {
        "description": "ONCF trains connect major cities with modern services",
        "pros": "Comfortable, reliable, scenic routes",
        "cons": "Limited network, may sell out during peak times",
        "approximate_cost": "$10-30 one way"
    },
    "grand_taxis": {
        "description": "Shared Mercedes taxis that operate between cities",
        "pros": "Affordable, frequent departures",
        "cons": "Can be crowded, no fixed schedule",
        "approximate_cost": "$5-15 per person"
    },
    "buses": {
        "description": "Extensive network with companies like CTM and Supratours",
        "pros": "Widespread coverage, affordable",
        "cons": "Variable comfort levels, can be slow",
        "approximate_cost": "$5-20 one way"
    },
    "car_rental": {
        "description": "Available in major cities and airports",
        "pros": "Freedom to explore, convenient",
        "cons": "Challenging driving conditions, parking difficulties in medinas",
        "approximate_cost": "$30-60 per day plus fuel"
    }
}

CULTURAL_TIPS = [
    "Dress modestly, especially when visiting religious sites.",
    "Learn basic Arabic or French phrases to connect with locals.",
    "Haggling is expected in souks but should be done respectfully.",
    "Friday is the Muslim holy day when many businesses may close early.",
    "Ramadan affects opening hours and food availability during the day.",
    "Always ask permission before photographing people.",
    "Remove shoes before entering someone's home or certain areas in riads.",
    "Tipping (around 10%) is customary for services."
]

SAFETY_TIPS = [
    "Morocco is generally safe but stay alert in crowded areas.",
    "Use registered guides for desert excursions and mountain treks.",
    "Drink bottled water to avoid stomach issues.",
    "Keep valuables secure and be wary of common scams targeting tourists.",
    "Have a copy of your passport and important documents.",
    "Use official taxis with meters or agree on prices beforehand.",
    "Register with your embassy if staying for an extended period.",
    "In summer, take precautions against extreme heat, especially in desert regions."
]

def get_morocco_info():
    """Return comprehensive Morocco travel information."""
    return {
        "destinations": MOROCCO_DESTINATIONS,
        "transportation": TRANSPORT_OPTIONS,
        "cultural_tips": CULTURAL_TIPS,
        "safety_tips": SAFETY_TIPS
    }
