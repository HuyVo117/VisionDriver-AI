"""Knowledge base for Phase 1 driving Q&A."""

from typing import Dict, Any


def build_knowledge_base() -> Dict[str, Any]:
    return {
        "parking_rules": {
            "no_parking_sign": "No parking allowed at this location.",
            "disabled_parking": "Reserved for disabled permit vehicles.",
            "loading_zone": "Loading and unloading zone only.",
        },
        "speed_limits": {
            "residential": 50,
            "urban": 60,
            "highway": 100,
        },
        "traffic_signs": {
            "8_2_1": "Mandatory straight ahead",
            "8_2_2": "Mandatory right turn",
            "3_1": "No entry",
        },
        "camera_locations": {
            "highway_km_100": "Traffic camera near km 100+500m",
        },
    }
