import logging
from typing import Dict
from datetime import datetime

class OutputFormatter:
    def __init__(self):
        self.timings: Dict[str, float] = {}
        
    def add_timing(self, operation: str, duration: float):
        self.timings[operation] = duration
    
    def print_summary(self):
        if self.timings:
            print("\n🕒 Performance Summary:")
            print("=" * 50)
            total_time = 0
            for operation, duration in self.timings.items():
                total_time += duration
                print(f"  • {operation}: {duration:.2f}s")
            print("-" * 50)
            print(f"  Total Time: {total_time:.2f}s")
            print("=" * 50)
            
    @staticmethod
    def print_startup_ideas(ideas: str):
        print("\n🚀 Generated Startup Ideas")
        print("=" * 50)
        # Split the ideas and format each one
        ideas_list = ideas.split("==================================================")
        for idea in ideas_list:
            if idea.strip():
                print(idea.strip())
                print("=" * 50)

# Global formatter instance
formatter = OutputFormatter()
