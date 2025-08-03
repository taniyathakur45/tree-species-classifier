import pandas as pd
import numpy as np

def create_species_mapping():
    """Create a mapping from species ID to tree name based on the dataset."""
    try:
        # Load the dataset
        df = pd.read_csv('tree_species_dataset.csv')
        
        # Get unique species and their counts
        species_counts = df['common_name'].value_counts()
        
        # Create mapping from index to species name
        species_mapping = {}
        for idx, (species_name, count) in enumerate(species_counts.items()):
            species_mapping[idx] = {
                'name': species_name,
                'count': count,
                'scientific_name': df[df['common_name'] == species_name]['scientific_name'].iloc[0] if not df[df['common_name'] == species_name]['scientific_name'].isna().all() else 'Unknown'
            }
        
        print(f"Created mapping for {len(species_mapping)} species")
        
        # Save the mapping
        import json
        with open('species_mapping.json', 'w') as f:
            json.dump(species_mapping, f, indent=2)
        
        print("Species mapping saved to species_mapping.json")
        
        # Display some examples
        print("\nSample species mapping:")
        for i in range(min(10, len(species_mapping))):
            species = species_mapping[i]
            print(f"ID {i}: {species['name']} ({species['scientific_name']}) - {species['count']} trees")
            
        return species_mapping
        
    except Exception as e:
        print(f"Error creating species mapping: {e}")
        return None

if __name__ == "__main__":
    create_species_mapping() 