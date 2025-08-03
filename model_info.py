import tensorflow as tf
import pandas as pd
import numpy as np

def analyze_model():
    """Analyze the loaded model to understand its structure and expected inputs."""
    try:
        # Load the model
        model = tf.keras.models.load_model('tree_species_model.h5')
        
        print("=== MODEL ANALYSIS ===")
        print(f"Model type: {type(model)}")
        print(f"Number of layers: {len(model.layers)}")
        
        # Get input shape
        input_shape = model.input_shape
        print(f"Input shape: {input_shape}")
        
        # Get output shape
        output_shape = model.output_shape
        print(f"Output shape: {output_shape}")
        
        # Get number of classes
        num_classes = output_shape[1] if len(output_shape) > 1 else 1
        print(f"Number of output classes: {num_classes}")
        
        # Print model summary
        print("\n=== MODEL SUMMARY ===")
        model.summary()
        
        # Try to understand what features the model expects
        print("\n=== FEATURE ANALYSIS ===")
        print("Based on the dataset structure, your model likely expects these features:")
        print("- latitude_coordinate")
        print("- longitude_coordinate") 
        print("- diameter_breast_height_CM")
        print("- height_M")
        print("- condition (encoded)")
        print("- native (encoded)")
        print("- city (encoded)")
        
        # Check if we can make a test prediction
        print("\n=== TEST PREDICTION ===")
        try:
            # Create a sample input based on the input shape
            if input_shape[1] is not None:  # If we know the number of features
                sample_input = np.random.random((1, input_shape[1]))
                prediction = model.predict(sample_input)
                print(f"Test prediction shape: {prediction.shape}")
                print(f"Sample prediction: {prediction[0][:5]}...")  # Show first 5 values
            else:
                print("Input shape is not fully defined")
        except Exception as e:
            print(f"Test prediction failed: {e}")
            
    except Exception as e:
        print(f"Error loading model: {e}")

if __name__ == "__main__":
    analyze_model() 