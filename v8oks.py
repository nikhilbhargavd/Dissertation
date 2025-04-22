import numpy as np

def extract_oks_score(metrics_file):
    """
    Extracts the OKS score from a SLEAP metrics .npz file.
    
    Args:
        metrics_file (str): Path to the metrics.val.npz file.
    
    Returns:
        float: The OKS score if found, otherwise None.
    """
    try:
        # Load the metrics file
        data = np.load(metrics_file, allow_pickle=True)

        # Print all available keys
        print("Available keys:", data.files)

        # Extract the 'metrics' dictionary
        metrics_data = data.get("metrics", None)

        if metrics_data is not None:
            print(f"Metrics data type: {type(metrics_data)}")

            # If it's a dictionary or structured array, check its contents
            if isinstance(metrics_data, dict):
                print("Keys in 'metrics':", metrics_data.keys())

                # Look for OKS in the dictionary
                oks_score = metrics_data.get("oks", None)

                if oks_score is not None:
                    print(f"OKS Score: {oks_score}")
                    return oks_score
                else:
                    print("'oks' key not found in 'metrics'.")
                    return None
            
            # If metrics is a numpy structured array, print fields
            elif isinstance(metrics_data, np.ndarray):
                print("Structured array fields:", metrics_data.dtype.names)

                # Check if 'oks' is a field in the structured array
                if "oks" in metrics_data.dtype.names:
                    oks_score = metrics_data["oks"]
                    print(f"OKS Score: {oks_score}")
                    return oks_score
                else:
                    print("'oks' not found in structured array.")
                    return None

        print("No valid OKS data found.")
        return None

    except Exception as e:
        print(f"Error loading file: {e}")
        return None

# File path
metrics_path = r"C:\Users\nikhi\LabProject\new\v8\v8metrics.val.npz"

# Extract OKS score
extract_oks_score(metrics_path)
