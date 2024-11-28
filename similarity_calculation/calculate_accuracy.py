import pandas as pd

# Function to sort the data
def sort_lumbar_vertebra(data):
    # Extract numeric part of the Lumbar_Image to assist in sorting
    data['Lumbar_Image_num'] = data['Lumbar_Image'].str.extract('(\d+)').astype(int)

    # Sort Vertebra based on 'L1' to 'L5', 'S1' ordering, using custom sorting
    vertebra_order = ['L1', 'L2', 'L3', 'L4', 'L5', 'S1']
    data['Vertebra_num'] = pd.Categorical(data['Vertebra'], categories=vertebra_order, ordered=True)

    # Sorting the DataFrame
    sorted_data = data.sort_values(by=['Lumbar_Image_num', 'Vertebra_num']).drop(columns=['Lumbar_Image_num', 'Vertebra_num'])
    return sorted_data

# Function to calculate accuracy
def calculate_accuracy(data):
    # Define total images
    total_images = 17
    
    # Filter rows where Vertebra is 'L5'
    l5_data = data[data['Vertebra'] == 'L5']
    
    # Calculate how many of these have Most_Similar == True
    correct_predictions = l5_data['Most_Similar'].sum()  # since 'Most_Similar' is boolean, summing gives the count of True values
    
    # Calculate accuracy
    accuracy = correct_predictions / total_images
    return accuracy

# Load the CSV files
whole_similarity_file = 'output\every_bone_similarity\\all_overlap_region\sorted_whole_similarity.csv'
box_similarity_file = 'output\every_bone_similarity\\box_region\sorted_box_similarity.csv'
consistent_width_similarity_file = 'output\every_bone_similarity\consistent_width\sorted_consistent_width_similarity.csv'

# Load the data
whole_similarity_data = pd.read_csv(whole_similarity_file)
box_similarity_data = pd.read_csv(box_similarity_file)
consistent_width_similarity_data = pd.read_csv(consistent_width_similarity_file)

# Sort the data
sorted_whole_similarity_data = sort_lumbar_vertebra(whole_similarity_data)
sorted_box_similarity_data = sort_lumbar_vertebra(box_similarity_data)
sorted_consistent_width_similarity_data = sort_lumbar_vertebra(consistent_width_similarity_data)

# Calculate accuracies
whole_similarity_accuracy = calculate_accuracy(sorted_whole_similarity_data)
box_similarity_accuracy = calculate_accuracy(sorted_box_similarity_data)
consistent_width_similarity_accuracy = calculate_accuracy(sorted_consistent_width_similarity_data)

# Print results
print(f"Whole Similarity Accuracy: {whole_similarity_accuracy:.2f}")
print(f"Box Similarity Accuracy: {box_similarity_accuracy:.2f}")
print(f"Consistent Width Similarity Accuracy: {consistent_width_similarity_accuracy:.2f}")

# Add source labels to identify the data origin
sorted_whole_similarity_data['Source'] = 'Whole Similarity'
sorted_box_similarity_data['Source'] = 'Box Similarity'
sorted_consistent_width_similarity_data['Source'] = 'consistent width Similarity'

# Combine all datasets into one
combined_data = pd.concat([sorted_whole_similarity_data, sorted_box_similarity_data, sorted_consistent_width_similarity_data], ignore_index=True)

# Save the combined data to a new CSV file
combined_data.to_csv('output\every_bone_similarity\\17_combined_similarity_data.csv', index=False)
