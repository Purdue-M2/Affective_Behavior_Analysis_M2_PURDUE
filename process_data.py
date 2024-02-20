import os



import os
###############################generate labels#########################################
# # Base directory where folders are located
# base_dir = 'data/cropped_aligned'

# # Output directory for the generated text files
# output_dir = 'labels'
# os.makedirs(output_dir, exist_ok=True)

# # # Emotion categories in the order they appear in the annotation files
# # emotions = ["Neutral", "Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise", "Other"]
# # Define the base directories
# image_root = 'data/cropped_aligned'
# annotation_root = 'data/6th ABAW Annotations/EXPR_Recognition_Challenge/Validation_Set'

# # Output directory for the generated text files
# output_dir = 'labels/val'
# os.makedirs(output_dir, exist_ok=True)

# # # Iterate over each annotation file in the Train_Set directory
# for annotation_file in os.listdir(annotation_root):
#     if annotation_file.endswith('.txt'):
#         # Extract the file name without extension to match with image folder
#         base_name = os.path.splitext(annotation_file)[0]
#         annotation_file_path = os.path.join(annotation_root, annotation_file)
#         image_folder_path = os.path.join(image_root, base_name)
#         # print(annotation_file_path)
#         # print(image_folder_path)
#         # Check if the corresponding image folder exists
#         if not os.path.exists(image_folder_path):
#             print(f"Image folder for {base_name} does not exist, skipping...")
#             continue

#         # # Read annotations, skipping the header
#         with open(annotation_file_path, 'r') as file:
#             next(file)  # Skip the header line
#             annotations = file.read().strip().split('\n')
        
#         # # Ensure there's an output file for each image folder
#         output_file_path = os.path.join(output_dir, base_name + '_annotations.txt')
#         with open(output_file_path, 'w') as output_file:
#             # Iterate over each annotation and corresponding image
#             for i, annotation in enumerate(annotations):
#                 # Assuming image files follow a sequential naming convention
#                 # Modify this as per your actual image file naming pattern
#                 image_name = f"{i+1:05d}.jpg" # Example: 1.jpg, 2.jpg, etc.
#                 image_path = os.path.join(image_folder_path, image_name)
                
#                 # Check if the image exists before writing to output
#                 if os.path.exists(image_path):
#                     # Write image path and corresponding annotation to the output file
#                     output_file.write(f"{image_path}\t{annotation}\n")
#                 else:
#                     print(f"Image {image_name} not found in folder {base_name}, skipping...")

#################################### Merge ##################################
# Directory containing the generated text files
# output_dir = 'AU_labels/train'

# # File to store the merged content
# merged_file_path = os.path.join(output_dir, 'AU_merged_train_annotations.txt')

# # Open the file for writing
# with open(merged_file_path, 'w') as merged_file:
#     # Iterate over each file in the output directory
#     for filename in os.listdir(output_dir):
#         file_path = os.path.join(output_dir, filename)
        
#         # Skip if it's not a file
#         if not os.path.isfile(file_path):
#             continue
        
#         # Skip the merged file itself if it's already present
#         if filename == 'AU_merged_train_annotations.txt':
#             continue

#         # Open each file and append its contents to the merged file
#         with open(file_path, 'r') as file:
#             for line in file:
#                 merged_file.write(line)

# print(f"All text files have been merged into {merged_file_path}")

# import os

# # Define the path to the merged_annotations.txt file
# merged_file_path = 'merged_val_annotations.txt'

# # Define the path for the new file to create
# new_file_path = 'expr_val.txt'

# # Open the merged file for reading and the new file for writing
# with open(merged_file_path, 'r') as merged_file, open(new_file_path, 'w') as new_file:
#     # Iterate over each line in the merged file
#     for line in merged_file:
#         # Check if the line ends with -1 (considering newline characters)
#         if not line.strip().endswith('-1'):
#             # Write the line to the new file if it doesn't end with -1
#             new_file.write(line)

# print(f"Filtered content has been written to {new_file_path}")


############################################################### AU ##########################
# import os

# image_root = 'data/cropped_aligned'
# annotation_root = 'data/6th ABAW Annotations/AU_Detection_Challenge/Validation_Set'
# output_dir = 'AU_labels/val'
# os.makedirs(output_dir, exist_ok=True)

# # Define AU names if you want to include them in the output
# au_names = ["AU1", "AU2", "AU4", "AU6", "AU7", "AU10", "AU12", "AU15", "AU23", "AU24", "AU25", "AU26"]

# # Iterate over each annotation file in the Validation_Set directory
# for annotation_file in os.listdir(annotation_root):
#     if annotation_file.endswith('.txt'):
#         base_name = os.path.splitext(annotation_file)[0]
#         annotation_file_path = os.path.join(annotation_root, annotation_file)
#         image_folder_path = os.path.join(image_root, base_name)
        
#         if not os.path.exists(image_folder_path):
#             print(f"Image folder for {base_name} does not exist, skipping...")
#             continue

#         # Open the annotation file and skip the header
#         with open(annotation_file_path, 'r') as file:
#             next(file)  # Skip the header line
#             annotations = file.read().strip().split('\n')
        
#         # Open the output file for writing
#         output_file_path = os.path.join(output_dir, base_name + '_annotations.txt')
#         with open(output_file_path, 'w') as output_file:
#             # Iterate over each annotation and corresponding image
#             for i, annotation in enumerate(annotations):
#                 image_name = f"{i+1:05d}.jpg"
#                 image_path = os.path.join(image_folder_path, image_name)
                
#                 # Check if the image exists before writing to output
#                 if os.path.exists(image_path):
#                     # Write image path and corresponding annotation to the output file
#                     output_file.write(f"{image_path}\t{annotation}\n")
#                 else:
#                     print(f"Image {image_name} not found in folder {base_name}, skipping...")

########################################  remove -1 #############################################
# Define the path of the original text file and the new file to save the filtered data
input_file_path = 'AU_merged_val_annotations.txt'
output_file_path = 'au_val.txt'

# Open the original file to read and the new file to write
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    # Iterate through each line in the original file
    for line in input_file:
        # Split the line into image path and labels, assuming the first split is the path, and the rest are labels
        parts = line.strip().split('\t')
        if len(parts) < 2:
            print('!!!!!!!!!!!')
            continue  # Skip lines that don't have the expected format
        image_path, labels_str = parts[0], parts[1]
        labels = labels_str.split(',')  # Adjust this if labels are not separated by commas

        # Check if "-1" is present in the labels
        if '-1' not in labels:  # This ensures we are checking within the labels only
            # If "-1" is not found, write the line to the new file
            output_file.write(line)

print(f'Filtered data saved to {output_file_path}')