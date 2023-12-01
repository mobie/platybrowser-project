from skimage import io, measure
import pandas as pd

# Load the 3D label mask image
image = io.imread('/Users/tischer/Desktop/platy-nuclei.tif')  # Make sure to provide the correct path to your file

# Measure the properties of the labeled regions
properties = measure.regionprops_table(image, properties=('label', 'centroid'))

# Convert the results to a DataFrame
df = pd.DataFrame(properties)

# Rename the columns to match the desired output
df.rename(columns={'label': 'spot_id', 'centroid-0': 'z', 'centroid-1': 'y', 'centroid-2': 'x'}, inplace=True)

# Apply the scaling to the coordinates
df['x'] *= 0.32  # Scaling for x
df['y'] *= 0.32  # Scaling for y
df['z'] *= 0.4   # Scaling for z

# Save the DataFrame as a TSV file
tsv_string = df.to_csv(sep='\t', index=False)

string_test = tsv_string[:50]

for char in string_test:
    if char == '\t':
        print('\\t', end='')  # Print \t for tabs
    else:
        print(char, end='')  # Print the character as is

# Save the string to a file
with open('centroids.tsv', 'w') as file:
    file.write(tsv_string)

#df.to_csv('centroids.tsv', sep='\t', index=False)

