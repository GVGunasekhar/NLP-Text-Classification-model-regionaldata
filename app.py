import spacy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import pandas as pd

# Load spaCy's small English model
nlp = spacy.load('en_core_web_sm')

# Load data from Excel
dataframe = pd.read_csv(r'C:\Users\swamy.LAPTOP-0RVCTRRS\OneDrive\Desktop\Text-Classification -NLP-Model\REGIONAL DATASET.csv')


# Print column names for debugging
print("Column names in the DataFrame:", dataframe.columns)

# Strip any whitespace from column names
dataframe.columns = dataframe.columns.str.strip()

data = {}

# Populate the data dictionary from the Excel sheet
for index, row in dataframe.iterrows():
    district = row['Districts']  # Correct column name
    cities = [city.strip().lower() for city in row['Cities/Towns/Cities'].split(',')]  # Corrected column name
    data[district] = cities

# Initialize an empty DataFrame to hold user input categorized by district
full_df = pd.DataFrame(columns=["User Input", "City/Town", "Predicted District"])

# Function to detect named entities (NER) and extract city/town names
def detect_entity(input_text):
    input_text = input_text.lower()  # Convert input to lowercase
    input_doc = nlp(input_text)
    for ent in input_doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            return ent.text
    return None

# Function to detect city/town using fuzzy matching and spaCy NER
def detect_city(input_text):
    input_text = input_text.lower()  # Convert input to lowercase
    city_town = detect_entity(input_text)

    if not city_town:  # If NER doesn't find anything, use fuzzy matching
        all_towns = [town.lower() for towns in data.values() for town in towns]  # Convert towns to lowercase
        match = process.extractOne(input_text, all_towns, scorer=fuzz.token_set_ratio)
        if match:  # Check if match is not None
            city_town, score = match
            if score < 80:  # Setting a threshold for matching accuracy
                city_town = None

    return city_town

# Function to find the corresponding district
def find_district(city_town):
    if city_town:
        city_town = city_town.lower()  # Convert city/town to lowercase for matching
        for district, towns in data.items():
            if city_town in towns:
                return district
    return None

# Take a single user input
user_input = input("Enter your details or describe something about a place: ")

# Limit the input to the first 70 characters for detection (you can change this value)
limited_input = user_input[:70]

# Detect city/town and predicted district using limited input
city_town = detect_city(limited_input)
predicted_district = find_district(city_town) if city_town else None

# Store the input and results in the DataFrame
new_entry = {
    "User Input": user_input,  # Store the full input
    "City/Town": city_town,
    "Predicted District": predicted_district
}
full_df = pd.concat([full_df, pd.DataFrame([new_entry])], ignore_index=True)

if predicted_district:
    print(f"User input categorized under District: {predicted_district}")
else:
    print("District not found.")

# Function to save the DataFrame to an Excel file
def save_to_excel(predicted_district):
    # Use the predicted district for the file name, if available
    if predicted_district:
        output_file = f"{predicted_district} District.xlsx"
    else:
        output_file = "UserInput_Output.xlsx"  # Default file name if no district is found
    full_df.to_excel(output_file, index=False)
    print(f"Data saved to {output_file}")

# Save the DataFrame to Excel after processing the input
save_to_excel(predicted_district)
