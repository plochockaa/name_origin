# This script provides an interface to infer the likely country of origin for a given forename and/or surname.
# 
import pandas as pd
import os
from openai import OpenAI
import json
from dotenv import load_dotenv
from utils.cleaning import normalize_name, romanize_if_needed


# Load environment variables from .env file
load_dotenv()


# STEP 3: Lookup tables 
def lookup_name_country(
    forename=None,
    surname=None,
    forename_table=None,
    surname_table=None,
    romanize=True
):
    """
    Lookup likely countries for a forename, surname, or both.
    
    Returns a dictionary with:
        'forename': {name, romanized, candidates}
        'surname':  {name, romanized, candidates}
        'combined': combined candidate countries (union)
    """

    result = {}
    candidates_forename = []
    candidates_surname = []

    # -----------------------------
    # Forename lookup
    # -----------------------------
    if forename and forename_table is not None:
        romanized = romanize_if_needed(forename) if romanize else forename
        norm_forename = normalize_name(romanized)

        row = forename_table.loc[forename_table["name"] == norm_forename]
        if not row.empty:
            countries_value = row.iloc[0]["countries"]
            # Convert string representation of list to actual list
            if isinstance(countries_value, str):
                candidates_forename = json.loads(countries_value.replace("'", '"'))
            else:
                candidates_forename = countries_value if isinstance(countries_value, list) else []
        
        result["forename"] = {
            "input": forename,
            "romanized": romanized,
            "normalised": norm_forename,
            "countries": candidates_forename
        }

    # -----------------------------
    # Surname lookup
    # -----------------------------
    if surname and surname_table is not None:
        romanized = romanize_if_needed(surname) if romanize else surname
        norm_surname = normalize_name(romanized)

        row = surname_table.loc[surname_table["name"] == norm_surname]
        if not row.empty:
            countries_value = row.iloc[0]["countries"]
            # Convert string representation of list to actual list
            if isinstance(countries_value, str):
                candidates_surname = json.loads(countries_value.replace("'", '"'))
            else:
                candidates_surname = countries_value if isinstance(countries_value, list) else []
        
        result["surname"] = {
            "input": surname,
            "romanized": romanized,
            "normalised": norm_surname,
            "countries": candidates_surname
        }

    # -----------------------------
    # Combined countries if both
    # -----------------------------
    if forename and surname:
        # Combine both lists
        combined = candidates_forename + candidates_surname
    elif forename:
        combined = candidates_forename
    elif surname:
        combined = candidates_surname
    else:
        combined = []
    
    result["combined_lookup"] = combined
    return result


# STEP 4: LLM call to get probabilities based on initial lookup 
def llm_call(firstname,lastname,country_list):
    # Check if API key is available
    api_key = os.getenv("HF_TOKEN")
    if not api_key:
        print("Warning: HF_TOKEN environment variable not set. Skipping LLM call.")
        return {}
    
    prompt = f"""
    I have a name {firstname} {lastname} with possible countries: {', '.join(country_list)}.

    Please output a JSON object listing each country with its estimated probability (0-1)
    that this name originates from that country.

    Example format:
    {{
        "Egypt": 0.2,
        "United States": 0.5,
        "Australia": 0.3
    }}
    """

    try:
        client = OpenAI(
            base_url="https://router.huggingface.co/v1",
            api_key=api_key,
        )

        completion = client.chat.completions.create(
            model="CohereLabs/command-a-reasoning-08-2025:cohere",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0,
        )

        # Extract the LLM response
        llm_response_text = completion.choices[0].message.content

        # Convert to a Python dict safely
        try:
            # Try to extract JSON from the response (it might have extra text)
            start_idx = llm_response_text.find('{')
            end_idx = llm_response_text.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = llm_response_text[start_idx:end_idx]
                country_probabilities = json.loads(json_str)
            else:
                country_probabilities = json.loads(llm_response_text)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse JSON. Error: {e}")
            country_probabilities = {}

        return country_probabilities
    
    except Exception as e:
        print(f"Warning: LLM call failed with error: {e}")
        print("Returning countries without probabilities.")
        return {}

###############################
# Main function
def infer_country(firstname, lastname, forename_table, surname_table):


    # Use lookup tables
    result = lookup_name_country(
        forename=firstname,
        surname=lastname,
        forename_table=forename_table,
        surname_table=surname_table
    )

    country_list = result.get("combined_lookup", [])
    
    # Ensure country_list is actually a list
    if isinstance(country_list, str):
        country_list = []
    
    # Support with LLM call only if we have some countries from lookup
    if country_list:
        country_probabilities = llm_call(firstname,lastname,country_list)
        result["country_probabilities"] = country_probabilities
    else:
        result["country_probabilities"] = {}

    return result


def format_output(result):
    """Format the result in a clean, readable way"""
    forename_data = result.get('forename', {})
    surname_data = result.get('surname', {})
    
    firstname = forename_data.get('input', 'N/A')
    surname = surname_data.get('input', 'N/A')
    
    # Get normalized names
    norm_firstname = forename_data.get('normalised', 'N/A')
    norm_surname = surname_data.get('normalised', 'N/A')
    
    # Get romanized names
    roman_firstname = forename_data.get('romanized', firstname)
    roman_surname = surname_data.get('romanized', surname)
    
    countries = result.get('combined_lookup', [])
    probabilities = result.get('country_probabilities', {})
    
    print(f"\n{'='*60}")
    print(f"NAME ORIGIN ANALYSIS")
    print(f"{'='*60}")
    print(f"First Name (Original):  {firstname}")
    print(f"Surname (Original):     {surname}")
    
    # Show romanization if different from original
    if roman_firstname != firstname or roman_surname != surname:
        print(f"First Name (Romanized): {roman_firstname}")
        print(f"Surname (Romanized):    {roman_surname}")
    
    print(f"First Name (Normalized): {norm_firstname}")
    print(f"Surname (Normalized):    {norm_surname}")
    
    # Show lookup table results
    print(f"\n{'Lookup Table Results:'}")
    print(f"{'-'*60}")
    unique_countries = list(set(countries))
    if unique_countries:
        print(f"Countries found: {', '.join(unique_countries)}")
        print(f"Total matches: {len(countries)} (unique: {len(unique_countries)})")
    else:
        print("No countries found in lookup tables")
    
    # Show LLM probability results
    if probabilities:
        print(f"\n{'LLM Probability Analysis:'}")
        print(f"{'-'*60}")
        print(f"{'Country':<30} {'Probability':<15}")
        print(f"{'-'*45}")
        # Sort by probability (highest first)
        sorted_countries = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for country, prob in sorted_countries:
            print(f"{country:<30} {prob:.1%}")
    else:
        print(f"\n(No LLM probability analysis available)")
    
    print(f"{'='*60}\n")

# Example names to test:
#"Александр", 
# "Иванов", 
# "Ελένη",
# "Αθανασίου", 


if __name__ == "__main__":
    # Test the function
    result = infer_country(
        "Kim",
        "Lee", 
        pd.read_csv("data/final_forenames.csv"), 
        pd.read_csv("data/final_surnames.csv")
    )
    
    # Display formatted output
    format_output(result)