import certifi
import os
import ssl

# Configure SSL to use certifi CA bundle and ensure requests honors it
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl.create_default_context(cafile=certifi.where())

import urllib.request as _urllib_request
# Monkey-patch urlopen to include certifi CA bundle by default
_orig_urlopen = _urllib_request.urlopen
def _urlopen(url, *args, **kwargs):
    if 'context' not in kwargs:
        kwargs['context'] = ssl.create_default_context(cafile=certifi.where())
    return _orig_urlopen(url, *args, **kwargs)
_urllib_request.urlopen = _urlopen

import urllib3
from urllib3.poolmanager import PoolManager
# Monkey-patch PoolManager to enforce certifi CA bundle
_orig_pool_init = PoolManager.__init__
def _pool_init(self, *args, **kwargs):
    kwargs['ssl_context'] = ssl.create_default_context(cafile=certifi.where())
    return _orig_pool_init(self, *args, **kwargs)
PoolManager.__init__ = _pool_init

from urllib3.util import ssl_ as _ssl_util
# Monkey-patch create_urllib3_context to use certifi CA bundle
_ssl_util.create_urllib3_context = lambda *args, **kwargs: ssl.create_default_context(cafile=certifi.where())

import nfl_data_py as nfl
import json
import requests
import pandas as pd
from dotenv import load_dotenv
import time # Import time for potential rate limiting/delay

# --- Helper function to get the current max version ---
def get_current_max_version(supabase_url, table_name, headers, session):
    """Queries Supabase to find the maximum value in the 'version' column."""
    # Use order and limit to efficiently get the max value from the DB
    url = f"{supabase_url}/{table_name}?select=version&order=version.desc&limit=1"
    print(f"Fetching max version from: {url}")
    try:
        response = session.get(url, headers=headers, timeout=15)
        response.raise_for_status()
        data = response.json()

        if not data:
            # No records found, or version column is all NULL
            print("No existing version found. Starting with version 0.")
            return 0
        else:
            # Data is a list like [{"version": 5}] or [{"version": null}]
            max_version = data[0].get('version')
            if max_version is None:
                 print("Max version found is NULL. Starting with version 0.")
                 return 0
            else:
                 print(f"Found max existing version: {max_version}")
                 return int(max_version) # Ensure it's an integer

    except requests.exceptions.RequestException as e:
        print(f"Error fetching max version: {e}")
        # Decide how to handle: raise error, return default, etc.
        # For safety, let's stop the script if we can't get the version.
        raise ValueError("Could not determine the current max version from Supabase.") from e
    except Exception as e:
        print(f"An unexpected error occurred while fetching max version: {e}")
        raise ValueError("Unexpected error determining max version.") from e

# --- Main script ---
def main():
    # Configuration for the season and Supabase
    season_year = 2024
    # Assuming week might be relevant later, but not directly used for version calculation here
    # week = 1

    load_dotenv()

    SUPABASE_URL = os.getenv("SUPABASE_URL")
    API_KEY = os.getenv("SUPABASE_KEY")
    if not SUPABASE_URL or not API_KEY:
        print("Missing SUPABASE_URL or API_KEY in environment variables.")
        return
    SUPABASE_URL = f"{SUPABASE_URL}/rest/v1"
    ROSTERS_TABLE_NAME = "Rosters" # Use a variable for the table name

    # Headers for general GET requests
    GET_HEADERS = {
        "apikey": API_KEY,
        "Authorization": f"Bearer {API_KEY}",
    }

    # Headers specifically for the UPSERT operation
    UPSERT_HEADERS = {
        "apikey": API_KEY,
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=representation,resolution=merge-duplicates"
    }

    # Create a session and configure it to use certifi's CA bundle
    session = requests.Session()
    session.verify = certifi.where()

    # --- Get Current Max Version and Determine New Version ---
    try:
        max_existing_version = get_current_max_version(SUPABASE_URL, ROSTERS_TABLE_NAME, GET_HEADERS, session)
        new_version = max_existing_version + 1
        print(f"Using new version for this sync: {new_version}")
    except ValueError as e:
        print(f"Stopping script due to error getting version: {e}")
        return
    # --- End Version Logic ---

    # 1. Import the seasonal rosters as a single DataFrame.
    print(f"Importing rosters for {season_year}...")
    try:
        roster_df = nfl.import_seasonal_rosters([season_year])
        roster_df = roster_df.replace({pd.NA: None, pd.NaT: None, float('nan'): None})
        roster_df['position'] = roster_df['position'].fillna('UNK')
        roster_df['player_name'] = roster_df['player_name'].fillna('Unknown Player')
        roster_df['team'] = roster_df['team'].fillna('UNK')
    except Exception as e:
        print(f"Error importing rosters: {e}")
        return

    if roster_df.empty:
        print(f"No roster data found for {season_year}.")
        return
    print(f"Found {len(roster_df)} roster entries.")

    # 2. Define the mapping (already done in code)

    # 3. Fetch Teams from Supabase
    print("Fetching teams from Supabase...")
    teams_url = f"{SUPABASE_URL}/Teams?select=id,teamId"
    try:
        # Use GET_HEADERS here
        teams_response = session.get(teams_url, headers=GET_HEADERS, timeout=10)
        teams_response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Teams data: {e}")
        return

    teams_data = teams_response.json()
    team_mapping = { team["teamId"]: team["id"] for team in teams_data if "teamId" in team and "id" in team }
    print(f"Built mapping for {len(team_mapping)} teams.")

    # 4. Build the payload
    payload = []
    skipped_count = 0
    processed_count = 0
    problematic_records = []
    print("Processing roster data...")
    for index, row in roster_df.iterrows():
        processed_count += 1
        player_name = row.get("player_name", "Unknown Player")
        position = row.get("position", "UNK")
        team_abbr_raw = row.get("team", "UNK")
        jersey_number = row.get("jersey_number")
        headshot_url = row.get("headshot_url")
        age_raw = row.get("age")
        height = row.get("height")
        weight_raw = row.get("weight")
        college = row.get("college")
        status = row.get("status")
        years_exp_raw = row.get("years_exp")

        if team_abbr_raw == "UNK":
             # print(f"Skipping row {index}: Missing essential team data for player {player_name}.")
             skipped_count += 1
             problematic_records.append({'index': index, 'reason': 'Missing team', 'data': row.to_dict()})
             continue

        slug_jersey = str(int(jersey_number)) if pd.notna(jersey_number) else "NA"
        try:
            slug = f"{position}-{player_name}-{team_abbr_raw}-{slug_jersey}".replace(" ", "-").replace(".", "").lower()
            if len(slug) > 255: slug = slug[:255]
        except Exception as e:
            # print(f"Error creating slug for row {index}: {e}")
            skipped_count += 1
            problematic_records.append({'index': index, 'reason': f'Slug creation error: {e}', 'data': row.to_dict()})
            continue

        number_val = None
        if pd.notna(jersey_number):
            try: number_val = int(jersey_number)
            except (ValueError, TypeError): pass

        age_val = None
        if pd.notna(age_raw):
            try: age_val = int(age_raw)
            except (ValueError, TypeError): pass

        weight_val = None
        if pd.notna(weight_raw):
            try: weight_val = int(weight_raw)
            except (ValueError, TypeError): pass

        years_exp_val = None
        if pd.notna(years_exp_raw):
             try: years_exp_val = int(years_exp_raw)
             except (ValueError, TypeError): pass

        team_abbr = team_abbr_raw
        if team_abbr == "LA": team_abbr = "LAR"
        elif team_abbr == "JAC": team_abbr = "JAX"

        team_fk = team_mapping.get(team_abbr)
        if team_fk is None:
            # print(f"Warning: No matching team found for team abbreviation '{team_abbr}' (Original: '{team_abbr_raw}') for player {player_name}. Skipping record.")
            skipped_count += 1
            problematic_records.append({'index': index, 'reason': f'Team FK not found for {team_abbr}', 'data': row.to_dict()})
            continue

        # *** ADD 'version' TO THE PLAYER RECORD ***
        player_record = {
            "slug": slug,
            "position": position,
            "status": status,
            "number": number_val,
            "height": height,
            "weight": weight_val,
            "name": player_name,
            "college": college,
            "headshotURL": headshot_url,
            "age": age_val,
            "years_exp": years_exp_val,
            "teamId": team_fk,
            "version": new_version  # Add the calculated new version
        }

        payload.append(player_record)

    print(f"Processed {processed_count} rows. Built payload with {len(payload)} records. Skipped {skipped_count} records.")
    if skipped_count > 0:
         print(f"Review problematic records (first 5): {problematic_records[:5]}")

    if not payload:
        print("Payload is empty. No data to insert/update.")
        return

    # 5. Upsert the records into the Supabase Rosters table.
    rosters_url = f"{SUPABASE_URL}/{ROSTERS_TABLE_NAME}?on_conflict=slug"
    print(f"Attempting to upsert {len(payload)} records with version {new_version}...")
    # print(f"URL: {rosters_url}")
    # print(f"Headers: {UPSERT_HEADERS}")

    try:
        # Use UPSERT_HEADERS here
        response = session.post(rosters_url, headers=UPSERT_HEADERS, json=payload, timeout=120)
        response.raise_for_status()

        print("Data upserted successfully:")
        returned_data = response.json()
        print(f"Response status code: {response.status_code}")
        print(f"Supabase returned {len(returned_data)} affected records.")

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred during upsert: {http_err}")
        print(f"Response status code: {http_err.response.status_code}")
        try:
            error_details = http_err.response.json()
            print(f"Supabase error details: {json.dumps(error_details, indent=2)}")
        except json.JSONDecodeError:
            print(f"Response text: {http_err.response.text}")
        if http_err.response.status_code == 400:
             print("Got 400 Bad Request. Check data types (especially 'version' being int) and payload format.")
             # print("First 5 payload records:", json.dumps(payload[:5], indent=2))

    except requests.exceptions.RequestException as req_err:
        print(f"A network or request error occurred: {req_err}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()