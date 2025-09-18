import os
import warnings
warnings.warn(
    "This script is deprecated. Use 'python scripts/injuries_cli.py' instead. "
    "The new CLI uses standardized loaders with nfl_data_py and follows the core data pipeline pattern.",
    DeprecationWarning,
    stacklevel=2
)
import requests # Ensure requests is imported
import json
from supabase import create_client, Client
from dotenv import load_dotenv
from datetime import datetime
import logging
import mimetypes # For guessing content type
from urllib.parse import urlparse # To get file extension
import time # Import the time module

# Control flag for database writes vs console output
# Set to False to print JSON/actions to console, True to write to Supabase & Storage
WRITE_TO_SUPABASE = True  # Set to True to enable DB/Storage writes

# --- Configuration ---
load_dotenv()

logging.basicConfig(
    level=logging.INFO, # Change to DEBUG for more verbose output
    format='%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Supabase Credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY") # Should be service_role key

# API Credentials & Rate Limiting
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST")
API_ENDPOINT = f"https://{RAPIDAPI_HOST}/injuries"
API_BATCH_SIZE = 8  # Number of API requests per batch
API_DELAY_SECONDS = 61 # Delay between batches (slightly over 1 minute)

# Supabase Storage Bucket Name
STORAGE_BUCKET_NAME = "player" # Make sure this matches your bucket name

logger.info("WRITE_TO_SUPABASE flag set to: %s", WRITE_TO_SUPABASE)
logger.info("API Batch Size: %d, Delay Between Batches: %d seconds", API_BATCH_SIZE, API_DELAY_SECONDS)


# --- Input Validation ---
if not all([SUPABASE_URL, SUPABASE_KEY, RAPIDAPI_KEY, RAPIDAPI_HOST]):
    logger.error("Missing required environment variables.")
    exit(1)
if not SUPABASE_KEY or not SUPABASE_KEY.startswith('ey'): # Basic sanity check for service role key format
     logger.warning("SUPABASE_KEY is missing or does not look like a Service Role JWT. Ensure it's set correctly in .env")

# --- Supabase Client Initialization ---
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    logger.info("Successfully connected to Supabase.")
except Exception as e:
    logger.error("Failed to connect to Supabase: %s", str(e))
    exit(1)

# --- Helper Functions ---

def get_version_info():
    """Queries Supabase for the latest version in Injuries table and calculates the next."""
    try:
        response = supabase.table("Injuries") \
            .select("version") \
            .order("version", desc=True) \
            .limit(1) \
            .execute()
        latest_version = 0
        if response is None: logger.error("Supabase call returned None fetching latest version. Assuming 0.")
        elif hasattr(response, 'error') and response.error: logger.warning("Error fetching latest version, assuming 0: %s", response.error)
        elif hasattr(response, 'data') and response.data: latest_version = response.data[0].get('version', 0)

        next_version = latest_version + 1
        logger.info("Latest Injury version found: %s. Next version for this run: %s", latest_version, next_version)
        return latest_version, next_version
    except Exception as e:
        logger.error("Exception fetching latest Injury version: %s", str(e), exc_info=True)
        exit(1)

def fetch_teams():
    """Fetches all teams from the Supabase Teams table."""
    try:
        response = supabase.table("Teams").select("id, SportAPI_id").execute()
        if response is None: logger.error("Supabase call returned None fetching teams."); return []
        elif hasattr(response, 'error') and response.error: logger.error("Error fetching teams from Supabase: %s", response.error); return []
        elif hasattr(response, 'data') and response.data: logger.info("Fetched %s teams from Supabase.", len(response.data)); return response.data
        else: logger.warning("No teams found in the Supabase 'Teams' table."); return []
    except Exception as e:
        logger.error("Exception fetching teams from Supabase: %s", str(e), exc_info=True)
        return []

def fetch_injuries_from_api(team_api_id):
    """Fetches injuries for a specific team ID from the API."""
    headers = {'x-rapidapi-key': RAPIDAPI_KEY, 'x-rapidapi-host': RAPIDAPI_HOST}
    params = {'team': team_api_id}
    logger.debug("Fetching injuries for API team ID: %s", team_api_id)
    try:
        response = requests.get(API_ENDPOINT, headers=headers, params=params, timeout=45)
        response.raise_for_status()
        data = response.json()
        api_errors = data.get("errors")
        if api_errors and ((isinstance(api_errors, list) and api_errors) or (isinstance(api_errors, dict) and api_errors)):
            logger.error("API returned errors for team %s: %s", team_api_id, api_errors)
            return None
        api_response_list = data.get("response")
        if isinstance(api_response_list, list):
            results_count = data.get('results', len(api_response_list))
            logger.info("Fetched %s injuries from API for team %s.", results_count, team_api_id)
            return api_response_list
        else:
            if data.get('results') == 0 and 'response' in data:
                logger.info("API reported 0 injuries for team %s.", team_api_id)
                return []
            logger.warning("Unexpected API response structure for team %s. Data: %s", team_api_id, data)
            return None
    except requests.exceptions.Timeout: logger.error("API request timed out for team %s.", team_api_id); return None
    except requests.exceptions.HTTPError as e: logger.error("HTTP error fetching injuries for team %s: %s - %s", team_api_id, e.response.status_code, e.response.text); return None
    except requests.exceptions.RequestException as e: logger.error("Network/Request error fetching injuries for team %s: %s", team_api_id, str(e)); return None
    except json.JSONDecodeError as e:
        logger.error("Failed to decode JSON response for team %s: %s", team_api_id, str(e))
        responseText = response.text if 'response' in locals() and hasattr(response, 'text') else "Response text not available"
        logger.error("Response text: %s", responseText)
        return None
    except Exception as e: logger.error("Unexpected error during API call for team %s: %s", team_api_id, str(e), exc_info=True); return None

# ==============================================================
# === REVISED upload_player_image function ===
# ==============================================================
def upload_player_image(api_player_id, original_img_url):
    """
    Downloads image, uploads to Storage. Returns Supabase path or None.
    Fixes upsert value to be string "true".
    """
    if not original_img_url:
        logger.debug("No original image URL provided for API player %s.", api_player_id)
        return None
    try:
        logger.debug("Downloading image for player %s from %s", api_player_id, original_img_url)
        img_headers = {'User-Agent': 'Mozilla/5.0'}
        img_response = requests.get(original_img_url, stream=True, timeout=30, headers=img_headers)
        img_response.raise_for_status()
        image_data = img_response.content
        content_type = img_response.headers.get('content-type', 'application/octet-stream')
        logger.debug("Image downloaded successfully (%s bytes, type: %s)", len(image_data), content_type)
        parsed_url = urlparse(original_img_url)
        _, ext = os.path.splitext(parsed_url.path)
        if not ext or len(ext) > 5:
             guessed_ext = mimetypes.guess_extension(content_type)
             if guessed_ext and len(guessed_ext) <=5 :
                 ext = guessed_ext
                 logger.debug("Guessed extension '%s' from content type '%s'", ext, content_type)
             else:
                  ext = '.png'
                  logger.warning("Could not determine/guess valid extension for URL %s (Content-Type: %s), defaulting to %s", original_img_url, content_type, ext)
        ext = "." + ext.lstrip('.').lower()
        supabase_file_path = f"{api_player_id}{ext}"
        logger.debug("Determined Supabase path: %s", supabase_file_path)

        if WRITE_TO_SUPABASE:
            logger.info("Uploading image to Supabase Storage: %s/%s", STORAGE_BUCKET_NAME, supabase_file_path)
            # **** THE FIX IS HERE ****
            file_options = {"contentType": content_type, "cacheControl": "3600", "upsert": "true"} # Use string "true"
            # *************************
            supabase.storage.from_(STORAGE_BUCKET_NAME).upload(path=supabase_file_path, file=image_data, file_options=file_options)
            logger.info("Successfully uploaded image to Supabase path: %s", supabase_file_path)
            return supabase_file_path
        else:
            logger.info("[Dry Run] Would upload image for player %s to Supabase path: %s", api_player_id, supabase_file_path)
            return supabase_file_path
    except requests.exceptions.RequestException as e: logger.error("Failed download image player %s: %s", api_player_id, str(e)); return None
    except Exception as e: logger.error("Failed upload image player %s: %s", api_player_id, str(e), exc_info=True); return None
# ==============================================================
# ==============================================================


def get_public_url_from_path(path):
    """Generates public URL for storage path."""
    if not path: return None
    try:
        public_url = supabase.storage.from_(STORAGE_BUCKET_NAME).get_public_url(path)
        logger.debug("Generated public URL: %s", public_url)
        return public_url
    except Exception as e:
        logger.error("Failed generate public URL for path '%s': %s", path, str(e), exc_info=True)
        return None

def get_or_create_player(api_player_id, player_name, original_player_img_url):
    """
    Gets/Creates Player record WITHOUT CACHING, using direct requests call for SELECT.
    Handles image upload, stores full public URL.
    Returns Supabase Player PK ID or None.
    """
    if not api_player_id:
        logger.warning("Received invalid API Player ID. Cannot process player.")
        return None

    logger.debug("Processing player API ID: %s. Querying DB using direct requests.", api_player_id)
    public_img_url = None
    supabase_player_pk_id = None
    current_db_url = None
    player_exists = False

    try:
        # --- Query DB using direct requests ---
        query_url = f"{SUPABASE_URL}/rest/v1/Player?select=id,img_url&playerId=eq.{api_player_id}"
        headers = {
            'apikey': SUPABASE_KEY, # Use the SERVICE_ROLE_KEY here
            'Authorization': f'Bearer {SUPABASE_KEY}',
            'Accept': 'application/json' # Explicitly request JSON
        }

        logger.debug("Making direct GET request to: %s", query_url)
        direct_response = requests.get(query_url, headers=headers, timeout=30)

        # Check the status code from the direct request
        if direct_response.status_code == 200:
            try:
                data = direct_response.json()
                if data and isinstance(data, list) and len(data) > 0:
                    player_exists = True
                    supabase_player_pk_id = data[0]['id']
                    current_db_url = data[0].get('img_url')
                    logger.debug("Found Player via direct requests. API ID: %s -> PK: %s.", api_player_id, supabase_player_pk_id)
                else:
                    logger.debug("Player API ID %s not found via direct requests (200 OK, empty data).", api_player_id)
                    player_exists = False
            except json.JSONDecodeError:
                 logger.error("Failed to decode JSON response from direct requests for player %s (Status 200).", api_player_id)
                 logger.error("Response text: %s", direct_response.text)
                 return None
            except Exception as e:
                 logger.error("Error processing successful (200 OK) direct requests response for player %s: %s", api_player_id, e, exc_info=True)
                 return None
        elif direct_response.status_code == 406:
             logger.error("Direct requests query received 406 Not Acceptable for player API ID %s.", api_player_id)
             logger.error("Response text: %s", direct_response.text)
             return None
        else:
             logger.error("Direct requests query failed for player API ID %s. Status: %s, Response: %s",
                          api_player_id, direct_response.status_code, direct_response.text)
             return None

        # --- Process based on whether player exists ---
        if player_exists:
            # Player was found - proceed with potential image update logic
            if original_player_img_url:
                uploaded_file_path = upload_player_image(api_player_id, original_player_img_url)
                public_img_url = get_public_url_from_path(uploaded_file_path)

            if public_img_url and public_img_url != current_db_url:
                logger.info("Updating Player %s (PK %s) img_url in DB to: %s", player_name, supabase_player_pk_id, public_img_url)
                if WRITE_TO_SUPABASE:
                    update_resp = supabase.table("Player").update({"img_url": public_img_url}).eq("id", supabase_player_pk_id).execute()
                    if update_resp is None: logger.error("Supabase update img_url call returned None for player PK %s.", supabase_player_pk_id)
                    elif hasattr(update_resp, 'error') and update_resp.error: logger.error("Failed to update img_url for player PK %s: %s", supabase_player_pk_id, update_resp.error)
                else: logger.info("[Dry Run] Would update Player PK %s img_url to %s", supabase_player_pk_id, public_img_url)
            elif public_img_url and public_img_url == current_db_url:
                 logger.debug("Public URL '%s' already up-to-date for player PK %s.", public_img_url, supabase_player_pk_id)
            elif not original_player_img_url and current_db_url:
                 logger.debug("API provided no image URL for player PK %s, leaving existing URL in DB.", supabase_player_pk_id)

            return supabase_player_pk_id # Return the existing ID

        else:
            # --- Player Not Found - Create New ---
            logger.info("Player '%s' (API ID: %s) not found via direct request. Creating.", player_name, api_player_id)

            if original_player_img_url:
                uploaded_file_path = upload_player_image(api_player_id, original_player_img_url)
                public_img_url = get_public_url_from_path(uploaded_file_path)
                if not public_img_url: logger.warning("Failed upload/URL gen for new player %s.", player_name)

            player_data_to_insert = {
                "playerId": api_player_id, "name": player_name, "img_url": public_img_url
            }

            if WRITE_TO_SUPABASE:
                insert_response = supabase.table("Player").insert(player_data_to_insert).execute()
                if insert_response is None: logger.error("Supabase insert player call returned None for player '%s'.", player_name); return None
                if hasattr(insert_response, 'error') and insert_response.error: logger.error("Failed insert player '%s'. Error: %s", player_name, insert_response.error); return None
                if insert_response.data and len(insert_response.data) > 0:
                    new_player_pk_id = insert_response.data[0]['id']
                    logger.info("Inserted Player '%s' (API ID: %s) -> PK: %s.", player_name, api_player_id, new_player_pk_id)
                    return new_player_pk_id
                else:
                    logger.error("Insert player '%s' succeeded but returned no data.", player_name)
                    return None
            else: # Dry run
                logger.info("[Dry Run] Would insert Player: %s", player_data_to_insert)
                return -api_player_id

    except requests.exceptions.RequestException as req_e:
        logger.error("Direct requests connection error checking player API ID %s: %s", api_player_id, str(req_e))
        return None
    except Exception as e:
        logger.error("Unexpected exception during get_or_create_player (direct requests) for API ID %s: %s", api_player_id, str(e), exc_info=True)
        return None


def process_injuries(api_injuries, supabase_team_id, latest_version, next_version):
    """Processes injuries, returns lists of updates/inserts for Injuries table."""
    updates_to_perform = []
    inserts_to_perform = []

    if not api_injuries: return updates_to_perform, inserts_to_perform

    for injury in api_injuries:
        player_data = injury.get("player", {})
        api_player_id = player_data.get("id")
        player_name = player_data.get("name")
        original_player_img_url = player_data.get("image")
        injury_date_str = injury.get("date")
        status = injury.get("status")
        description = injury.get("description")

        supabase_player_fk_id = get_or_create_player(
            api_player_id, player_name, original_player_img_url
        )

        if supabase_player_fk_id is None:
            logger.warning("Could not get/create player for API ID %s ('%s'). Skipping injury record.", api_player_id, player_name)
            continue

        if not all([injury_date_str, status]):
            logger.warning("Skipping injury: missing date/status. Team %s, PlayerFK %s", supabase_team_id, supabase_player_fk_id)
            continue
        try: datetime.strptime(injury_date_str, '%Y-%m-%d')
        except ValueError: logger.warning("Skipping injury: invalid date format '%s'. Team %s, PlayerFK %s", injury_date_str, supabase_team_id, supabase_player_fk_id); continue

        try:
            existing_injury_id = None
            if latest_version > 0:
                response = supabase.table("Injuries") \
                    .select("id") \
                    .eq("player", supabase_player_fk_id) \
                    .eq("team", supabase_team_id) \
                    .eq("version", latest_version) \
                    .maybe_single() \
                    .execute()
                if response is None: logger.error("Supabase query None checking existing injury: PlayerFK %s.", supabase_player_fk_id); continue
                elif hasattr(response, 'error') and response.error: logger.error("Failed check existing injury: PlayerFK %s. Error: %s", supabase_player_fk_id, response.error); continue
                elif response.data: existing_injury_id = response.data['id']

            if existing_injury_id is not None:
                update_payload = {"date": injury_date_str, "status": status, "description": description, "version": next_version}
                updates_to_perform.append((existing_injury_id, update_payload))
                logger.debug("Marked Injury UPDATE: PK %s -> v%s", existing_injury_id, next_version)
            else:
                insert_payload = {"player": supabase_player_fk_id, "team": supabase_team_id, "date": injury_date_str, "status": status, "description": description, "version": next_version}
                inserts_to_perform.append(insert_payload)
                logger.debug("Marked Injury INSERT: PlayerFK %s, Team %s -> v%s", supabase_player_fk_id, supabase_team_id, next_version)

        except Exception as e:
            logger.error("Exception checking/processing injury for PlayerFK %s: %s", supabase_player_fk_id, str(e), exc_info=True)
            continue

    return updates_to_perform, inserts_to_perform


# --- Main Execution ---
if __name__ == "__main__":
    logger.info("--- Starting Injury Update Script (Using Direct Requests for Player SELECT) ---")

    latest_version, next_version = get_version_info()
    teams = fetch_teams()

    if not teams: logger.warning("No teams found. Exiting."); exit(0)

    all_injury_updates = []
    all_injury_inserts = []
    total_api_injuries_processed = 0
    total_teams = len(teams)
    processed_teams_count = 0

    # --- Batch Processing Loop ---
    for i in range(0, total_teams, API_BATCH_SIZE):
        current_batch_teams = teams[i : i + API_BATCH_SIZE]
        batch_number = (i // API_BATCH_SIZE) + 1
        total_batches = (total_teams + API_BATCH_SIZE - 1) // API_BATCH_SIZE
        logger.info("--- Starting Batch %d of %d (%d teams) ---", batch_number, total_batches, len(current_batch_teams))

        for team in current_batch_teams:
            processed_teams_count += 1
            supabase_team_id = team.get('id')
            api_team_id = team.get('SportAPI_id')
            if not supabase_team_id or not api_team_id: logger.warning("Skipping team missing ID: %s", team); continue

            logger.info("Processing team ID: %s (API ID: %s) [Team %d/%d, Batch %d]", supabase_team_id, api_team_id, processed_teams_count, total_teams, batch_number)
            api_injuries = fetch_injuries_from_api(api_team_id)
            if api_injuries is None: logger.warning("Skipping team %s API fetch error.", supabase_team_id); continue

            updates, inserts = process_injuries(
                api_injuries, supabase_team_id, latest_version, next_version
            )
            all_injury_updates.extend(updates)
            all_injury_inserts.extend(inserts)
            total_api_injuries_processed += len(api_injuries)

        logger.info("--- Finished Batch %d of %d ---", batch_number, total_batches)
        is_last_batch = (processed_teams_count >= total_teams)
        if not is_last_batch:
            logger.info("Waiting for %d seconds before next batch...", API_DELAY_SECONDS)
            time.sleep(API_DELAY_SECONDS)
        else: logger.info("Last batch processed.")
    # --- End Batch Loop ---

    logger.info("--- All Batches Complete ---")
    logger.info("Total teams processed: %d", processed_teams_count)
    logger.info("Total API injury entries received: %s", total_api_injuries_processed)
    logger.info("Injury records marked for UPDATE to v%s: %s", next_version, len(all_injury_updates))
    logger.info("Injury records marked for INSERT with v%s: %s", next_version, len(all_injury_inserts))

    # --- Perform DB Ops / Print ---
    update_errors = 0
    insert_errors = 0
    if WRITE_TO_SUPABASE:
        logger.info("--- Writing INJURY data to Supabase ---")
        # Updates
        if all_injury_updates:
            logger.info("Performing %s Injury updates...", len(all_injury_updates))
            for record_id, payload in all_injury_updates:
                try:
                    response = supabase.table("Injuries").update(payload).eq("id", record_id).execute()
                    if response is None: logger.error("Update Injury ID %s returned None.", record_id); update_errors += 1
                    elif hasattr(response, 'error') and response.error: logger.error("Failed Update Injury ID %s: %s", record_id, response.error); update_errors += 1
                    elif not (hasattr(response, 'data') and response.data): logger.warning("Update Injury ID %s succeeded but no data.", record_id)
                except Exception as e: logger.error("Exception Update Injury ID %s: %s", record_id, str(e), exc_info=True); update_errors += 1
            logger.info("Injury updates finished: %s errors.", update_errors)
        else: logger.info("No Injury updates needed.")
        # Inserts
        if all_injury_inserts:
            logger.info("Performing %s Injury inserts...", len(all_injury_inserts))
            try:
                response = supabase.table("Injuries").insert(all_injury_inserts).execute()
                if response is None: logger.error("Bulk Insert Injury returned None."); insert_errors = len(all_injury_inserts)
                elif hasattr(response, 'error') and response.error: logger.error("Bulk Insert Injury Error: %s", response.error); insert_errors = len(all_injury_inserts)
                elif hasattr(response, 'data') and response.data:
                    inserted_count = len(response.data)
                    expected_count = len(all_injury_inserts)
                    logger.info("Inserted %s / %s Injury records.", inserted_count, expected_count)
                    if inserted_count != expected_count: logger.warning("Bulk insert mismatch: %s/%s.", inserted_count, expected_count); insert_errors = expected_count - inserted_count
                else: logger.error("Bulk Insert Injury no error but no data."); insert_errors = len(all_injury_inserts)
            except Exception as e: logger.error("Exception Bulk Insert Injuries: %s", str(e), exc_info=True); insert_errors = len(all_injury_inserts)
            logger.info("Injury inserts finished: %s errors.", insert_errors)
        else: logger.info("No Injury inserts needed.")
    else: # Dry Run
        logger.info("--- [Dry Run] Logging intended operations ---")
        logger.info("[Dry Run] Player ops logged during processing.")
        if all_injury_updates: print("\n--- Injury Updates Planned (JSON) ---"); print(json.dumps([{"update_injury_id": r_id, "payload": p} for r_id, p in all_injury_updates], indent=2))
        else: print("\n--- No Injury Updates Planned ---")
        if all_injury_inserts: print("\n--- Injury Inserts Planned (JSON) ---"); print(json.dumps(all_injury_inserts, indent=2))
        else: print("\n--- No Injury Inserts Planned ---")

    # Final Summary
    total_errors = update_errors + insert_errors
    outcome = "Successfully" if total_errors == 0 else f"with {total_errors} INJURY Errors"
    mode = "written to Supabase" if WRITE_TO_SUPABASE else "[Dry Run]"
    logger.info("--- Script Finished %s (%s, Version %s) ---", outcome, mode, next_version)
    if WRITE_TO_SUPABASE and total_errors > 0: logger.error("--- Review logs for specific errors. ---"); exit(1)
    elif not WRITE_TO_SUPABASE: logger.info("--- Dry run complete. ---")