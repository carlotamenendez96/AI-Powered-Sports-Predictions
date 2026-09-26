import requests
import zipfile
import io
import os
import shutil
import sys

# Config
BASE_URL = "https://www.football-data.co.uk/mmz4281"
DATA_ZIP_URL = f"{BASE_URL}/2526/data.zip" # Defaulting to current season for the update button
TARGET_DIR = "data_sets/MatchHistory"
SEASON_SUFFIX = "25-26"

# 1. Main Leagues (from data.zip) -> Legacy Mapping
MAIN_MAPPING = {
    "E0": "ENG-Premier_League",
    "E1": "ENG-Championship",
    "E2": "ENG-League_1",      
    "E3": "ENG-League_2",      
    "EC": "ENG-Conference",    
    "D1": "GER-Bundesliga",
    "D2": "GER-Bundesliga_2",  
    "I1": "ITA-Serie_A",
    "I2": "ITA-Serie_B",
    "SP1": "ESP-La_Liga",      
    "SP2": "ESP-Segunda",      
    "F1": "FRA-Ligue_1",
    "F2": "FRA-Ligue_2",
    "N1": "NED-Eredivisie",
    "B1": "BEL-Jupiler_League",
    "P1": "POR-Liga_1",        
    "T1": "TUR-Ligi_1",        
    "SC0": "SCO-Premier_League",
    "SC1": "SCO-Division_1",    
    "SC2": "SCO-Division_2",    
    "SC3": "SCO-Division_3",    
    "G1": "GR-Super_League"     
}

# 2. Extra Leagues (Direct URLs)
EXTRA_LEAGUES_MAP = {
    "ARG": ("ARG", "ARG-Liga_Profesional"),
    "AUT": ("AUT", "AUT-Bundesliga"),
    "BRA": ("BRA", "BRA-Serie_A"),
    "CHN": ("CHN", "CHN-Super_League"),
    "DNK": ("DNK", "DEN-Superliga"), 
    "FIN": ("FIN", "FIN-Veikkausliiga"),
    "IRL": ("IRL", "IRL-Premier_Division"),
    "JPN": ("JPN", "JPN-J1_League"),
    "MEX": ("MEX", "MEX-Liga_MX"),
    "NOR": ("NOR", "NOR-Eliteserien"),
    "POL": ("POL", "POL-Ekstraklasa"),
    "ROU": ("ROU", "ROU-Liga_1"),
    "RUS": ("RUS", "RUS-Premier_League"),
    "SWZ": ("SWZ", "SUI-Super_League"), 
    "SWE": ("SWE", "SWE-Allsvenskan"),
    "USA": ("USA", "USA-MLS")
}

def download_file(url):
    print(f"Fetching {url}...")
    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        return r.content
    except Exception as e:
        print(f"[-] Failed to download {url}: {e}")
        return None

def update_data():
    """Refresh the football-data.co.uk CSVs. Returns the number of files written."""
    print(f"[*] Starting Data Update for Season {SEASON_SUFFIX}...")
    written = 0
    
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)

    # --- Step 1: Main Data Zip ---
    zip_content = download_file(DATA_ZIP_URL)
    
    if zip_content:
        print("[*] Extracting Main Leagues...")
        try:
            with zipfile.ZipFile(io.BytesIO(zip_content)) as z:
                for filename in z.namelist():
                    if not filename.endswith(".csv"):
                        continue
                    
                    code = os.path.splitext(os.path.basename(filename))[0]
                    
                    if code in MAIN_MAPPING:
                        target_name = f"{MAIN_MAPPING[code]}_{SEASON_SUFFIX}.csv"
                        target_path = os.path.join(TARGET_DIR, target_name)
                        
                        with open(target_path, 'wb') as f:
                            f.write(z.read(filename))
                        written += 1
                        print(f"    -> Updated {target_name}")
        except Exception as e:
            print(f"[-] Error parsing zip: {e}")
            
    # --- Step 2: Extra Leagues ---
    print("\n[*] Updating Extra Leagues...")
    for code, (url_code, base_name) in EXTRA_LEAGUES_MAP.items():
        # Using /new/ for latest updates
        csv_url = f"https://www.football-data.co.uk/new/{url_code}.csv"
        csv_content = download_file(csv_url)
        
        if csv_content:
            # Consolidated files - NO Suffix
            target_name = f"{base_name}.csv"
            target_path = os.path.join(TARGET_DIR, target_name)
            
            with open(target_path, 'wb') as f:
                f.write(csv_content)
            written += 1
            print(f"    -> Updated {target_name}")

    expected = len(MAIN_MAPPING) + len(EXTRA_LEAGUES_MAP)
    print(f"\n[+] Update complete. {written}/{expected} files written.")
    return written


if __name__ == "__main__":
    # Every download is wrapped in its own try/except, so a total outage (no
    # network, football-data.co.uk down, URL scheme changed) used to print a
    # wall of "[-] Failed to download" and still exit 0 — indistinguishable
    # from success to any caller. run_predictions.sh now chains this step, so
    # it needs a real exit code: nothing written is a failure.
    if update_data() == 0:
        print("[-] No files were written — every download failed. See errors above.")
        sys.exit(1)
