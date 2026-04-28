import os
import json
import pandas as pd
import requests
import zipfile
import gzip
import shutil
import time
from pathlib import Path
from curl_cffi import requests as curl_requests
import urllib3
import lzma
import platform
import subprocess
import shlex

# Disable SSL warnings for downloads with verify=False
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Set base directory to the script's directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = BASE_DIR  # Will be overridden at runtime from config["path"]

# Removed here(): all write paths now use OUTPUT_DIR via os.path.join

def dim_desc(df):
    """Function for returning dimensions of dataframe as a string
    
    Args:
        df (pandas.DataFrame): The dataframe whose dimensions are to be described
    
    Returns:
        str: A string describing the dimensions in the format "X rows x Y columns"
    """
    return f"{df.shape[0]} rows x {df.shape[1]} columns"

def fips_info(fips_codes, reverse=False):
    """Function for converting FIPS codes to state abbreviations or vice versa. Used for creating destination folders for census data
    
    Args:
        fips_codes (str or list): FIPS code(s) or abbreviation(s) to convert. Can be a single string or a list
        reverse (bool, optional): If True, converts abbreviations to FIPS codes. If False, converts FIPS codes to abbreviations. Defaults to False.
    
    Returns:
        dict: Dictionary with key "abbr" or "fips" containing the converted values. Returns None for invalid codes.
              If input is a list, returns a list of converted values; if input is a string, returns a single converted value.
    """
    # Dictionary mapping FIPS codes to state abbreviations
    fips_to_abbr = {
        "01": "AL", "02": "AK", "04": "AZ", "05": "AR", "06": "CA", "08": "CO", "09": "CT",
        "10": "DE", "11": "DC", "12": "FL", "13": "GA", "15": "HI", "16": "ID", "17": "IL",
        "18": "IN", "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", "24": "MD",
        "25": "MA", "26": "MI", "27": "MN", "28": "MS", "29": "MO", "30": "MT", "31": "NE",
        "32": "NV", "33": "NH", "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
        "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", "45": "SC", "46": "SD",
        "47": "TN", "48": "TX", "49": "UT", "50": "VT", "51": "VA", "53": "WA", "54": "WV",
        "55": "WI", "56": "WY", "72": "PR"
    }
    
    if reverse:
        # Create reverse mapping from abbreviations to FIPS codes
        abbr_to_fips = {abbr: fips for fips, abbr in fips_to_abbr.items()}
        
        if isinstance(fips_codes, list):
            result = {"fips": [abbr_to_fips.get(code, None) for code in fips_codes]}
            return result
        else:
            return {"fips": abbr_to_fips.get(fips_codes, None)}
    else:
        if isinstance(fips_codes, list):
            result = {"abbr": [fips_to_abbr.get(code, None) for code in fips_codes]}
            return result
        else:
            return {"abbr": fips_to_abbr.get(fips_codes, None)}

# Helper function to download files with retry logic
def try_download(src, dst):
    """Function for downloading a file with timeout and retry logic
    
    Args:
        src (str): The source URL to download from
        dst (str): The destination file path where the downloaded file will be saved
    
    Returns:
        int: Status code (0 for success, -1 for failure)
    """
    timeout = 3600  # 1 hour timeout
    retries = 3
    status = 1
    
    while status != 0 and retries > 0:
        try:
            # First try with SSL verification enabled
            response = requests.get(src, timeout=timeout, stream=True)
            response.raise_for_status()
            
            with open(dst, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            status = 0
        except Exception as e:
            # If SSL error, try again with SSL verification disabled
            if "SSL" in str(e) or "certificate" in str(e).lower():
                try:
                    response = requests.get(src, timeout=timeout, stream=True, verify=False)
                    response.raise_for_status()
                    
                    with open(dst, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    status = 0
                except Exception as e2:
                    print(f"Download attempt failed (with SSL disabled): {e2}")
                    retries -= 1
                    status = -1
            else:
                print(f"Download attempt failed: {e}")
                retries -= 1
                status = -1
    
    if status != 0:
        print(f"Download failed: {src}")
        exit(1)
    
    return status

def try_curl_cffi(src, dst):
    """Function for downloading a file using curl_cffi with timeout and retry logic
    
    Args:
        src (str): The source URL to download from
        dst (str): The destination file path where the downloaded file will be saved
    
    Returns:
        int: Status code (0 for success, -1 for failure)
    """
    timeout = 3600  # 1 hour timeout
    retries = 3
    status = 1
    
    while status != 0 and retries > 0:
        try:
            # Use browser impersonation with appropriate headers
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Sec-Fetch-Dest": "document",
                "Sec-Fetch-Mode": "navigate",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-User": "?1",
                "Cache-Control": "max-age=0"
            }
            
            # Try with Chrome impersonation and SSL verification disabled for problematic sites
            response = curl_requests.get(
                src, 
                timeout=timeout,
                impersonate="chrome110",
                headers=headers,
                verify=False  # Disable SSL verification to handle certificate issues
            )
            
            if response.status_code >= 400:
                raise Exception(f"HTTP Error {response.status_code}")
            
            with open(dst, 'wb') as f:
                f.write(response.content)
            status = 0
            
        except Exception as e:
            print(f"Download attempt failed: {e}")
            retries -= 1
            status = -1
            time.sleep(2)  # Add a small delay between retries
    
    if status != 0:
        print(f"Download failed: {src}")
        exit(1)
    
    return status

def try_download_text(src, dst):
    """Function for downloading text content from web pages (like Census crosswalk files)
    
    Args:
        src (str): The source URL to download from
        dst (str): The destination file path where the downloaded text will be saved
    
    Returns:
        int: Status code (0 for success, -1 for failure)
    """
    timeout = 3600  # 1 hour timeout
    retries = 3
    status = 1
    
    while status != 0 and retries > 0:
        try:
            # Use browser-like headers to get the actual text content
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
                "Accept": "text/plain,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
                "Connection": "keep-alive",
                "Cache-Control": "max-age=0"
            }
            
            # First try with SSL verification enabled
            response = requests.get(src, timeout=timeout, headers=headers, stream=True)
            response.raise_for_status()
            
            # Write the content as text (not binary)
            with open(dst, 'w', encoding='utf-8') as f:
                for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                    if chunk:
                        f.write(chunk)
            status = 0
            
        except Exception as e:
            # If SSL error, try again with SSL verification disabled
            if "SSL" in str(e) or "certificate" in str(e).lower():
                try:
                    response = requests.get(src, timeout=timeout, headers=headers, stream=True, verify=False)
                    response.raise_for_status()
                    
                    with open(dst, 'w', encoding='utf-8') as f:
                        for chunk in response.iter_content(chunk_size=8192, decode_unicode=True):
                            if chunk:
                                f.write(chunk)
                    status = 0
                except Exception as e2:
                    print(f"Text download attempt failed (with SSL disabled): {e2}")
                    retries -= 1
                    status = -1
            else:
                print(f"Text download attempt failed: {e}")
                retries -= 1
                status = -1
    
    if status != 0:
        print(f"Text download failed: {src}")
        exit(1)
    
    return status

def get_census_metadata(name, vintage, type_="variables"):
    """Function for getting ACS and Decennial metadata from Census API
    
    Args:
        name (str): The census dataset name (e.g., "acs/acs5", "dec/dhc", "dec/sf1")
        vintage (str): The year of the census data (e.g., "2020", "2019"). Comes from the config.json file
        type_ (str, optional): The type of metadata to retrieve. Defaults to "variables". Can be "variables" or "geography"
    
    Returns:
        pandas.DataFrame: DataFrame containing the metadata with variables as rows and metadata fields as columns
    """
    url = f"https://api.census.gov/data/{vintage}/{name}/{type_}.json"
    response = requests.get(url)
    response.raise_for_status()
    return pd.DataFrame(response.json()["variables"]).T.reset_index().rename(columns={"index": "name"})

def get_census_data(name, vintage, vars, region, regionin, key):
    """Function for making a Census API call and getting the data. Works with all ACS years and Decennial years before 2020
    
    Args:
        name (str): The census dataset name (e.g., "acs/acs5", "dec/sf1")
        vintage (str): The year of the census data (e.g., "2020", "2019")
        vars (list): List of variables to retrieve from the Census API
        region (str): The geographic level to retrieve data for (e.g., "block group:*")
        regionin (str): The geographic filter for the region (e.g., "state:24 county:*")
        key (str): The Census API key for authentication
    
    Returns:
        pandas.DataFrame: DataFrame containing the census data with variables as columns and geographic units as rows
    """
    # Build the API URL
    base_url = f"https://api.census.gov/data/{vintage}/{name}"
    
    # Prepare parameters
    params = {
        "get": ",".join(vars),
        "for": region,
        "in": regionin,
        "key": key
    }
    
    # Make the request
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    # Convert to DataFrame
    data = response.json()
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df

def get_new_dec_data(vintage, state_fips):
    """Function for getting 2020 Decennial data using the Census API
    
    Args:
        vintage (str): The year of the decennial data (should be "2020")
        state_fips (str): The FIPS code for the state to retrieve data for
    
    Returns:
        pandas.DataFrame: DataFrame containing the 2020 decennial data with variables as columns and geographic units as rows
    """
    base_url = f"https://api.census.gov/data/{vintage}/dec/dhc?get=group(P18)&ucgid=pseudo(0400000US{state_fips}$1500000)"
    response = requests.get(base_url)
    response.raise_for_status()
    data = response.json()
    headers = data[0]
    rows = data[1:]
    df = pd.DataFrame(rows, columns=headers)
    return df

def pull_census_data(state_fips, year_ACS, year_DEC, ACS_table_codes, DEC_table_codes, key, verbose=1):
    """Function for pulling census data for given states
    
    Args:
        state_fips (str): The FIPS code for the state to retrieve data for
        year_ACS (str): The year of the ACS data
        year_DEC (str): The year of the decennial data
        verbose: If 1, print output. If 0, suppress output. Defaults to 1.
        
    Returns:
        Outputs data files in the census folder
    """

    config_path = os.path.join(BASE_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json file not found at {config_path}. Please create this file with the required configuration.")
    
    with open(config_path, "r") as f:
        config = json.load(f)

    # Get metadata
    ACS_metadata = get_census_metadata(name="acs/acs5", vintage=year_ACS)
    if year_DEC == 2020:
        DEC_metadata = get_census_metadata(name="dec/dhc", vintage=year_DEC)
    else:
        DEC_metadata = get_census_metadata(name="dec/sf1", vintage=year_DEC)
    
    # Save metadata to CSV files
    # ACS_metadata.to_csv('ACS_metadata.csv')
    # DEC_metadata.to_csv('DEC_metadata.csv')
    
    # Combine ACS and DEC metadata
    metadata_required  = pd.concat([ACS_metadata, DEC_metadata], ignore_index=True)
    
    # Build name-label mapping dictionary (kept in-memory; no file output)
    name_label_mapping = dict(zip(metadata_required["name"], metadata_required["label"]))
    
    printed_acs_source = False
    printed_dec_source = False
    acs_source_url = f"https://api.census.gov/data/{year_ACS}/acs/acs5"
    if year_DEC == 2020:
        dec_source_url = f"https://api.census.gov/data/{year_DEC}/dec/dhc"
    else:
        dec_source_url = f"https://api.census.gov/data/{year_DEC}/dec/sf1"
    for state_i in state_fips:
        # Process ACS table codes
        # Note: no 2023 data for B09019/B09020/B09021
        # so we need to use 2022 data for these tables if year_ACS is 2023
    
        # Effective vintage: use 2022 for B09019/B09020/B09021 when year_ACS is 2023 (no 2023 data)
        missing_years = {2023, 2024}
        FALLBACK_2022_CODES = {'B09019', 'B09020', 'B09021'}
        def _effective_acs_year(code, year_ACS):
            if year_ACS in missing_years and code in FALLBACK_2022_CODES:
                print(f"Using 2022 data for {code} because 2023 and 2024 data are not available via Census API")
                return 2022
            return year_ACS
        # Process ACS table codes
        table_codes = pd.DataFrame({
            "table_codes": [f"group({code})" for code in ACS_table_codes],
            "table_name": [f"ACSDT5Y{_effective_acs_year(code, year_ACS)}.{code}-Data" for code in ACS_table_codes],
            "vintage": [_effective_acs_year(code, year_ACS) for code in ACS_table_codes],
        })   
    
       
        # table_codes = pd.DataFrame({
        #     "table_codes": [f"group({code})" for code in ACS_table_codes],
        #     "table_name": [f"ACSDT5Y{year_ACS}.{code}-Data" for code in ACS_table_codes]
        # })
        
        for _, row in table_codes.iterrows():
            # print(f"Downloading {row['table_name']}")
            
            # Make the API call and get the data
            data = get_census_data(
                name="acs/acs5",
                vintage=row["vintage"],   # was: year_ACS
                vars=[row["table_codes"]],
                region="block group:*",
                regionin=f"state:{state_i} county:*",
                key=key
            )
            
            # Process data using Pandas
            data = data.drop(columns=["state", "county", "tract", "block group"], errors="ignore")
            
            cols = data.columns.to_list()
            if "GEO_ID" in cols and "NAME" in cols: # Move GEO_ID and NAME to the beginning
                cols.remove("GEO_ID")
                cols.remove("NAME")
                cols = ["GEO_ID", "NAME"] + cols
                data = data[cols]
            
            data = data.astype(str) # Convert all columns to string
            data_labels = ACS_metadata[ACS_metadata["name"].isin(data.columns)][["name", "label"]] # Get labels from metadata
            
            label_dict = dict(zip(data_labels["name"], data_labels["label"])) # Create a dictionary of column names to labels
            all_labels = {col: col for col in data.columns} # Ensure all columns have a label
            all_labels.update(label_dict)
            label_df = pd.DataFrame([all_labels]) # Create a DataFrame with labels as the first row
            
            data_with_labels = pd.concat([label_df, data], ignore_index=True)  # Combine label row with data
            data_with_labels = data_with_labels.loc[:, ~data_with_labels.columns.str.endswith(('EA', 'M', 'MA'))] # Remove columns ending with EA, M, or MA
            
            # Create destination folder and save to CSV
            state_abbr = fips_info(state_i)["abbr"]
            destination_folder = os.path.join(OUTPUT_DIR, "census", state_abbr.upper())
            os.makedirs(destination_folder, exist_ok=True)
            file_name = f"{row['table_name']}.csv"
            file_path = os.path.join(destination_folder, file_name)
            data_with_labels.to_csv(file_path, index=False)
            if verbose:
                if not printed_acs_source:
                    print(f"-- Downloading from {acs_source_url}")
                    printed_acs_source = True
                print(f"-- census/{state_abbr.upper()}/{file_name}")
        
        # Process DEC table codes        
        table_codes = pd.DataFrame({
            "table_codes": [f"group({code})" for code in DEC_table_codes],
            "table_name": [f"DECENNIALSF1{year_DEC}.{code}-Data" for code in DEC_table_codes]
        })
        
        for _, row in table_codes.iterrows():
            # print(f"Downloading {row['table_name']}")
            
            # Make the API call and get the data
            if year_DEC == 2020:
                data = get_new_dec_data(year_DEC, state_i)
                # Remove columns ending in 'A'
                data = data.loc[:, ~data.columns.str.endswith('A')]
                data.drop(columns=["ucgid"], inplace=True)
            
            else:
                data = get_census_data(
                    name="dec/sf1",
                    vintage=year_DEC,
                    vars=[row["table_codes"]],
                    region="block group:*",
                    regionin=f"state:{state_i} county:*",
                    key=key
                )
            
            # Process data using Pandas
            data = data.drop(columns=["state", "county", "tract", "block group"], errors="ignore")
            cols = data.columns.to_list() 
            if "GEO_ID" in cols and "NAME" in cols: # Move GEO_ID and NAME to the beginning of the dataframe
                cols.remove("GEO_ID")
                cols.remove("NAME")
                cols = ["NAME","GEO_ID"] + cols
                data = data[cols]
        
            data = data.astype(str) # Convert all columns to string
            data_labels = DEC_metadata[DEC_metadata["name"].isin(data.columns)][["name", "label"]] # Get labels from metadata
            
            
            label_dict = dict(zip(data_labels["name"], data_labels["label"])) # Create a dictionary of column names to labels
            all_labels = {col: col for col in data.columns} # Ensure all columns have a label
            all_labels.update(label_dict)
            label_df = pd.DataFrame([all_labels]) # Create a DataFrame with labels as the first row
            
            data_with_labels = pd.concat([label_df, data], ignore_index=True) # Combine label row with data
            
            if year_DEC == 2020: # 2020 Decennial data has a different format for the labels
                data_with_labels.iloc[0, 2:] = data_with_labels.iloc[0, 2:].str.replace(':', '') # Strip ":" from the second row (labels) starting from fourth column
                data_with_labels.iloc[0, 2:] = data_with_labels.iloc[0, 2:].str[3:] # Remove first two characters from each string in the second row starting from fourth column
            
            data_with_labels = data_with_labels.loc[:, ~data_with_labels.columns.str.endswith('ERR')] # Remove columns ending with ERR
            
            # Create destination folder and save to CSV
            state_abbr = fips_info(state_i)["abbr"]
            destination_folder = os.path.join(OUTPUT_DIR, "census", state_abbr.upper())
            os.makedirs(destination_folder, exist_ok=True)
            file_name = f"{row['table_name']}.csv"
            file_path = os.path.join(destination_folder, file_name)
            data_with_labels.to_csv(file_path, index=False)
            if verbose:
                if not printed_dec_source:
                    print(f"-- Downloading from {dec_source_url}")
                    printed_dec_source = True
                print(f"-- census/{state_abbr.upper()}/{file_name}")

def pull_pums_data(states, year, verbose=1):
    """Function for pulling PUMS microdata for given states
    
    Args:
        states (list): List of state abbreviations
        year (str): The year of the PUMS data
        verbose: If 1, print output. If 0, suppress output. Defaults to 1.
        
    Returns:
        Outputs data files in the pums folder
        
    Notes: 2024 and 2025 urls require password to access. If main year > 2023, use 2023 data.
    This means that the combinatorial optimization algorithm will combine households from 2023 PUMS data
    such that the set reasonably approximates Census distributions for target variables in the year specified in the config file.
    This can still produce a population that is reasonably representative of the target year.
    """
    
    states = [s.lower() for s in states]

    file_urls = []
    
    # Set the URL of the file to download
    for state in states:
        if year > 2023:
            year = 2023
        file_urls.append((state, f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/5-Year/csv_h{state}.zip"))
        file_urls.append((state, f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/5-Year/csv_p{state}.zip"))
        # if verbose:
        #     print('file_urls', file_urls)
    printed_source_header = False
    pums_source_url = f"https://www2.census.gov/programs-surveys/acs/data/pums/{year}/5-Year/"
    if verbose:
        print("\n*** Running DownloadData.pull_pums_data() ***")
    for state_i in states:
        
        urls_list = [url for state, url in file_urls if state == state_i]
        
        for url in urls_list:
            download_url = url
            
            # Specify the destination folder and file name
            destination_folder = os.path.join(OUTPUT_DIR, "pums")
            file_name = os.path.basename(download_url)
            destination_file = os.path.join(destination_folder, file_name)
            # print(destination_folder)
            # Create the destination folder if it doesn't exist
            os.makedirs(destination_folder, exist_ok=True)
            
            # Download the file
            try_curl_cffi(download_url, destination_file)
            
            # Extract the contents of the zip file
            with zipfile.ZipFile(destination_file, 'r') as zip_ref:
                zip_ref.extractall(destination_folder)
            
            # Remove the zip file
            os.remove(destination_file)
            
            state_fips = fips_info(state_i.upper(), reverse=True)['fips']
            if file_name == f'csv_h{state_i}.zip':
                df_h = pd.read_csv(f"{destination_folder}/psam_h{state_fips}.csv", low_memory=False)
                if year >= 2020:
                    # ACCESSINET(formerly ACCESS), TYPEHUGQ (formerly TYPE).
                    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/variable_changes/ACS2016-2020_PUMS_Variable_Changes_and_Explanations.pdf
                    df_h.rename(columns={'ACCESSINET': 'ACCESS'}, inplace=True) # 
                    df_h.rename(columns={'TYPEHUGQ': 'TYPE'}, inplace=True)
                if year >= 2021:
                    # FES variable deleted in 2021. Can be recreated from WORKSTAT.
                    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/variable_changes/ACS2017-2021_PUMS_Variable_Changes_and_Explanations.pdf
                    df_h.loc[(df_h['WORKSTAT'] == 1)|(df_h['WORKSTAT'] == 2)|(df_h['WORKSTAT'] == 5), 'FES'] = 1
                    df_h.loc[(df_h['WORKSTAT'] == 3)|(df_h['WORKSTAT'] == 6), 'FES'] = 2
                    df_h.loc[(df_h['WORKSTAT'] == 7)|(df_h['WORKSTAT'] == 8), 'FES'] = 3
                    df_h.loc[(df_h['WORKSTAT'] == 9), 'FES'] = 4
                    df_h.loc[(df_h['WORKSTAT'] == 10)|(df_h['WORKSTAT'] == 11), 'FES'] = 5
                    df_h.loc[(df_h['WORKSTAT'] == 12), 'FES'] = 6
                    df_h.loc[(df_h['WORKSTAT'] == 13)|(df_h['WORKSTAT'] == 14), 'FES'] = 7
                    df_h.loc[(df_h['WORKSTAT'] == 15), 'FES'] = 8 
                if year >= 2022:
                    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/variable_changes/ACS2018-2022_PUMS_Variable_Changes_and_Explanations.pdf
                    # We may want to use PUMA10 since that's the definition we were using before
                    df_h.rename(columns={'PUMA10': 'PUMA'}, inplace=True)
                if year >= 2023:
                    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/variable_changes/ACS2019-2023_PUMS_Variable_Changes_and_Explanations.pdf
                    # ST variable now called STATE
                    df_h.rename(columns={'STATE': 'ST'}, inplace=True)
                df_h.to_csv(f"{destination_folder}/psam_h{state_fips}.csv", index=False)
                if verbose:
                    if not printed_source_header:
                        print(f"-- Downloading from {pums_source_url}")
                        printed_source_header = True
                    print(f"-- pums/psam_h{state_fips}.csv")
            else: # psam_p{state_fips}.csv
                df_p = pd.read_csv(f"{destination_folder}/psam_p{state_fips}.csv", low_memory=False)
                if year >= 2022:
                    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/variable_changes/ACS2018-2022_PUMS_Variable_Changes_and_Explanations.pdf
                    # In 2022, these variables were renamed to reflect new definitions from 2020 census
                    # We may want to use PUMA10 since that's the definition we were using before
                    df_p.rename(columns={'POWPUMA10': 'POWPUMA'}, inplace=True)
                    df_p.rename(columns={'PUMA10': 'PUMA'}, inplace=True)
                if year >= 2023:
                    # https://www2.census.gov/programs-surveys/acs/tech_docs/pums/variable_changes/ACS2019-2023_PUMS_Variable_Changes_and_Explanations.pdf
                    # ST variable now called STATE
                    # WKW ("Weeks worked during past 12 months") variable deleted. Can now use WKWN
                    df_p.rename(columns={'STATE': 'ST'}, inplace=True)
                    df_p.rename(columns={'WKWN': 'WKW'}, inplace=True)
                df_p.to_csv(f"{destination_folder}/psam_p{state_fips}.csv", index=False)
                if verbose:
                    if not printed_source_header:
                        print(f"-- Downloading from {pums_source_url}")
                        printed_source_header = True
                    print(f"-- pums/psam_p{state_fips}.csv")

def download_shapefiles(state_fips, year, verbose=1):
    """Function for downloading shapefiles from census web server
    
    Args:
        state_fips (list): List of state abbreviations
        year (str): The year of the shapefiles
        verbose: If 1, print output. If 0, suppress output. Defaults to 1
        
    Returns:
        Outputs data files in the geo folder
    """
    
    file_urls = []
    
    # Set the URL of the file to download
    for state in state_fips:
        file_urls.append((state, f"https://www2.census.gov/geo/tiger/TIGER{year}/BG/tl_{year}_{state}_bg.zip"))
    
    printed_source_header = False
    for state_i in state_fips:
        
        if verbose:
            print("\n*** Running DownloadData.download_shapefiles() ***")
        
        urls_list = [url for state, url in file_urls if state == state_i]
        
        for url in urls_list:
            download_url = url
            
            # Specify the destination folder and file name
            destination_folder = os.path.join(OUTPUT_DIR, "geo")
            
            file_name = os.path.basename(download_url)
            destination_file = os.path.join(destination_folder, file_name)
            
            # Create the destination folder if it doesn't exist
            os.makedirs(destination_folder, exist_ok=True)
            
            # Download the file
            try_curl_cffi(download_url, destination_file)
            
            # Extract the contents of the zip file
            with zipfile.ZipFile(destination_file, 'r') as zip_ref:
                zip_ref.extractall(destination_folder)
            
            if verbose:
                if not printed_source_header:
                    print(f"-- Downloading from https://www2.census.gov/geo/tiger/TIGER{year}/BG/")
                    printed_source_header = True
                print(f"-- geo/{file_name}")

def pull_LODES(states_main, states_aux, year, verbose=1):
    """Function for pulling LEHD LODES data (commuting patterns) for given states
    
    Args:
        states_main (list): List of state abbreviations for the main states
        states_aux (list): List of state abbreviations for the auxiliary states
        year (str): The year of the LODES data
        verbose: If 1, print output. If 0, suppress output. Defaults to 1.
        
    Returns:
        Outputs data files in the work folder
        
    Notes: No LODES data for 2024 or 2025, so use 2023 data if year > 2023.
    """
    printed_source_header = False
    if verbose:
        print("\n*** Running DownloadData.pull_LODES() ***")
    # Determine version based on year
    version = "LODES8" if year >= 2020 else "LODES7"
    
    # For the states on the "main" list, download OD main JT01, OD aux JT01, and WAC S000 JT01
    for state_i in states_main:
        # print(f"*** Downloading LODES data for state = {state_i.upper()} ***")
        state_i = state_i.lower()
        state_dir = os.path.join(OUTPUT_DIR, "work")
        os.makedirs(state_dir, exist_ok=True)
        
        # Download and save OD main JT01
        # print(f"downloading lodes od main {state_i}")
        
        # For OD main JT01 
        if year > 2023:
            year = 2023
        od_main_url = f"https://lehd.ces.census.gov/data/lodes/{version}/{state_i}/od/{state_i}_od_main_JT01_{year}.csv.gz"
        outfile = os.path.join(state_dir, f"{state_i}_od_main_JT01_{year}.csv.gz")
        try_download(od_main_url, outfile)
        
        # Process the file to match old format
        with gzip.open(outfile, 'rt') as f_in:
            with gzip.open(outfile + '.tmp', 'wt') as f_out:
                # Read header
                header = f_in.readline().strip()
                # Write new header with year and state
                f_out.write("year,state," + header + "\n")
                
                # Process each line
                for line in f_in:
                    parts = line.strip().split(',')
                    # Truncate geocodes to 12 digits
                    parts[0] = parts[0][:12]  # w_geocode
                    parts[1] = parts[1][:12]  # h_geocode
                    # Add year and state
                    f_out.write(f"{year},{state_i.upper()}," + ",".join(parts) + "\n")
        if verbose:
            if not printed_source_header:
                print("-- Downloading from https://lehd.ces.census.gov/data/lodes/")
                printed_source_header = True
            print(f"-- work/{state_i}_od_main_JT01_{year}.csv.gz")
        
        # Replace original file with processed file
        os.replace(outfile + '.tmp', outfile)
        
        # For OD aux JT01
        # print(f"downloading lodes od aux {state_i}")
        od_aux_url = f"https://lehd.ces.census.gov/data/lodes/{version}/{state_i}/od/{state_i}_od_aux_JT01_{year}.csv.gz"
        outfile = os.path.join(state_dir, f"{state_i}_od_aux_JT01_{year}.csv.gz")
        try_download(od_aux_url, outfile)
        
        # Process the file to match old format
        with gzip.open(outfile, 'rt') as f_in:
            with gzip.open(outfile + '.tmp', 'wt') as f_out:
                header = f_in.readline().strip() # Read header
                f_out.write("year,state," + header + "\n")  # Write new header with year and state
                
                # Process each line
                for line in f_in:
                    parts = line.strip().split(',')
                    # Truncate geocodes to 12 digits
                    parts[0] = parts[0][:12]  # w_geocode
                    parts[1] = parts[1][:12]  # h_geocode
                    # Add year and state
                    f_out.write(f"{year},{state_i.upper()}," + ",".join(parts) + "\n")
        if verbose:
            if not printed_source_header:
                print("-- Downloading from https://lehd.ces.census.gov/data/lodes/")
                printed_source_header = True
            print(f"-- work/{state_i}_od_aux_JT01_{year}.csv.gz")
        
        # Replace original file with processed file
        os.replace(outfile + '.tmp', outfile)
        
        # For WAC S000 JT01
        # print(f"downloading lodes wac {state_i}")
        wac_url = f"https://lehd.ces.census.gov/data/lodes/{version}/{state_i}/wac/{state_i}_wac_S000_JT01_{year}.csv.gz"
        outfile = os.path.join(state_dir, f"{state_i}_wac_S000_JT01_{year}.csv.gz")
        try_download(wac_url, outfile)
        
        # Process the file to match old format
        with gzip.open(outfile, 'rt') as f_in:
            with gzip.open(outfile + '.tmp', 'wt') as f_out:
                header = f_in.readline().strip() # Read header
                f_out.write("year,state," + header + "\n")  # Write new header with year and state
                
                # Process each line
                for line in f_in:
                    parts = line.strip().split(',')
                    # Truncate geocode to 12 digits
                    parts[0] = parts[0][:12]  # w_geocode
                    # Add year and state
                    f_out.write(f"{year},{state_i.upper()}," + ",".join(parts) + "\n")
        if verbose:
            if not printed_source_header:
                print("-- Downloading from https://lehd.ces.census.gov/data/lodes/")
                printed_source_header = True
            print(f"-- work/{state_i}_wac_S000_JT01_{year}.csv.gz")
        
        # Replace original file with processed file
        os.replace(outfile + '.tmp', outfile)
    
    # For the states on the "aux" list just download OD aux JT01
    for state_i in states_aux:
        state_dir = os.path.join(OUTPUT_DIR, "work")
        os.makedirs(state_dir, exist_ok=True)
        
        # Download and save OD aux JT01
        # print(f"downloading lodes od aux {state_i}")
        od_aux_url = f"https://lehd.ces.census.gov/data/lodes/{version}/{state_i.lower()}/od/{state_i.lower()}_od_aux_JT01_{year}.csv.gz"
        outfile = os.path.join(state_dir, f"{state_i.lower()}_od_aux_JT01_{year}.csv.gz")
        try_download(od_aux_url, outfile)
        
        # Process the file to match old format
        with gzip.open(outfile, 'rt') as f_in:
            with gzip.open(outfile + '.tmp', 'wt') as f_out:
                header = f_in.readline().strip() # Read header
                f_out.write("year,state," + header + "\n")  # Write new header with year and state
                
                # Process each line
                for line in f_in:
                    parts = line.strip().split(',')
                    # Truncate geocodes to 12 digits
                    parts[0] = parts[0][:12]  # w_geocode
                    parts[1] = parts[1][:12]  # h_geocode
                    # Add year and state
                    f_out.write(f"{year},{state_i.upper()}," + ",".join(parts) + "\n")
        if verbose:
            if not printed_source_header:
                print("-- Downloading from https://lehd.ces.census.gov/data/lodes/")
                printed_source_header = True
            print(f"-- work/{state_i.lower()}_od_aux_JT01_{year}.csv.gz")
        
        # Replace original file with processed file
        os.replace(outfile + '.tmp', outfile)

def download_cbp_data(verbose=1):
    """Function for downloading CBP data
    
    Args:
        verbose: If 1, print output. If 0, suppress output. Defaults to 1.
        
    Returns:
        Outputs data files in the work folder
    """
    if verbose:
        print("\n*** Running DownloadData.download_cbp_data() ***")
    cbp_dir = os.path.join(OUTPUT_DIR, "work")
    os.makedirs(cbp_dir, exist_ok=True)
    
    url = "https://www2.census.gov/programs-surveys/cbp/datasets/2016/cbp16co.zip"
    outfile = os.path.join(cbp_dir, "cbp16co.zip")
    try_download(url, outfile)
    if verbose:
        print("-- Downloading from https://www2.census.gov/programs-surveys/cbp/datasets/2016/")
        print("-- work/cbp16co.zip")

def download_ct_puma_crosswalk(main_year, verbose=1):
    """Function for downloading Census Tract to PUMA crosswalk file
    
    Args:
        main_year (str): The year of the crosswalk file
        verbose: If 1, print output. If 0, suppress output. Defaults to 1.
        
    Returns:
        Outputs data files in the geo folder
    """
    if verbose:
        print("\n*** Running DownloadData.download_ct_puma_crosswalk() ***")
    geo_dir = os.path.join(OUTPUT_DIR, "geo")
    os.makedirs(geo_dir, exist_ok=True)
    
    # # https://www2.census.gov/geo/docs/maps-data/data/rel2020/2020_Census_Tract_to_2020_PUMA.txt
    if main_year >= 2020:
        url = "https://www2.census.gov/geo/docs/maps-data/data/rel2020/2020_Census_Tract_to_2020_PUMA.txt"
        outfile = os.path.join(geo_dir, "2020_Census_Tract_to_2020_PUMA.txt")
    else:
        url = "https://www2.census.gov/geo/docs/maps-data/data/rel/2010_Census_Tract_to_2010_PUMA.txt"
        outfile = os.path.join(geo_dir, "2010_Census_Tract_to_2010_PUMA.txt")
    
    # Use text download for these web pages since they display data in browser
    try_download_text(url, outfile)
    if verbose:
        print("-- Downloading from https://www2.census.gov/geo/docs/maps-data/data/rel/")
        print(f"-- geo/{os.path.basename(outfile)}")
        
    # urls2018 = ["https://mcdc.missouri.edu/temp/geocorr2018_2523203354.csv", # geocorr2018_puma_to_county.csv
    #             "https://mcdc.missouri.edu/temp/geocorr2018_2523207598.csv", # geocorr2018_puma_to_cbsa.csv
    #             "https://mcdc.missouri.edu/temp/geocorr2018_2523205248.csv", # geocorr2018_puma_urban_rural.csv
    #             "https://mcdc.missouri.edu/temp/geocorr2018_2523202710.csv", # geocorr2018_cbg_to_cbsa.csv
    #             "https://mcdc.missouri.edu/temp/geocorr2018_2523209378.csv"] # geocorr2018_cbg_urban_rural.csv

    # urls2020 = ["https://mcdc.missouri.edu/temp/geocorr2022_2523203649.csv", # geocorr2022_puma_to_county.csv
    #             "https://mcdc.missouri.edu/temp/geocorr2022_2523203792.csv", # geocorr2022_puma_to_cbsa.csv
    #             "https://mcdc.missouri.edu/temp/geocorr2022_2523202650.csv", # geocorr2022_puma_urban_rural.csv
    #             "https://mcdc.missouri.edu/temp/geocorr2022_2523203752.csv", # geocorr2022_cbg_to_cbsa.csv
    #             "https://mcdc.missouri.edu/temp/geocorr2022_2523208908.csv"] # geocorr2022_cbg_urban_rural.csv

    # filenames = ["geocorr2018_puma_to_county.csv",
    #             "geocorr2018_puma_to_cbsa.csv",
    #             "geocorr2018_puma_urban_rural.csv",
    #             "geocorr2018_cbg_to_cbsa.csv",
    #             "geocorr2018_cbg_urban_rural.csv"]

    # if main_year < 2020:
    #     urls = urls2018
    # else:
    #     urls = urls2020

    # for url, filename in zip(urls, filenames):
    #     outfile = os.path.join(geo_dir, filename)

    #     try:
    #         response = requests.get(url)
    #         response.raise_for_status()
            
    #         with open(outfile, 'wb') as f:
    #             f.write(response.content)
            
    #         # print(f"Downloaded {outfile}")
    #     except Exception as e:
    #         print(f"Failed to download {url}: {e}")
            
def geocorr_files(verbose=1):
    """Copy geocorr files from the local 'geocorr' folder into the 'geo' folder.

    If main_year < 2020, copies only files matching 'geocorr2018*'.
    If main_year >= 2020, copies only files matching 'geocorr2022*'.
    
    Returns:
        Outputs data files in the geo folder
    """
    if verbose:
        print("\n*** Running DownloadData.geocorr_files() ***")

    # Load main_year from config to decide which geocorr files to copy
    config_path = os.path.join(BASE_DIR, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"config.json file not found at {config_path}. Please create this file with the required configuration."
        )
    with open(config_path, "r") as f:
        config = json.load(f)

    if "main_year" not in config:
        raise KeyError("main_year not found in config. Please add the main year to the configuration.")

    try:
        main_year = int(config["main_year"])
    except Exception:
        raise ValueError("main_year in config.json must be an integer.")

    if main_year < 2020:
        prefix = "geocorr2018"
    else:
        prefix = "geocorr2022"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    source_dir = os.path.join(script_dir, "geocorr")
    dest_dir = os.path.join(OUTPUT_DIR, "geo")

    if not os.path.isdir(source_dir):
        print(f"Source folder not found: {source_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)

    copied_any = False
    for entry in os.listdir(source_dir):
        if not entry.startswith(prefix):
            continue
        src_path = os.path.join(source_dir, entry)
        if os.path.isfile(src_path):
            dst_path = os.path.join(dest_dir, entry)
            shutil.copy2(src_path, dst_path)
            copied_any = True

    if not copied_any and verbose:
        print(f"No files matching '{prefix}*' to copy from {source_dir}")
    elif copied_any and verbose:
        print("-- Copying package files from src/geopops/geocorr")
        print(f"-- geo/{prefix}*")

def download_school_data(main_year, verbose=1):
    """Function for downloading school data
    
    Args:
        main_year (str): The year of the school data
        verbose: If 1, print output. If 0, suppress output. Defaults to 1.
        
    Returns:
        Outputs data files in the school folder
    
    Notes: This function uses the python package 'zipfile' to extract the zip file.
    This will cause an error "NotImplementedError" if the interpreter used to build the environment 
    does not have full LZMA support. 
    
    MacOS workaround: use the system 'unzip' command instead of 'zipfile'.
    Windows workaround: use Expand-Archive or 7‑Zip
    """
    school_dir = os.path.join(OUTPUT_DIR, "school")
    os.makedirs(school_dir, exist_ok=True)
    
    if verbose:
        print("\n*** Running DownloadData.download_school_data() ***")
    
    # Download school location data
    # https://nces.ed.gov/programs/edge/data/EDGE_GEOCODE_PUBLICSCH_2021.zip
    url = f"https://nces.ed.gov/programs/edge/data/EDGE_GEOCODE_PUBLICSCH_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}.zip"

    # Download the zip file
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes

    # Save the file directly to the school folder
    zip_filename = f"EDGE_GEOCODE_PUBLICSCH_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}.zip"
    zip_path = os.path.join(school_dir, zip_filename)

    # Download the file directly
    with open(zip_path, 'wb') as f:
        f.write(response.content)
    if verbose:
        print("-- Downloading from https://nces.ed.gov/programs/edge/data/")
        print(f"-- school/{zip_filename}")

    # Extract the zip file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(school_dir)    

    # Move files from extracted folder to main school folder
    extracted_folder = os.path.join(school_dir, f"EDGE_GEOCODE_PUBLICSCH_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}")
    files_to_move = [f"EDGE_GEOCODE_PUBLICSCH_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}.xlsx",
                    "Shapefiles_SCH", "Shapefile_SCH"] # if year > 2019, uses Shapefile_SCH (singular) folder instead of Shapefiles_SCH (plural) 

    for item_to_move in files_to_move:
        source_path = os.path.join(extracted_folder, item_to_move)
        dest_path = os.path.join(school_dir, item_to_move)
        if os.path.exists(source_path):
            # If destination already exists, remove it first
            if os.path.exists(dest_path):
                if os.path.isdir(dest_path):
                    shutil.rmtree(dest_path)
                else:
                    os.remove(dest_path)
            os.rename(source_path, dest_path)
        # else:
        #     print(f"Item {item_to_move} not found")

    # Normalize school shapefile folder name to 'Shapefiles_SCH'
    
    singular = os.path.join(school_dir, "Shapefile_SCH")
    plural   = os.path.join(school_dir, "Shapefiles_SCH")

    if os.path.isdir(singular):
        # If a plural folder already exists, remove or merge as you prefer; simplest is remove it first
        if os.path.isdir(plural):
            shutil.rmtree(plural)
        os.rename(singular, plural)

    # Delete files (zip file and empty extracted folder)
    files_to_delete = [
        zip_path,  # The zip file
        extracted_folder  # The entire extracted folder (now empty)
    ]

    for item_to_delete in files_to_delete:
        if os.path.exists(item_to_delete):
            if os.path.isdir(item_to_delete):
                # Recursively delete directory
                for root, dirs, files in os.walk(item_to_delete, topdown=False):
                    for file_name in files:
                        os.remove(os.path.join(root, file_name))
                    for dir_name in dirs:
                        os.rmdir(os.path.join(root, dir_name))
                os.rmdir(item_to_delete)
            else:
                os.remove(item_to_delete)
        # else:
        #     print(f"Item {os.path.basename(item_to_delete)} not found")
            
    # Download school enrollment data
    # print("Downloading school enrollment data... takes a while")

    # URLs for school enrollment data
    
    if main_year == 2019:
        en_str = "082120"
    elif main_year == 2020:
        en_str = "080621"
    elif main_year == 2021:
        en_str = "071722"
    elif main_year == 2022:
        en_str = "083023"
    elif main_year == 2023:
        en_str = "073124"
    elif main_year == 2024:
        en_str = "073025"
    
    enrollment_urls = [
        f"https://nces.ed.gov/ccd/Data/zip/ccd_sch_029_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}_w_1a_{en_str}.zip",  # Directory
        f"https://nces.ed.gov/ccd/Data/zip/ccd_SCH_052_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}_l_1a_{en_str}.zip",  # Membership
        f"https://nces.ed.gov/ccd/Data/zip/ccd_sch_059_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}_l_1a_{en_str}.zip"   # Staff
    ]

    # Download and extract each enrollment data file
    for url in enrollment_urls:
        # Extract filename from URL
        filename = url.split('/')[-1]
        zip_path = os.path.join(school_dir, filename)
        
        # print(f"Downloading {filename}...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Download the file (overwrite if exists)
        with open(zip_path, 'wb') as f:
            f.write(response.content)
        
        # Extract the outer zip file
        # This will cause an error "NotImplementedError" if the interpreter used to build the environment 
        # does not have full LZMA support
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(school_dir)
        except NotImplementedError:
            system = platform.system()
            zip_basename = os.path.basename(zip_path)

            if system == "Darwin":  # macOS
                # Use system 'unzip'
                subprocess.run(
                    ["unzip", "-o", zip_basename],
                    cwd=school_dir,
                    check=True,
                )

            elif system == "Windows":
                # Use PowerShell Expand-Archive
                # Build a safe PowerShell command string
                src = os.path.join(school_dir, zip_basename)
                dst = school_dir
                ps_cmd = (
                    f"Expand-Archive -LiteralPath {shlex.quote(src)} "
                    f"-DestinationPath {shlex.quote(dst)} -Force"
                )
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_cmd],
                    check=True,
                )

            else:
                # Other platforms: re-raise with guidance
                raise RuntimeError(
                    "ZIP uses a compression method not supported by this Python. "
                    "On Unix-like systems, install a Python build with LZMA support "
                    "or extract the archive manually using system tools."
                )

        # Handle nested CSV and SAS ZIPs created by NCES (e.g. *_CSV.zip, *_SAS.zip)
        for nested_name in os.listdir(school_dir):
            if not nested_name.lower().endswith(".zip"):
                continue

            nested_path = os.path.join(school_dir, nested_name)
            upper_name = nested_name.upper()

            # CSV archives: extract to get the actual .csv, then delete the inner zip
            if upper_name.endswith("_CSV.ZIP"):
                try:
                    with zipfile.ZipFile(nested_path, "r") as nested_zip:
                        nested_zip.extractall(school_dir)
                except NotImplementedError:
                    system = platform.system()
                    if system == "Darwin":
                        subprocess.run(
                            ["unzip", "-o", nested_name],
                            cwd=school_dir,
                            check=True,
                        )
                    elif system == "Windows":
                        src = nested_path
                        dst = school_dir
                        ps_cmd = (
                            f"Expand-Archive -LiteralPath {shlex.quote(src)} "
                            f"-DestinationPath {shlex.quote(dst)} -Force"
                        )
                        subprocess.run(
                            ["powershell", "-NoProfile", "-Command", ps_cmd],
                            check=True,
                        )
                    else:
                        raise RuntimeError(
                            "Nested CSV ZIP uses a compression method not supported by this Python. "
                            "On Unix-like systems, install a Python build with LZMA support "
                            "or extract the archive manually using system tools."
                        )

                # Remove the nested CSV zip after extraction
                if os.path.exists(nested_path):
                    os.remove(nested_path)

            # SAS archives: we don't need SAS files, just delete the zip
            elif upper_name.endswith("_SAS.ZIP"):
                if os.path.exists(nested_path):
                    os.remove(nested_path)

        # Check what files were extracted directly to the school folder
        # print(f"Files in school folder after extraction: {[f for f in os.listdir(path) if f.endswith('.csv') or f.endswith('.sas7bdat')]}")
        
        # Delete .sas7bdat files that were extracted directly to the school folder
        for item in os.listdir(school_dir):
            if item.endswith('.sas7bdat'):
                file_path = os.path.join(school_dir, item)
                os.remove(file_path)
                # print(f"Deleted {item} from school folder")
        
        # Delete the zip file
        files_to_delete = [zip_path]
        
        for item_to_delete in files_to_delete:
            if os.path.exists(item_to_delete):
                if os.path.isdir(item_to_delete):
                    # Recursively delete directory
                    for root, dirs, files in os.walk(item_to_delete, topdown=False):
                        for file_name in files:
                            os.remove(os.path.join(root, file_name))
                        for dir_name in dirs:
                            os.rmdir(os.path.join(root, dir_name))
                    os.rmdir(item_to_delete)
                else:
                    os.remove(item_to_delete)
            else:
                print(f"Item {os.path.basename(item_to_delete)} not found")

class DownloadData:
    """Orchestrates data downloads using the existing functions in this module.

    This class provides a simple programmatic interface so callers can import
    and run the full download workflow (or pieces of it) from another module.
    """

    def __init__(self, config=None, base_dir=None, verbose=1, auto_run=True):
        """Create a downloader.

        Args:
            config: Optional dict with configuration. If provided, takes
                precedence over loading from config.json.
            base_dir: Optional base directory to use for relative paths.
                Defaults to this file's directory.
            verbose: If 1, print output. If 0, suppress output. Defaults to 1.
        """
        self.verbose = verbose
        self.base_dir = base_dir if base_dir is not None else BASE_DIR
        if config is not None:
            self.config = config
        else:
            cfg_path = os.path.join(self.base_dir, "config.json")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"config.json file not found at {cfg_path}. Please create this file with the required configuration.")
            with open(cfg_path, "r") as f:
                self.config = json.load(f)
        # Initialize OUTPUT_DIR from config["path"] (fallback to package dir)
        global OUTPUT_DIR
        OUTPUT_DIR = self.config.get("path", self.base_dir)
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        if auto_run:
            self.run_all()
        
    def run_all(self):
        """Run the full download workflow using the loaded configuration."""
        config = self.config
        if self.verbose:
            print("")
            print("============================================================")
            print("Running DownloadData()")
            print("============================================================")
            
        # Basic validation for keys required by multiple steps
        if "census_api_key" not in config:
            raise KeyError("census_api_key not found in config. Please add your Census API key to the configuration.")
        if "main_year" not in config:
            raise KeyError("main_year not found in config. Please add the main year to the configuration.")
        if "decennial_year" not in config:
            raise KeyError("decennial_year not found in config. Please add the decennial year to the configuration.")

        # Delegate to per-step methods so they can also be called individually
        self.pull_census_data()
        self.pull_pums_data()
        self.download_shapefiles()
        self.pull_LODES()
        self.download_ct_puma_crosswalk()
        self.geocorr_files()
        self.download_cbp_data()
        self.download_school_data()
        if self.verbose:
            print("\nAll DownloadData() steps complete")

    def _main_state_fips_and_abbr(self):
        """Derive main state FIPS codes and abbreviations from config['geos']."""
        raw_geos = self.config["geos"]
        main_fips = []
        for geo_value in raw_geos:
            geo_str = str(geo_value)
            if len(geo_str) < 2:
                geo_str = geo_str.zfill(2)
            main_fips.append(geo_str[:2])
        main_fips = list(set(main_fips))
        main_abbr = [abbr for abbr in fips_info(main_fips)["abbr"] if abbr is not None]
        return main_fips, main_abbr

    def pull_census_data(self):
        """Run the census download step using this instance's configuration."""
        config = self.config
        v = self.verbose
        if v:
            print("*** Running DownloadData.pull_census_data() ***")

        if "census_api_key" not in config:
            raise KeyError("census_api_key not found in config.")
        if "main_year" not in config:
            raise KeyError("main_year not found in config.")
        if "decennial_year" not in config:
            raise KeyError("decennial_year not found in config.")

        key = config["census_api_key"]
        year_ACS = config["main_year"]
        year_DEC = config["decennial_year"]

        main_fips, _ = self._main_state_fips_and_abbr()
        state_fips = main_fips

        ACS_table_codes = config["acs_required"]
        if year_DEC == 2020:
            DEC_table_codes = [config["dec_required"][1]]  # "P18" for 2020
        else:
            DEC_table_codes = [config["dec_required"][0]]  # "P43" otherwise

        pull_census_data(
            state_fips=state_fips,
            year_ACS=year_ACS,
            year_DEC=year_DEC,
            ACS_table_codes=ACS_table_codes,
            DEC_table_codes=DEC_table_codes,
            key=key,
            verbose=v,
        )

    def pull_pums_data(self):
        """Run the PUMS download step using this instance's configuration."""
        config = self.config
        v = self.verbose

        _, main_abbr = self._main_state_fips_and_abbr()

        if "use_pums" not in config or config["use_pums"] is None:
            states = main_abbr
        else:
            use_pums_fips = [str(v).zfill(2) for v in config["use_pums"]]
            states = [abbr for abbr in fips_info(use_pums_fips)["abbr"] if abbr is not None]

        if "main_year" not in config:
            raise KeyError("main_year not found in config.")
        year = config["main_year"]

        pull_pums_data(states=states, year=year, verbose=v)

    def download_shapefiles(self):
        """Run the shapefile download step using this instance's configuration."""
        config = self.config
        v = self.verbose

        main_fips, _ = self._main_state_fips_and_abbr()
        state_fips = main_fips

        if "main_year" not in config:
            raise KeyError("main_year not found in config.")
        year = config["main_year"]

        download_shapefiles(state_fips, year, v)

    def pull_LODES(self):
        """Run the LODES download step using this instance's configuration."""
        config = self.config
        v = self.verbose

        main_fips, main_abbr = self._main_state_fips_and_abbr()
        states_main = main_abbr

        if "commute_states" not in config or config["commute_states"] is None:
            states_aux = []
        else:
            aux_states_fips = [str(v).zfill(2) for v in config["commute_states"]]
            aux_states_filtered = [state for state in aux_states_fips if state not in main_fips]
            states_aux = [abbr for abbr in fips_info(aux_states_filtered)["abbr"] if abbr is not None]

        if "main_year" not in config:
            raise KeyError("main_year not found in config.")
        year = config["main_year"]

        pull_LODES(states_main, states_aux, year, v)

    def download_ct_puma_crosswalk(self):
        """Run the CT→PUMA crosswalk download step using this instance's configuration."""
        config = self.config
        v = self.verbose

        if "main_year" not in config:
            raise KeyError("main_year not found in config.")
        main_year = config["main_year"]

        download_ct_puma_crosswalk(main_year, v)

    def download_cbp_data(self):
        """Run the CBP download step using this instance's configuration."""
        v = self.verbose
        download_cbp_data(v)

    def geocorr_files(self):
        """Run the geocorr copy step using this instance's configuration."""
        v = self.verbose
        geocorr_files(v)

    def download_school_data(self):
        """Run only the school-data download step using current configuration."""
        if "main_year" not in self.config:
            raise KeyError("main_year not found in config.")
        year = self.config["main_year"]
        v = self.verbose
        download_school_data(year, v)

    def census_metadata(self, refresh=False):
        """Return the combined ACS/Decennial name→label mapping.

        Args:
            refresh (bool): If True, recompute from the Census API even if a local
                file exists. If False (default), load from file if available.

        Returns:
            dict: Mapping from variable name to label.
        """
        mapping_path = os.path.join(OUTPUT_DIR, "census_metadata.json")
        if not refresh and os.path.exists(mapping_path):
            with open(mapping_path, "r") as f:
                return json.load(f)

        # Recompute from API
        main_year = self.config["main_year"]
        decennial_year = self.config["decennial_year"]
        acs_meta = get_census_metadata(name="acs/acs5", vintage=main_year)
        if decennial_year == 2020:
            dec_meta = get_census_metadata(name="dec/dhc", vintage=decennial_year)
        else:
            dec_meta = get_census_metadata(name="dec/sf1", vintage=decennial_year)
        metadata_required = pd.concat([acs_meta, dec_meta], ignore_index=True)
        name_label_mapping = dict(zip(metadata_required["name"], metadata_required["label"]))
        return name_label_mapping

    def pipeline(self):
        """Print a summary of each step: function, websites, output folder, and files.

        This does not hit the network; it reports the planned sources and destinations
        based on the current configuration and naming patterns in this module.
        """
        config = self.config
        main_year = config["main_year"]
        decennial_year = config["decennial_year"]
        raw_geos = config["geos"]

        # Derive state info (mirrors logic in run_all)
        main_fips = []
        for geo_value in raw_geos:
            geo_str = str(geo_value)
            if len(geo_str) < 2:
                geo_str = geo_str.zfill(2)
            main_fips.append(geo_str[:2])
        main_fips = sorted(list(set(main_fips)))
        main_abbr = [abbr for abbr in fips_info(main_fips)["abbr"] if abbr is not None]

        # Determine LODES version
        lodes_version = "LODES8" if main_year >= 2020 else "LODES7"

        steps = [
            {
                "function": "pull_census_data",
                "websites": [
                    f"https://api.census.gov/data/{main_year}/acs/acs5",
                    f"https://api.census.gov/data/{decennial_year}/dec/{'dhc' if decennial_year == 2020 else 'sf1'}",
                ],
                "output_folder": os.path.join(OUTPUT_DIR, "census", "<STATE_ABBR>"),
                "files": [
                    f"ACSDT5Y{main_year}.<ACS_TABLE>-Data.csv",
                    f"DECENNIALSF1{decennial_year}.<DEC_TABLE>-Data.csv",
                ],
            },
            {
                "function": "pull_pums_data",
                "websites": [
                    f"https://www2.census.gov/programs-surveys/acs/data/pums/{main_year}/5-Year/csv_h<state>.zip",
                    f"https://www2.census.gov/programs-surveys/acs/data/pums/{main_year}/5-Year/csv_p<state>.zip",
                ],
                "output_folder": os.path.join(OUTPUT_DIR, "pums"),
                "files": [
                    "psam_h<STATE_FIPS>.csv",
                    "psam_p<STATE_FIPS>.csv",
                ],
            },
            {
                "function": "download_shapefiles",
                "websites": [
                    f"https://www2.census.gov/geo/tiger/TIGER{main_year}/BG/tl_{main_year}_<STATE_FIPS>_bg.zip",
                ],
                "output_folder": os.path.join(OUTPUT_DIR, "geo"),
                "files": [
                    "Extracted TIGER/Line BG shapefile contents",
                ],
            },
            {
                "function": "download_ct_puma_crosswalk",
                "websites": [
                    (
                        f"https://www2.census.gov/geo/docs/maps-data/data/rel2020/2020_Census_Tract_to_2020_PUMA.txt"
                        if main_year >= 2020
                        else f"https://www2.census.gov/geo/docs/maps-data/data/rel/2010_Census_Tract_to_2010_PUMA.txt"
                    )
                ],
                "output_folder": os.path.join(OUTPUT_DIR, "geo"),
                "files": [
                    (
                        "2020_Census_Tract_to_2020_PUMA.txt (text from web page)"
                        if main_year >= 2020
                        else "2010_Census_Tract_to_2010_PUMA.txt (text from web page)"
                    )
                ],
            },
            {
                "function": "pull_LODES",
                "websites": [
                    f"https://lehd.ces.census.gov/data/lodes/{lodes_version}/<state>/od/<state>_od_main_JT01_{main_year}.csv.gz",
                    f"https://lehd.ces.census.gov/data/lodes/{lodes_version}/<state>/od/<state>_od_aux_JT01_{main_year}.csv.gz",
                    f"https://lehd.ces.census.gov/data/lodes/{lodes_version}/<state>/wac/<state>_wac_S000_JT01_{main_year}.csv.gz",
                ],
                "output_folder": os.path.join(OUTPUT_DIR, "work"),
                "files": [
                    f"<STATE>_od_main_JT01_{main_year}.csv.gz",
                    f"<STATE>_od_aux_JT01_{main_year}.csv.gz",
                    f"<STATE>_wac_S000_JT01_{main_year}.csv.gz",
                ],
            },
            {
                "function": "download_cbp_data",
                "websites": [
                    "https://www2.census.gov/programs-surveys/cbp/datasets/2016/cbp16co.zip",
                ],
                "output_folder": os.path.join(OUTPUT_DIR, "work"),
                "files": [
                    "cbp16co.zip",
                ],
            },
            {
                "function": "geocorr_files",
                "websites": [
                    "(package-local) src/geopops/geocorr/*.csv",
                ],
                "output_folder": os.path.join(OUTPUT_DIR, "geo"),
                "files": [
                    "geocorr2018_* (copied)",
                ],
            },
            {
                "function": "download_school_data",
                "websites": [
                    f"https://nces.ed.gov/programs/edge/data/EDGE_GEOCODE_PUBLICSCH_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}.zip",
                    # Enrollment datasets (examples)
                    f"https://nces.ed.gov/ccd/Data/zip/ccd_sch_029_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}_w_1a_082120.zip",
                    f"https://nces.ed.gov/ccd/Data/zip/ccd_SCH_052_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}_l_1a_082120.zip",
                    f"https://nces.ed.gov/ccd/Data/zip/ccd_sch_059_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}_l_1a_082120.zip",
                ],
                "output_folder": os.path.join(OUTPUT_DIR, "school"),
                "files": [
                    f"EDGE_GEOCODE_PUBLICSCH_{str(main_year)[-2:]}{str(main_year + 1)[-2:]}.xlsx",
                    "Shapefiles_SCH/* (extracted)",
                    "Enrollment CSVs (extracted; some SAS files deleted)",
                ],
            },
        ]

        # Print overview (always print when pipeline is called)
        print("Pipeline overview:\n")
        print(f"- OUTPUT_DIR: {OUTPUT_DIR}")
        print(f"- main_year: {main_year}, decennial_year: {decennial_year}")
        if main_abbr:
            print(f"- states: {', '.join(main_abbr)}")
        print("")

        for step in steps:
            print(f"Function: {step['function']}")
            print("Websites:")
            for site in step["websites"]:
                print(f"  - {site}")
            print(f"Output folder:\n  - {step['output_folder']}")
            print("Files:")
            for f in step["files"]:
                print(f"  - {f}")
            print("")

def main():
    """Main function to preserve CLI behavior using the class wrapper."""
    downloader = DownloadData()
    downloader.run_all()
if __name__ == "__main__":
    main()