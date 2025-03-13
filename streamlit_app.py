import streamlit as st
import pandas as pd
import numpy as np
import yaml
import re
import io
import json
import time
import plotly.express as px
from urllib.parse import urlparse, urlunparse
from rapidfuzz import fuzz
from fuzzy_match import match
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
from functools import lru_cache
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="URL Migration Wizard | Valtech",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define improved contrast color palette based on Valtech brand colors
PRIMARY = "#FF5959"      # Coral - primary brand color
PRIMARY_DARK = "#E03131" # Darker version of Coral for hover states
SECONDARY = "#002FA7"    # Ocean - secondary accent
DARK_BG = "#171717"      # Soft Black for dark backgrounds
TEXT_DARK = "#000000"    # Black for text on light backgrounds
TEXT_LIGHT = "#FFFFFF"   # White for text on dark backgrounds
BG_LIGHT = "#FFFFFF"     # White for card backgrounds
BG_MEDIUM = "#F0F0F0"    # Light gray for page background
ACCENT_SUCCESS = "#6BB324" # Green derived from Moss for success indicators
ACCENT_WARNING = "#FF9800" # Orange for warnings
ACCENT_ERROR = "#E03131"   # Red for errors
ACCENT_INFO = "#002FA7"    # Ocean for info
BORDER_LIGHT = "#D1D3CA"   # Gray for borders

# Custom styling with improved contrast
st.markdown(f"""
<style>
    /* Main Background */
    .stApp {{
        background-color: {BG_MEDIUM};
    }}
    
    /* Headers */
    h1, h2, h3 {{
        color: {DARK_BG};
        font-weight: 700;
    }}
    
    /* Accent Elements */
    .accent-text {{
        color: {PRIMARY};
        font-weight: 600;
    }}
    
    /* Buttons */
    .stButton button {{
        background-color: {PRIMARY};
        color: {TEXT_LIGHT} !important;
        border: none;
        font-weight: 500;
        border-radius: 4px;
        transition: all 0.3s ease;
    }}
    .stButton button:hover {{
        background-color: {PRIMARY_DARK};
        color: {TEXT_LIGHT} !important;
    }}
    
    /* Custom header */
    .header-container {{
        display: flex;
        align-items: center;
        background-color: {DARK_BG};
        padding: 1.2rem 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }}
    
    /* Card containers */
    .card-container {{
        background-color: {BG_LIGHT};
        padding: 1.5rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
        border-top: 4px solid {PRIMARY};
    }}
    
    /* Tool tips */
    .tooltip {{
        position: relative;
        display: inline-block;
        cursor: pointer;
    }}
    
    .tooltip .tooltiptext {{
        visibility: hidden;
        width: 250px;
        background-color: {DARK_BG};
        color: {TEXT_LIGHT};
        text-align: left;
        border-radius: 6px;
        padding: 10px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 0;
        margin-left: 0;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.9rem;
        line-height: 1.4;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    .tooltip:hover .tooltiptext {{
        visibility: visible;
        opacity: 1;
    }}
    
    /* Status indicator */
    .status-indicator {{
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 6px;
    }}
    
    .status-success {{
        background-color: {ACCENT_SUCCESS};
    }}
    
    .status-warning {{
        background-color: {ACCENT_WARNING};
    }}
    
    .status-error {{
        background-color: {ACCENT_ERROR};
    }}
    
    /* Beta banner */
    .beta-banner {{
        background-color: {PRIMARY};
        color: {TEXT_LIGHT};
        padding: 8px;
        border-radius: 4px;
        text-align: center;
        font-size: 0.9rem;
        margin-bottom: 1.5rem;
        font-weight: 500;
    }}
    
    /* Config controls */
    .config-section {{
        background-color: {BG_LIGHT};
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        border-left: 5px solid {PRIMARY};
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }}
    
    /* Section headers */
    .section-header {{
        color: {DARK_BG};
        font-weight: 600;
        font-size: 1.2rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid {PRIMARY};
    }}
    
    /* File uploader */
    .file-uploader {{
        border: 2px dashed {BORDER_LIGHT};
        border-radius: 8px;
        padding: 2rem 1.5rem;
        text-align: center;
        background-color: {BG_LIGHT};
        transition: all 0.3s ease;
    }}
    .file-uploader:hover {{
        border-color: {PRIMARY};
    }}
    
    /* Results section */
    .results-header {{
        border-bottom: 3px solid {PRIMARY};
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
        color: {DARK_BG};
    }}
    
    /* Footer */
    .footer {{
        text-align: center;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid {BORDER_LIGHT};
        color: {DARK_BG};
        font-size: 0.9rem;
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 0;
        border-radius: 8px;
        background-color: {BG_LIGHT};
        padding: 5px;
    }}

    .stTabs [data-baseweb="tab"] {{
        height: 50px;
        background-color: transparent;
        border-radius: 4px;
        gap: 0;
        padding: 10px 15px;
        color: {DARK_BG};
        font-weight: 500;
    }}

    .stTabs [aria-selected="true"] {{
        background-color: {PRIMARY};
        color: {TEXT_LIGHT} !important;
    }}
    
    /* Metrics */
    .metric-card {{
        background-color: {BG_LIGHT};
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s ease;
        border-top: 4px solid {PRIMARY};
    }}
    
    .metric-card:hover {{
        transform: translateY(-5px);
    }}
    
    .metric-value {{
        font-size: 2.2rem;
        font-weight: bold;
        color: {PRIMARY};
        line-height: 1.2;
        margin-bottom: 0.5rem;
    }}
    
    .metric-label {{
        font-size: 1rem;
        color: {DARK_BG};
        font-weight: 500;
    }}
    
    /* Info boxes */
    .info-box {{
        background-color: rgba(0, 47, 167, 0.1);
        border-left: 4px solid {ACCENT_INFO};
        padding: 1rem;
        border-radius: 4px;
        margin: 1rem 0;
    }}
    
    /* Labels with better contrast */
    label {{
        color: {DARK_BG} !important;
        font-weight: 500 !important;
    }}
    
    /* Progress bar */
    .stProgress > div > div > div > div {{
        background-color: {PRIMARY};
    }}
</style>
""", unsafe_allow_html=True)

# Helper utility functions (same as in the provided script)
@lru_cache(maxsize=None)
def get_translation_table():
    umlaut_map = {'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss', 'æ': 'ae', 'ø': 'oe', 'å': 'aa'}
    return str.maketrans(umlaut_map)

def normalize_string(text, is_slug=False):
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('ASCII')
    text = text.translate(get_translation_table())

    if is_slug:
        text = re.sub(r'[\s_]+', '-', text)
        text = re.sub(r'[^a-z0-9-]', '', text).strip('-')
        text = re.sub(r'-+', '-', text)
    else:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

def clean_url(url, remove_query_params):
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '')
    clean_path = re.sub(r'\.(html|aspx|php|jsp|htm|cgi|pl)$', '', parsed.path)
    query = parsed.query if not remove_query_params else ''
    cleaned_url = urlunparse((parsed.scheme, domain, clean_path, '', query, ''))
    return cleaned_url

def extract_language_from_url(path, config, db_type):
    language_config = config['settings']['language_extraction'].get(db_type, {})
    if not language_config or language_config.get('regex') == False:
        return None
    language_regex = language_config.get('regex', None)
    default_language = language_config.get('default_language', None)
    if language_regex:
        match = re.match(language_regex, path)
        if match and match.group(1):
            return match.group(1)
        elif default_language:
            return default_language
    return None

def extract_product_id(text, query, config, db_type):
    text = str(text) if text is not None else ""
    product_id_config = config['settings'].get('product_id_extraction', {})
    website_regex = product_id_config.get(db_type, {}).get('regex', None)
    if not website_regex:
        return None
    min_length = product_id_config.get('min_product_id_length', 5)
    exclude_ids = product_id_config.get('exclude_product_ids', [])
    last_segment = text.strip('/').split('/')[-1]
    product_ids_from_path = re.findall(website_regex, last_segment)
    product_ids_from_query = []
    for param in query:
        if query[param] is not None:
            product_ids_from_query.extend(re.findall(website_regex, query[param]))
    potential_ids = product_ids_from_path + product_ids_from_query
    valid_product_ids = [pid for pid in potential_ids if len(pid) >= min_length and pid not in exclude_ids]
    return valid_product_ids[0] if valid_product_ids else None

def parse_url(url, remove_query_params, config, db_type):
    cleaned_url = clean_url(url, remove_query_params)
    parsed = urlparse(cleaned_url)
    query_params = {}
    for q in parsed.query.split('&'):
        if '=' in q:
            k, v = q.split('=', 1)
            query_params[k] = v
        else:
            query_params[q] = None
    product_id = extract_product_id(parsed.path, query_params, config, db_type)
    return {
        "original_url": cleaned_url,
        "domain": parsed.netloc,
        "path": process_path(parsed.path, parsed.query, remove_query_params),
        "product_id": product_id
    }

def process_path(path, query_params, remove_query_params):
    if path == '' or path == '/':
        return {
            "full": '/',
            "segments": [{"value": '/', "position": 0}],
            "levels": {"level-1": '/'},
            "slug": {
                "original": '/',
                "normalized": '/'
            }
        }
    segments = path.strip('/').split('/')
    if not remove_query_params and query_params:
        last_segment = f"{segments[-1]}?{query_params}"
    else:
        last_segment = segments[-1]
    return {
        "full": path,
        "segments": [{"value": seg, "position": i} for i, seg in enumerate(segments)],
        "levels": {f"level-{i+1}": '/' + '/'.join(segments[:i+1]) + '/' for i in range(len(segments))},
        "slug": {
            "original": last_segment if segments else '',
            "normalized": normalize_string(last_segment, is_slug=True) if segments else ''
        }
    }

def transform_metadata(row):
    def process_field(value):
        if pd.isna(value) or value == '':
            return None, ''
        return value, normalize_string(value)
    title_original, title_normalized = process_field(row.get('Title 1', ''))
    meta_description_original, meta_description_normalized = process_field(row.get('Meta Description 1', ''))
    h1_original, h1_normalized = process_field(row.get('H1-1', ''))
    return {
        "title": {
            "original": title_original,
            "normalized": title_normalized
        },
        "meta_description": {
            "original": meta_description_original,
            "normalized": meta_description_normalized
        },
        "h1": {
            "original": h1_original,
            "normalized": h1_normalized
        }
    }

def ensure_language_code(row):
    language_code = row.get('language_code')
    return language_code if language_code is not None else ''

def is_language_code_match(legacy_row, new_row, config):
    language_config = config.get('settings', {}).get('language_extraction', {})
    legacy_regex = language_config.get('legacy_website', {}).get('regex', None)
    new_regex = language_config.get('new_website', {}).get('regex', None)
    if legacy_regex is False or new_regex is False:
        return True
    legacy_lang = legacy_row.get('language_code')
    new_lang = new_row.get('language_code')
    if legacy_lang is None:
        legacy_lang = language_config.get('legacy_website', {}).get('default_language')
    if new_lang is None:
        new_lang = language_config.get('new_website', {}).get('default_language')
    if legacy_lang is None and new_lang is None:
        return True
    return legacy_lang == new_lang

def process_uploaded_file(file, remove_query_params, config, db_type):
    if file.name.endswith('.csv'):
        df = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file)
    else:
        st.error(f"Unsupported file format: {file.name}")
        return []

    # Check if 'Content Type' column exists and filter if needed
    if 'Content Type' in df.columns:
        df['Content Type'] = df['Content Type'].str.lower().str.strip()
        df = df[df['Content Type'] == 'text/html; charset=utf-8']

    # Optional: Filter out non-HTML resource URLs
    if config['settings'].get('exclude_non_html_resources', False):
        exclusion_pattern = r'(js|api|css)(\?|/|$)'
        df = df[~df['Address'].str.contains(exclusion_pattern, case=False, regex=True)]

    # Add optional column if it doesn't exist
    if 'custom_extraction_id' not in df.columns:
        df['custom_extraction_id'] = None

    def extract_custom_extraction_id(text):
        custom_extraction_regex = config['settings']['custom_extraction_id'].get(db_type, {}).get('regex', None)
        if custom_extraction_regex:
            match = re.search(custom_extraction_regex, str(text))
            if match:
                if match.lastindex:
                    return match.group(1)
                else:
                    return match.group(0)
        return None

    def transform_row(row):
        # Parse the URL and clean it, passing db_type
        url_data = parse_url(row.get('Address', ''), remove_query_params, config, db_type)
        # Extract metadata from the row
        metadata = transform_metadata(row)

        # Initialize the dictionary with default values
        transformed_row = {
            "original_address": row.get('Address', ''),
            "original_url": url_data.get('original_url', None),
            "domain": url_data.get('domain', None),
            "path": url_data.get('path', None),
            "language_code": extract_language_from_url(url_data.get('path', {}).get('full', ''), config, db_type),
            "product_id": url_data.get('product_id', None),
            "metadata": metadata,
        }

        # Only include custom_extraction_id if the column exists
        if 'custom_extraction_id' in df.columns:
            custom_extraction_id = extract_custom_extraction_id(row.get('custom_extraction_id', ''))
            transformed_row["custom_extraction_id"] = custom_extraction_id

        return transformed_row

    # Apply the transformation to each row
    transformed_data = df.apply(transform_row, axis=1).tolist()

    # Remove duplicates based on path and normalized slug
    seen = set()
    unique_data = []
    for entry in transformed_data:
        # Use the path + normalized slug as a unique key
        key = (entry['path']['full'], entry['path']['slug']['normalized'])
        if key not in seen:
            seen.add(key)
            unique_data.append(entry)
    
    return unique_data

def match_data_tfidf(legacy_data, new_data, config, min_confidence=0.80, progress_callback=None):
    matches = []
    
    # Convert to DataFrames
    legacy_df = pd.DataFrame(legacy_data)
    new_df = pd.DataFrame(new_data)
    
    if progress_callback:
        progress_callback(0.05, "Preparing data...")
    
    # Ensure all rows have 'language_code'
    legacy_df['language_code'] = legacy_df.apply(lambda row: ensure_language_code(row), axis=1)
    new_df['language_code'] = new_df.apply(lambda row: ensure_language_code(row), axis=1)
    
    # Precompute normalized fields
    legacy_df['slug_normalized'] = legacy_df.apply(lambda row: row['path']['slug']['normalized'], axis=1)
    legacy_df['product_id'] = legacy_df['product_id'].fillna('')
    legacy_df['custom_extraction_id'] = legacy_df['custom_extraction_id'].fillna('')
    
    new_df['slug_normalized'] = new_df.apply(lambda row: row['path']['slug']['normalized'], axis=1)
    new_df['product_id'] = new_df['product_id'].fillna('')
    new_df['custom_extraction_id'] = new_df['custom_extraction_id'].fillna('')
    
    # Extract title and h1 for fuzzy matching
    legacy_df['title_normalized'] = legacy_df.apply(lambda row: row['metadata']['title']['normalized'] if row.get('metadata') and row['metadata'].get('title') else '', axis=1)
    legacy_df['h1_normalized'] = legacy_df.apply(lambda row: row['metadata']['h1']['normalized'] if row.get('metadata') and row['metadata'].get('h1') else '', axis=1)
    
    new_df['title_normalized'] = new_df.apply(lambda row: row['metadata']['title']['normalized'] if row.get('metadata') and row['metadata'].get('title') else '', axis=1)
    new_df['h1_normalized'] = new_df.apply(lambda row: row['metadata']['h1']['normalized'] if row.get('metadata') and row['metadata'].get('h1') else '', axis=1)
    
    if progress_callback:
        progress_callback(0.1, "Starting product ID matching...")
    
    # 1. Product ID Matching with language validation
    for _, legacy_row in legacy_df[legacy_df['product_id'] != ''].iterrows():
        matching_new_rows = new_df[
            (new_df['product_id'] == legacy_row['product_id']) &
            (new_df['product_id'] != '')
        ]
        
        for _, new_row in matching_new_rows.iterrows():
            if is_language_code_match(legacy_row, new_row, config):
                matches.append({
                    'legacy_original_address': legacy_row['original_address'],
                    'new_original_address': new_row['original_address'],
                    'confidence_level': 1.0,
                    'matching_method': 'exact_match (product_id)',
                    'legacy_matched_value': legacy_row['product_id'],
                    'new_matched_value': new_row['product_id'],
                    'legacy_language': legacy_row['language_code'],
                    'new_language': new_row['language_code']
                })
    
    # Remove matched legacy rows
    matched_legacy_urls = [m['legacy_original_address'] for m in matches]
    unmatched_legacy_df = legacy_df[~legacy_df['original_address'].isin(matched_legacy_urls)]
    
    if progress_callback:
        progress_callback(0.25, "Product ID matching complete. Starting custom ID matching...")
    
    # 2. Custom Extraction ID Matching with language validation
    for _, legacy_row in unmatched_legacy_df[unmatched_legacy_df['custom_extraction_id'] != ''].iterrows():
        matching_new_rows = new_df[
            (new_df['custom_extraction_id'] == legacy_row['custom_extraction_id']) &
            (new_df['custom_extraction_id'] != '')
        ]
        
        for _, new_row in matching_new_rows.iterrows():
            if is_language_code_match(legacy_row, new_row, config):
                matches.append({
                    'legacy_original_address': legacy_row['original_address'],
                    'new_original_address': new_row['original_address'],
                    'confidence_level': 1.0,
                    'matching_method': 'exact_match (custom_extraction_id)',
                    'legacy_matched_value': legacy_row['custom_extraction_id'],
                    'new_matched_value': new_row['custom_extraction_id'],
                    'legacy_language': legacy_row['language_code'],
                    'new_language': new_row['language_code']
                })
    
    # Remove matched legacy rows
    matched_legacy_urls = [m['legacy_original_address'] for m in matches]
    unmatched_legacy_df = unmatched_legacy_df[~unmatched_legacy_df['original_address'].isin(matched_legacy_urls)]
    
    if progress_callback:
        progress_callback(0.4, "Custom ID matching complete. Starting slug matching...")
    
    # 3. Slug Matching with language validation
    for _, legacy_row in unmatched_legacy_df.iterrows():
        matching_new_rows = new_df[new_df['slug_normalized'] == legacy_row['slug_normalized']]
        
        for _, new_row in matching_new_rows.iterrows():
            if is_language_code_match(legacy_row, new_row, config):
                matches.append({
                    'legacy_original_address': legacy_row['original_address'],
                    'new_original_address': new_row['original_address'],
                    'confidence_level': 1.0,
                    'matching_method': 'exact_match (slug_normalized)',
                    'legacy_matched_value': legacy_row['slug_normalized'],
                    'new_matched_value': new_row['slug_normalized'],
                    'legacy_language': legacy_row['language_code'],
                    'new_language': new_row['language_code']
                })
    
    # Remove matched legacy rows
    matched_legacy_urls = [m['legacy_original_address'] for m in matches]
    unmatched_legacy_df = unmatched_legacy_df[~unmatched_legacy_df['original_address'].isin(matched_legacy_urls)]
    
    if progress_callback:
        progress_callback(0.55, "Slug matching complete. Starting fuzzy matching...")
    
    # 4. Fuzzy Matching
    fuzzy_config = config.get('settings', {}).get('fuzzy_matching', {})
    
    # If fuzzy matching is enabled
    if fuzzy_config.get('enabled', True):
        # Create vectorizers for enabled matching types
        vectorizers = {}
        vectors = {}
        
        if fuzzy_config.get('title', {}).get('enabled', True):
            vectorizers['title'] = TfidfVectorizer().fit(
                unmatched_legacy_df['title_normalized'].tolist() + new_df['title_normalized'].tolist()
            )
            vectors['legacy_title'] = vectorizers['title'].transform(unmatched_legacy_df['title_normalized'])
            vectors['new_title'] = vectorizers['title'].transform(new_df['title_normalized'])
        
        if fuzzy_config.get('h1', {}).get('enabled', True):
            vectorizers['h1'] = TfidfVectorizer().fit(
                unmatched_legacy_df['h1_normalized'].tolist() + new_df['h1_normalized'].tolist()
            )
            vectors['legacy_h1'] = vectorizers['h1'].transform(unmatched_legacy_df['h1_normalized'])
            vectors['new_h1'] = vectorizers['h1'].transform(new_df['h1_normalized'])
        
        if fuzzy_config.get('slug', {}).get('enabled', True):
            vectorizers['slug'] = TfidfVectorizer().fit(
                unmatched_legacy_df['slug_normalized'].tolist() + new_df['slug_normalized'].tolist()
            )
            vectors['legacy_slug'] = vectorizers['slug'].transform(unmatched_legacy_df['slug_normalized'])
            vectors['new_slug'] = vectorizers['slug'].transform(new_df['slug_normalized'])
        
        # Process each legacy row for fuzzy matching
        total_rows = len(unmatched_legacy_df)
        for idx, (_, legacy_row) in enumerate(unmatched_legacy_df.iterrows()):
            if progress_callback and idx % max(1, total_rows // 10) == 0:
                progress_callback(0.55 + 0.3 * (idx / total_rows), f"Fuzzy matching: {idx}/{total_rows}")
            
            best_match = None
            best_confidence = 0
            best_method = None
            best_legacy_value = None
            best_new_value = None
            best_new_row = None
            
            # Try title matching
            if fuzzy_config.get('title', {}).get('enabled', True):
                title_min_confidence = fuzzy_config.get('title', {}).get('min_confidence', min_confidence)
                if idx < len(vectors['legacy_title']):
                    cosine_sim_title = cosine_similarity(vectors['legacy_title'][idx], vectors['new_title']).flatten()
                    best_title_idx = np.argmax(cosine_sim_title)
                    if cosine_sim_title[best_title_idx] >= title_min_confidence:
                        if cosine_sim_title[best_title_idx] > best_confidence:
                            best_confidence = cosine_sim_title[best_title_idx]
                            best_method = 'fuzzy_match (title_normalized)'
                            best_new_row = new_df.iloc[best_title_idx]
                            best_legacy_value = legacy_row['title_normalized']
                            best_new_value = best_new_row['title_normalized']
            
            # Try h1 matching
            if fuzzy_config.get('h1', {}).get('enabled', True):
                h1_min_confidence = fuzzy_config.get('h1', {}).get('min_confidence', min_confidence)
                if idx < len(vectors['legacy_h1']):
                    cosine_sim_h1 = cosine_similarity(vectors['legacy_h1'][idx], vectors['new_h1']).flatten()
                    best_h1_idx = np.argmax(cosine_sim_h1)
                    if cosine_sim_h1[best_h1_idx] >= h1_min_confidence:
                        if cosine_sim_h1[best_h1_idx] > best_confidence:
                            best_confidence = cosine_sim_h1[best_h1_idx]
                            best_method = 'fuzzy_match (h1_normalized)'
                            best_new_row = new_df.iloc[best_h1_idx]
                            best_legacy_value = legacy_row['h1_normalized']
                            best_new_value = best_new_row['h1_normalized']
            
            # Try slug matching
            if fuzzy_config.get('slug', {}).get('enabled', True):
                slug_min_confidence = fuzzy_config.get('slug', {}).get('min_confidence', min_confidence)
                if idx < len(vectors['legacy_slug']):
                    cosine_sim_slug = cosine_similarity(vectors['legacy_slug'][idx], vectors['new_slug']).flatten()
                    best_slug_idx = np.argmax(cosine_sim_slug)
                    if cosine_sim_slug[best_slug_idx] >= slug_min_confidence:
                        if cosine_sim_slug[best_slug_idx] > best_confidence:
                            best_confidence = cosine_sim_slug[best_slug_idx]
                            best_method = 'fuzzy_match (slug_normalized)'
                            best_new_row = new_df.iloc[best_slug_idx]
                            best_legacy_value = legacy_row['slug_normalized']
                            best_new_value = best_new_row['slug_normalized']
            
            # If we found a match and it passes language validation
            if best_new_row is not None and is_language_code_match(legacy_row, best_new_row, config):
                matches.append({
                    'legacy_original_address': legacy_row['original_address'],
                    'new_original_address': best_new_row['original_address'],
                    'confidence_level': float(best_confidence),
                    'matching_method': best_method,
                    'legacy_matched_value': best_legacy_value,
                    'new_matched_value': best_new_value,
                    'legacy_language': legacy_row['language_code'],
                    'new_language': best_new_row['language_code']
                })
    
    if progress_callback:
        progress_callback(0.85, "Fuzzy matching complete. Adding unmatched entries...")
    
    # 5. Add unmatched entries
    matched_legacy_urls = [m['legacy_original_address'] for m in matches]
    unmatched_legacy_df = legacy_df[~legacy_df['original_address'].isin(matched_legacy_urls)]
    
    for _, row in unmatched_legacy_df.iterrows():
        matches.append({
            'legacy_original_address': row['original_address'],
            'new_original_address': None,
            'confidence_level': None,
            'matching_method': 'no_match',
            'legacy_matched_value': None,
            'new_matched_value': None,
            'legacy_language': row['language_code'],
            'new_language': None
        })
    
    if progress_callback:
        progress_callback(1.0, "Matching complete!")
    
    return matches

def calculate_match_statistics(matches):
    total_entries = len(matches)
    exact_matches = sum(1 for match in matches if match['matching_method'] and 'exact' in match['matching_method'])
    fuzzy_matches = sum(1 for match in matches if match['matching_method'] and 'fuzzy_match' in match['matching_method'])
    no_matches = sum(1 for match in matches if match['matching_method'] == 'no_match')
    
    stats = {
        "total_entries": total_entries,
        "matched_entries": total_entries - no_matches,
        "matched_percentage": ((total_entries - no_matches) / total_entries * 100) if total_entries > 0 else 0,
        "exact_matches": exact_matches,
        "exact_percentage": (exact_matches / total_entries * 100) if total_entries > 0 else 0,
        "fuzzy_matches": fuzzy_matches,
        "fuzzy_percentage": (fuzzy_matches / total_entries * 100) if total_entries > 0 else 0,
        "no_matches": no_matches,
        "no_match_percentage": (no_matches / total_entries * 100) if total_entries > 0 else 0
    }
    
    return stats

# Function to create default YAML config
def default_config():
    return {
        "settings": {
            "remove_query_params": True,
            "exclude_non_html_resources": True,
            "language_extraction": {
                "legacy_website": {
                    "regex": False,
                    "default_language": None
                },
                "new_website": {
                    "regex": False,
                    "default_language": None
                }
            },
            "product_id_extraction": {
                "min_product_id_length": 5,
                "exclude_product_ids": [2019, 2020, 2021, 2022],
                "legacy_website": {
                    "regex": ""
                },
                "new_website": {
                    "regex": ""
                }
            },
            "custom_extraction_id": {
                "legacy_website": {
                    "regex": "\\b0*(\\d+)\\b"
                },
                "new_website": {
                    "regex": "\\b0*(\\d+)\\b"
                }
            },
            "fuzzy_matching": {
                "enabled": True,
                "title": {
                    "enabled": True,
                    "min_confidence": 0.65
                },
                "h1": {
                    "enabled": True,
                    "min_confidence": 0.65
                },
                "slug": {
                    "enabled": True,
                    "min_confidence": 0.70
                }
            }
        }
    }

# Header with Logo and Title
st.markdown("""
<div class="header-container">
    <img src="https://placekitten.com/150/60" alt="Valtech Logo" class="header-logo">
    <div class="header-text">
        <h1 style="color: white; margin: 0;">URL Migration Wizard</h1>
        <p style="color: white; margin: 0; opacity: 0.8;">Seamlessly map legacy URLs to new website structure</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Beta banner
st.markdown("""
<div class="beta-banner">
    🚧 Beta Preview (v0.2) - This is a beta version. If something breaks, please contact the support team.
</div>
""", unsafe_allow_html=True)

# Create tabs for the main sections
tab1, tab2, tab3 = st.tabs(["🧙‍♂️ Setup", "🔄 Process", "📊 Results"])

with tab1:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Welcome to the URL Migration Wizard</div>', unsafe_allow_html=True)
    st.write("""
    This tool helps you efficiently match URLs between your legacy website and new website.
    Follow these steps to get started:
    
    1. Configure the matching settings below
    2. Upload your legacy and new website crawl files
    3. Process the data to generate URL mappings
    4. Review and export the results
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration section
    st.markdown('<div class="config-section">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Configuration Settings</div>', unsafe_allow_html=True)
    
    # Initialize session state for configuration
    if 'config' not in st.session_state:
        st.session_state.config = default_config()
    
    config = st.session_state.config
    
    # Basic settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tooltip">
            <strong>Remove Query Parameters</strong>
            <span class="tooltiptext">If enabled, URL query parameters will be removed during matching. Useful when query parameters don't affect page content.</span>
        </div>
        """, unsafe_allow_html=True)
        config['settings']['remove_query_params'] = st.toggle(
            "Remove Query Parameters", 
            value=config['settings']['remove_query_params'],
            key="remove_query_params"
        )
    
    with col2:
        st.markdown("""
        <div class="tooltip">
            <strong>Exclude Non-HTML Resources</strong>
            <span class="tooltiptext">When enabled, files like JavaScript, CSS, and API endpoints will be excluded from matching.</span>
        </div>
        """, unsafe_allow_html=True)
        config['settings']['exclude_non_html_resources'] = st.toggle(
            "Exclude Non-HTML Resources", 
            value=config['settings']['exclude_non_html_resources'],
            key="exclude_non_html"
        )
    
    # Language extraction settings
    st.markdown('<div class="section-header" style="font-size: 1.1rem; margin-top: 20px;">Language Settings</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tooltip">
            <strong>Legacy Website Language Extraction</strong>
            <span class="tooltiptext">Enable to extract language codes from legacy website URLs (e.g., /en/ for English). Helps ensure matching between same-language pages.</span>
        </div>
        """, unsafe_allow_html=True)
        legacy_lang_enabled = st.toggle(
            "Enable Language Extraction (Legacy)",
            value=config['settings']['language_extraction']['legacy_website']['regex'] != False,
            key="legacy_lang_enabled"
        )
        
        if legacy_lang_enabled:
            config['settings']['language_extraction']['legacy_website']['regex'] = st.text_input(
                "Legacy Website Language Regex",
                value=config['settings']['language_extraction']['legacy_website']['regex'] or "^/([a-z]{2})(/|$)",
                help="Regular expression to extract language code from URL path"
            )
            config['settings']['language_extraction']['legacy_website']['default_language'] = st.text_input(
                "Default Legacy Language",
                value=config['settings']['language_extraction']['legacy_website']['default_language'] or "",
                help="Default language code if not found in URL (e.g., 'en')"
            ) or None
        else:
            config['settings']['language_extraction']['legacy_website']['regex'] = False
    
    with col2:
        st.markdown("""
        <div class="tooltip">
            <strong>New Website Language Extraction</strong>
            <span class="tooltiptext">Enable to extract language codes from new website URLs. Matches will only be made between pages of the same language.</span>
        </div>
        """, unsafe_allow_html=True)
        new_lang_enabled = st.toggle(
            "Enable Language Extraction (New)",
            value=config['settings']['language_extraction']['new_website']['regex'] != False,
            key="new_lang_enabled"
        )
        
        if new_lang_enabled:
            config['settings']['language_extraction']['new_website']['regex'] = st.text_input(
                "New Website Language Regex",
                value=config['settings']['language_extraction']['new_website']['regex'] or "^/([a-z]{2})(/|$)",
                help="Regular expression to extract language code from URL path"
            )
            config['settings']['language_extraction']['new_website']['default_language'] = st.text_input(
                "Default New Language",
                value=config['settings']['language_extraction']['new_website']['default_language'] or "",
                help="Default language code if not found in URL (e.g., 'en')"
            ) or None
        else:
            config['settings']['language_extraction']['new_website']['regex'] = False
    
    # Product ID extraction settings
    st.markdown('<div class="section-header" style="font-size: 1.1rem; margin-top: 20px;">Product ID Settings</div>', unsafe_allow_html=True)
    
    # Base product ID settings
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tooltip">
            <strong>Minimum Product ID Length</strong>
            <span class="tooltiptext">Sets the minimum character length for a valid product ID. IDs shorter than this will be ignored.</span>
        </div>
        """, unsafe_allow_html=True)
        config['settings']['product_id_extraction']['min_product_id_length'] = st.number_input(
            "Minimum Product ID Length",
            min_value=1,
            value=config['settings']['product_id_extraction']['min_product_id_length'],
            help="Minimum length for extracted product IDs to be considered valid"
        )
    
    with col2:
        st.markdown("""
        <div class="tooltip">
            <strong>Exclude Product IDs</strong>
            <span class="tooltiptext">Enter numbers to exclude from product ID matching, separated by commas (e.g., years like 2022, 2023)</span>
        </div>
        """, unsafe_allow_html=True)
        exclude_ids_str = st.text_input(
            "Exclude Product IDs",
            value=", ".join(map(str, config['settings']['product_id_extraction']['exclude_product_ids'])),
            help="Comma-separated list of product IDs to exclude (e.g. years like 2022, 2023)"
        )
        config['settings']['product_id_extraction']['exclude_product_ids'] = [
            int(id.strip()) for id in exclude_ids_str.split(",") if id.strip().isdigit()
        ]
    
    # Website specific product ID regex
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tooltip">
            <strong>Legacy Website Product ID Regex</strong>
            <span class="tooltiptext">Regular expression to extract product IDs from legacy URLs. Example: \\d{5,} would match 5+ digit numbers.</span>
        </div>
        """, unsafe_allow_html=True)
        config['settings']['product_id_extraction']['legacy_website']['regex'] = st.text_input(
            "Legacy Website Product ID Regex",
            value=config['settings']['product_id_extraction']['legacy_website']['regex'],
            help="Leave empty to disable product ID matching for legacy website"
        )
    
    with col2:
        st.markdown("""
        <div class="tooltip">
            <strong>New Website Product ID Regex</strong>
            <span class="tooltiptext">Regular expression to extract product IDs from new URLs. Should match the same IDs as the legacy regex.</span>
        </div>
        """, unsafe_allow_html=True)
        config['settings']['product_id_extraction']['new_website']['regex'] = st.text_input(
            "New Website Product ID Regex",
            value=config['settings']['product_id_extraction']['new_website']['regex'],
            help="Leave empty to disable product ID matching for new website"
        )
    
    # Custom extraction ID settings
    st.markdown('<div class="section-header" style="font-size: 1.1rem; margin-top: 20px;">Custom ID Settings</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="tooltip">
            <strong>Legacy Website Custom ID Regex</strong>
            <span class="tooltiptext">Alternative pattern to extract custom IDs (article numbers, SKUs, etc.) from legacy URLs.</span>
        </div>
        """, unsafe_allow_html=True)
        config['settings']['custom_extraction_id']['legacy_website']['regex'] = st.text_input(
            "Legacy Website Custom ID Regex",
            value=config['settings']['custom_extraction_id']['legacy_website']['regex'],
            help="Regular expression to extract custom IDs (e.g., product codes, article IDs)"
        )
    
    with col2:
        st.markdown("""
        <div class="tooltip">
            <strong>New Website Custom ID Regex</strong>
            <span class="tooltiptext">Pattern to extract the same custom IDs from new website URLs.</span>
        </div>
        """, unsafe_allow_html=True)
        config['settings']['custom_extraction_id']['new_website']['regex'] = st.text_input(
            "New Website Custom ID Regex",
            value=config['settings']['custom_extraction_id']['new_website']['regex'],
            help="Regular expression to extract custom IDs (e.g., product codes, article IDs)"
        )
    
    # Fuzzy matching settings
    st.markdown('<div class="section-header" style="font-size: 1.1rem; margin-top: 20px;">Fuzzy Matching Settings</div>', unsafe_allow_html=True)
    
    # Enable fuzzy matching globally
    st.markdown("""
    <div class="tooltip">
        <strong>Enable Fuzzy Matching</strong>
        <span class="tooltiptext">When enabled, similar content will be matched even if URLs don't match exactly. Useful for content that has been moved or renamed.</span>
    </div>
    """, unsafe_allow_html=True)
    config['settings']['fuzzy_matching']['enabled'] = st.toggle(
        "Enable Fuzzy Matching",
        value=config['settings']['fuzzy_matching']['enabled'],
        key="fuzzy_enabled"
    )
    
    if config['settings']['fuzzy_matching']['enabled']:
        # Title fuzzy matching
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="tooltip">
                <strong>Enable Title Matching</strong>
                <span class="tooltiptext">Match pages with similar titles when exact matches can't be found.</span>
            </div>
            """, unsafe_allow_html=True)
            config['settings']['fuzzy_matching']['title']['enabled'] = st.toggle(
                "Enable Title Matching",
                value=config['settings']['fuzzy_matching']['title']['enabled'],
                key="title_enabled"
            )
        
        with col2:
            if config['settings']['fuzzy_matching']['title']['enabled']:
                st.markdown("""
                <div class="tooltip">
                    <strong>Title Match Confidence</strong>
                    <span class="tooltiptext">Minimum similarity threshold for title matches. Higher values require more similar titles.</span>
                </div>
                """, unsafe_allow_html=True)
                config['settings']['fuzzy_matching']['title']['min_confidence'] = st.slider(
                    "Title Match Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(config['settings']['fuzzy_matching']['title']['min_confidence']),
                    step=0.05,
                    help="Higher values mean more strict matching"
                )
        
        # H1 fuzzy matching
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="tooltip">
                <strong>Enable H1 Matching</strong>
                <span class="tooltiptext">Match pages with similar H1 headings when exact matches can't be found.</span>
            </div>
            """, unsafe_allow_html=True)
            config['settings']['fuzzy_matching']['h1']['enabled'] = st.toggle(
                "Enable H1 Matching",
                value=config['settings']['fuzzy_matching']['h1']['enabled'],
                key="h1_enabled"
            )
        
        with col2:
            if config['settings']['fuzzy_matching']['h1']['enabled']:
                st.markdown("""
                <div class="tooltip">
                    <strong>H1 Match Confidence</strong>
                    <span class="tooltiptext">Minimum similarity threshold for H1 heading matches. Higher values require more similar H1s.</span>
                </div>
                """, unsafe_allow_html=True)
                config['settings']['fuzzy_matching']['h1']['min_confidence'] = st.slider(
                    "H1 Match Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(config['settings']['fuzzy_matching']['h1']['min_confidence']),
                    step=0.05,
                    help="Higher values mean more strict matching"
                )
        
        # Slug fuzzy matching
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            <div class="tooltip">
                <strong>Enable Slug Matching</strong>
                <span class="tooltiptext">Match URLs with similar slugs (last part of URL path) when exact matches can't be found.</span>
            </div>
            """, unsafe_allow_html=True)
            config['settings']['fuzzy_matching']['slug']['enabled'] = st.toggle(
                "Enable Slug Matching",
                value=config['settings']['fuzzy_matching']['slug']['enabled'],
                key="slug_enabled"
            )
        
        with col2:
            if config['settings']['fuzzy_matching']['slug']['enabled']:
                st.markdown("""
                <div class="tooltip">
                    <strong>Slug Match Confidence</strong>
                    <span class="tooltiptext">Minimum similarity threshold for URL slug matches. Higher values require more similar slugs.</span>
                </div>
                """, unsafe_allow_html=True)
                config['settings']['fuzzy_matching']['slug']['min_confidence'] = st.slider(
                    "Slug Match Confidence",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(config['settings']['fuzzy_matching']['slug']['min_confidence']),
                    step=0.05,
                    help="Higher values mean more strict matching"
                )
    
    # Save configuration to session state
    st.session_state.config = config
    
    # Export/Import YAML
    st.markdown('<div class="section-header" style="font-size: 1.1rem; margin-top: 20px;">Save or Load Configuration</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        st.download_button(
            "Export Configuration",
            yaml.dump(config),
            file_name="url_migration_config.yaml",
            mime="text/yaml"
        )
    
    with col2:
        uploaded_config = st.file_uploader("Import Configuration", type="yaml")
        if uploaded_config:
            try:
                imported_config = yaml.safe_load(uploaded_config)
                st.session_state.config = imported_config
                st.success("Configuration imported successfully!")
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error importing configuration: {e}")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">Upload Files & Process</div>', unsafe_allow_html=True)
    st.write("""
    Upload your crawl files from both your legacy and new websites. The files should contain URLs and metadata 
    (title, meta description, H1) for each page. CSV or Excel formats are supported.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        st.markdown("""
        <div class="tooltip">
            <strong>Legacy Website Crawl</strong>
            <span class="tooltiptext">Upload a CSV or Excel file with URLs from your legacy website. This should contain all pages you want to redirect.</span>
        </div>
        """, unsafe_allow_html=True)
        
        legacy_file = st.file_uploader(
            "Upload Legacy Website Crawl",
            type=["csv", "xlsx", "xls"],
            key="legacy_file"
        )
        
        if legacy_file:
            st.success(f"✅ {legacy_file.name} uploaded")
            
        st.markdown('<div class="info-box">File should contain <code>Address</code>, <code>Title 1</code>, <code>Meta Description 1</code>, and <code>H1-1</code> columns</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="file-uploader">', unsafe_allow_html=True)
        st.markdown("""
        <div class="tooltip">
            <strong>New Website Crawl</strong>
            <span class="tooltiptext">Upload a CSV or Excel file with URLs from your new website. These are the destination URLs for redirects.</span>
        </div>
        """, unsafe_allow_html=True)
        
        new_file = st.file_uploader(
            "Upload New Website Crawl",
            type=["csv", "xlsx", "xls"],
            key="new_file"
        )
        
        if new_file:
            st.success(f"✅ {new_file.name} uploaded")
            
        st.markdown('<div class="info-box">File should contain the same column structure as the legacy website file</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Process button
    st.markdown('<div class="card-container">', unsafe_allow_html=True)
    process_col1, process_col2 = st.columns([3, 1])
    
    with process_col1:
        if legacy_file and new_file:
            st.markdown("""
            <div class="tooltip">
                <strong>Ready to Process!</strong>
                <span class="tooltiptext">Click the button to start processing files with the current configuration. This may take a few minutes for large files.</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("<strong>Upload both files to proceed</strong>", unsafe_allow_html=True)
    
    with process_col2:
        process_button = st.button(
            "🚀 Process Files",
            key="process_button",
            disabled=not (legacy_file and new_file),
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Process the files when the button is clicked
    if process_button and legacy_file and new_file:
        # Create containers for progress updates
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Progress callback function
        def progress_callback(progress, message):
            progress_bar.progress(progress)
            status_text.text(message)
        
        try:
            # Step 1: Process the files
            status_text.text("Processing legacy website crawl...")
            progress_bar.progress(0.1)
            
            legacy_data = process_uploaded_file(
                legacy_file,
                st.session_state.config['settings']['remove_query_params'],
                st.session_state.config,
                db_type='legacy_website'
            )
            
            progress_bar.progress(0.3)
            status_text.text("Processing new website crawl...")
            
            new_data = process_uploaded_file(
                new_file,
                st.session_state.config['settings']['remove_query_params'],
                st.session_state.config,
                db_type='new_website'
            )
            
            progress_bar.progress(0.5)
            status_text.text("Matching URLs...")
            
            # Step 2: Match the data
            matches = match_data_tfidf(
                legacy_data, 
                new_data, 
                st.session_state.config,
                progress_callback=progress_callback
            )
            
            # Step 3: Store the results in session state
            st.session_state.matches = matches
            st.session_state.match_stats = calculate_match_statistics(matches)
            st.session_state.legacy_data_length = len(legacy_data)
            st.session_state.new_data_length = len(new_data)
            
            # Complete
            progress_bar.progress(1.0)
            status_text.text("Processing complete! View results in the Results tab.")
            
            # Success message
            st.success(f"Successfully processed {len(legacy_data)} legacy URLs and {len(new_data)} new URLs.")
            
            # Auto-switch to results tab
            st.button("👉 View Results", on_click=lambda: st._config.set_option("runner.tabs", 2))
            
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
            st.exception(e)

with tab3:
    if 'matches' in st.session_state:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">URL Migration Results</div>', unsafe_allow_html=True)
        
        # Display statistics in cards
        stats = st.session_state.match_stats
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{stats["total_entries"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Total Legacy URLs</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{stats["matched_percentage"]:.1f}%</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Match Rate</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{stats["exact_matches"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Exact Matches</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{stats["fuzzy_matches"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-label">Fuzzy Matches</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualizations
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        
        st.markdown('<div class="section-header">Match Distribution</div>', unsafe_allow_html=True)
        
        # Create a pie chart for match types with better contrast colors
        match_data = {
            'Match Type': ['Exact Matches', 'Fuzzy Matches', 'No Matches'],
            'Count': [stats['exact_matches'], stats['fuzzy_matches'], stats['no_matches']]
        }
        match_df = pd.DataFrame(match_data)
        
        fig = px.pie(
            match_df, 
            values='Count', 
            names='Match Type',
            color='Match Type',
            color_discrete_map={
                'Exact Matches': ACCENT_SUCCESS,
                'Fuzzy Matches': PRIMARY,
                'No Matches': ACCENT_ERROR
            },
            hole=0.4
        )
        fig.update_layout(
            margin=dict(l=20, r=20, t=30, b=20),
            height=350
        )
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results table
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">URL Mapping Results</div>', unsafe_allow_html=True)
        
        # Convert matches to DataFrame for display
        matches_df = pd.DataFrame(st.session_state.matches)
        
        # Make confidence level display nicer
        matches_df['confidence_level'] = matches_df['confidence_level'].apply(
            lambda x: f"{x:.2f}" if pd.notnull(x) else "N/A"
        )
        
        # Filter options
        match_types = ["All"] + list(matches_df['matching_method'].dropna().unique())
        selected_match_type = st.selectbox("Filter by Match Type", match_types)
        
        if selected_match_type != "All":
            filtered_df = matches_df[matches_df['matching_method'] == selected_match_type]
        else:
            filtered_df = matches_df
        
        # Show the table with pagination
        st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400
        )
        
        # Download options
        col1, col2 = st.columns(2)
        
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="url_mapping_results.csv",
                mime="text/csv"
            )
        
        with col2:
            json_data = filtered_df.to_json(orient="records", indent=2)
            st.download_button(
                label="Download JSON",
                data=json_data,
                file_name="url_mapping_results.json",
                mime="application/json"
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
    else:
        st.markdown('<div class="card-container">', unsafe_allow_html=True)
        st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)
        st.info("No results available yet. Please upload files and process them in the Process tab.")
        st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>URL Migration Wizard v0.2 | © Valtech 2025</p>
</div>
""", unsafe_allow_html=True)
