import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import yaml
import time
import plotly.express as px
import plotly.graph_objects as go
from urllib.parse import urlparse, urlunparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# --------------------
# App Setup & Styling
# --------------------

st.set_page_config(
    page_title="URL Migration Wizard",
    page_icon="🧙‍♂️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with brand colors - high contrast for accessibility
st.markdown("""
<style>
    /* Font imports */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;700&display=swap');
    
    /* Base styling */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main containers */
    .main {
        background-color: #F3F2EF;
        color: #171717;
    }
    
    /* Sidebar styling */
    .css-1544g2n {
        background-color: #171717;
        color: #F3F2EF;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #171717;
        font-weight: 700;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: #FF5959;
        color: white;
        border: none;
        padding: 10px 15px;
        border-radius: 4px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #EA4545;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    
    /* Primary button (run analysis) */
    .primary-btn {
        background-color: #002FA7 !important;
    }
    .primary-btn:hover {
        background-color: #00217A !important;
    }
    
    /* Cards for sections */
    .css-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }
    
    /* Config panel */
    .config-panel {
        border: 1px solid #D1D3CA;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 10px;
        background-color: #FFFFFF;
    }
    
    /* Tooltips */
    .tooltip-container {
        position: relative;
        display: inline-block;
    }
    .tooltip-icon {
        color: #4C4C49;
        cursor: help;
        margin-left: 5px;
    }
    .tooltip-icon:hover + .tooltip-text,
    .tooltip-icon:active + .tooltip-text {
        visibility: visible;
        opacity: 1;
    }
    .tooltip-text {
        visibility: hidden;
        width: 250px;
        background-color: #171717;
        color: #F3F2EF;
        text-align: left;
        border-radius: 6px;
        padding: 8px 12px;
        position: absolute;
        z-index: 999;
        top: 100%;
        left: 0;
        margin-top: 5px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 0.8rem;
        line-height: 1.4;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    /* Upload box */
    .uploadbox {
        border: 2px dashed #D1D3CA;
        border-radius: 10px;
        padding: 30px 20px;
        text-align: center;
        background-color: #FFFFFF;
        transition: all 0.3s ease;
    }
    .uploadbox:hover {
        border-color: #FF5959;
    }
    
    /* Step circles */
    .step-circle {
        width: 30px;
        height: 30px;
        border-radius: 50%;
        background-color: #FF5959;
        color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        margin-right: 10px;
    }
    
    /* Metric container */
    .metric-container {
        background-color: white;
        border-radius: 8px;
        padding: 15px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        color: #FF5959;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #4C4C49;
    }
    
    /* Results table */
    .dataframe {
        width: 100%;
        border-collapse: collapse;
    }
    .dataframe th {
        background-color: #171717;
        color: white;
        padding: 10px;
        text-align: left;
    }
    .dataframe td {
        padding: 8px 10px;
        border-bottom: 1px solid #D1D3CA;
    }
    .dataframe tr:nth-child(even) {
        background-color: #F3F2EF;
    }
    
    /* Header with logo */
    .header-container {
        display: flex;
        align-items: center;
        margin-bottom: 20px;
    }
    .header-logo {
        height: 50px;
        margin-right: 15px;
    }
    .header-title {
        color: #171717;
        font-size: 1.8rem;
        margin: 0;
    }
    
    /* For tag pills */
    .tag-pill {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 5px;
    }
    .tag-exact {
        background-color: #B3FF60;
        color: #0D241E;
    }
    .tag-fuzzy {
        background-color: #002FA7;
        color: white;
    }
    .tag-none {
        background-color: #4C4C49;
        color: white;
    }
    
    /* Progress indicator */
    .css-progress {
        height: 10px;
        background-color: #D1D3CA;
        border-radius: 5px;
        overflow: hidden;
    }
    .css-progress-bar {
        height: 100%;
        background-color: #FF5959;
        width: 0%;
        transition: width 0.5s ease;
    }
    
    /* Beta banner */
    .beta-banner {
        background-color: #FF5959;
        color: white;
        padding: 5px 15px;
        border-radius: 5px;
        text-align: center;
        margin-bottom: 20px;
        font-size: 14px;
    }
    
    /* Success message */
    .success-message {
        background-color: #B3FF60;
        color: #0D241E;
        padding: 15px;
        border-radius: 5px;
        margin-top: 20px;
        font-weight: 500;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# --------------------
# Helper Functions
# --------------------

def img_to_base64(img):
    """Convert PIL Image to base64 string for embedding"""
    import base64
    from io import BytesIO
    
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
    
def create_tooltip(label, tooltip_text):
    """Creates a label with a tooltip"""
    return f"""
    <div style="display: flex; align-items: center;">
        <span>{label}</span>
        <div class="tooltip-container">
            <span class="tooltip-icon">ⓘ</span>
            <span class="tooltip-text">{tooltip_text}</span>
        </div>
    </div>
    """

@st.cache_data
def get_default_config():
    """Return the default YAML configuration"""
    default_yaml = """
    settings:
      remove_query_params: true

      exclude_non_html_resources: true

      language_extraction:
        legacy_website:
          regex: false
          default_language: None

        new_website:
          regex: false
          default_language: None

      product_id_extraction:
        enabled: false
        min_product_id_length: 5
        exclude_product_ids: [2019, 2020, 2021, 2022]
        legacy_website:
          regex: '' 
        new_website:
          regex: ''  

      custom_extraction_id:
        enabled: false
        legacy_website:
          regex: '\\b0*(\\d+)\\b'
        new_website:
          regex: '\\b0*(\\d+)\\b'

      fuzzy_matching:
        enabled: true
        title:
          enabled: true
          min_confidence: 0.65
        h1:
          enabled: true
          min_confidence: 0.65
        slug:
          enabled: true
          min_confidence: 0.70
    """
    return yaml.safe_load(default_yaml)

def clean_url(url, remove_query_params):
    """Clean URL by removing parameters and normalizing"""
    parsed = urlparse(url)
    domain = parsed.netloc.replace('www.', '')
    clean_path = re.sub(r'\.(html|aspx|php|jsp|htm|cgi|pl)$', '', parsed.path)
    query = parsed.query if not remove_query_params else ''
    cleaned_url = urlunparse((parsed.scheme, domain, clean_path, '', query, ''))
    return cleaned_url

def normalize_string(text, is_slug=False):
    """Normalize text strings for comparison"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    # Function implementation as in your script
    return text

def process_uploaded_file(file, remove_query_params, config, db_type):
    """Process the uploaded crawl file"""
    # Placeholder for the original script's function
    # In the full implementation, this would include all the URL processing logic
    df = pd.read_excel(file) if file.name.endswith(('xls', 'xlsx')) else pd.read_csv(file)
    
    # Simulate data processing
    st.session_state['processing_progress'] = 0.2
    time.sleep(0.5)
    st.session_state['processing_progress'] = 0.5
    time.sleep(0.5)
    st.session_state['processing_progress'] = 0.8
    time.sleep(0.5)
    st.session_state['processing_progress'] = 1.0
    
    # This would be replaced with actual data processing from your script
    return [{"original_address": url, "path": {"full": "/"}, "metadata": {}} for url in df['Address'].tolist()]

def match_data(legacy_data, new_data, config):
    """Match legacy URLs to new URLs"""
    # Placeholder for the matching algorithm
    # In the full implementation, this would include all the matching logic
    
    # Simulate processing
    st.session_state['matching_progress'] = 0.2
    time.sleep(0.5)
    st.session_state['matching_progress'] = 0.5
    time.sleep(0.5)
    st.session_state['matching_progress'] = 0.8
    time.sleep(0.5)
    st.session_state['matching_progress'] = 1.0
    
    # Generate sample results
    total = len(legacy_data)
    exact_match_count = int(total * 0.6)
    fuzzy_match_count = int(total * 0.3)
    no_match_count = total - exact_match_count - fuzzy_match_count
    
    matches = []
    
    # Generate exact matches
    for i in range(exact_match_count):
        if i < len(new_data):
            matches.append({
                'legacy_original_address': legacy_data[i]["original_address"],
                'new_original_address': new_data[i]["original_address"],
                'confidence_level': 1.0,
                'matching_method': 'exact_match (slug_normalized)',
                'legacy_matched_value': 'example-slug',
                'new_matched_value': 'example-slug',
                'legacy_language': 'en',
                'new_language': 'en'
            })
    
    # Generate fuzzy matches
    for i in range(exact_match_count, exact_match_count + fuzzy_match_count):
        if i < len(new_data):
            matches.append({
                'legacy_original_address': legacy_data[i]["original_address"],
                'new_original_address': new_data[i % len(new_data)]["original_address"],
                'confidence_level': round(np.random.uniform(0.70, 0.95), 2),
                'matching_method': 'fuzzy_match (title_normalized)',
                'legacy_matched_value': 'example title',
                'new_matched_value': 'similar title',
                'legacy_language': 'en',
                'new_language': 'en'
            })
    
    # Generate no matches
    for i in range(exact_match_count + fuzzy_match_count, total):
        matches.append({
            'legacy_original_address': legacy_data[i]["original_address"],
            'new_original_address': None,
            'confidence_level': None,
            'matching_method': 'no_match',
            'legacy_matched_value': None,
            'new_matched_value': None,
            'legacy_language': 'en',
            'new_language': None
        })
    
    return matches

def calculate_match_statistics(matches):
    """Calculate match statistics from results"""
    total_entries = len(matches)
    exact_matches = sum(1 for match in matches if match['matching_method'] and 'exact' in match['matching_method'])
    fuzzy_matches = sum(1 for match in matches if match['matching_method'] and 'fuzzy_match' in match['matching_method'])
    no_matches = sum(1 for match in matches if match['matching_method'] == 'no_match')
    
    return {
        'total': total_entries,
        'matched': total_entries - no_matches,
        'exact': exact_matches,
        'fuzzy': fuzzy_matches,
        'none': no_matches,
        'match_rate': (total_entries - no_matches) / total_entries if total_entries > 0 else 0
    }

def create_results_visualizations(stats):
    """Create visualizations for match results"""
    # Match types pie chart
    fig_pie = px.pie(
        names=['Exact Matches', 'Fuzzy Matches', 'No Matches'],
        values=[stats['exact'], stats['fuzzy'], stats['none']],
        color_discrete_sequence=['#B3FF60', '#002FA7', '#4C4C49'],
        hole=0.4
    )
    fig_pie.update_layout(
        margin=dict(t=0, b=0, l=0, r=0),
        legend=dict(orientation="h", yanchor="bottom", y=-0.2),
        height=300
    )
    
    # Match confidence histogram (only for matched URLs)
    if 'matches_df' in st.session_state and not st.session_state.matches_df.empty:
        # Filter out no_match entries
        matched_df = st.session_state.matches_df[st.session_state.matches_df['matching_method'] != 'no_match']
        
        if not matched_df.empty:
            fig_hist = px.histogram(
                matched_df,
                x='confidence_level',
                color='matching_method',
                color_discrete_map={
                    'exact_match (slug_normalized)': '#B3FF60',
                    'exact_match (product_id)': '#B3FF60',
                    'exact_match (custom_extraction_id)': '#B3FF60',
                    'fuzzy_match (title_normalized)': '#002FA7',
                    'fuzzy_match (h1_normalized)': '#002FA7',
                    'fuzzy_match (slug_normalized)': '#002FA7'
                },
                nbins=20,
                opacity=0.8
            )
            fig_hist.update_layout(
                margin=dict(t=30, b=0, l=0, r=0),
                title="Match Confidence Distribution",
                xaxis_title="Confidence Level",
                yaxis_title="Count",
                legend_title="Match Type",
                height=300
            )
            return fig_pie, fig_hist
    
    return fig_pie, None

# --------------------
# Session State Setup
# --------------------

# Initialize session state variables
if 'config' not in st.session_state:
    st.session_state.config = get_default_config()

if 'processing_progress' not in st.session_state:
    st.session_state.processing_progress = 0

if 'matching_progress' not in st.session_state:
    st.session_state.matching_progress = 0

if 'results_ready' not in st.session_state:
    st.session_state.results_ready = False

if 'legacy_data' not in st.session_state:
    st.session_state.legacy_data = None

if 'new_data' not in st.session_state:
    st.session_state.new_data = None

if 'matches' not in st.session_state:
    st.session_state.matches = None

if 'matches_df' not in st.session_state:
    st.session_state.matches_df = None

if 'statistics' not in st.session_state:
    st.session_state.statistics = None

# --------------------
# App Header
# --------------------

# Beta banner
st.markdown(
    """
    <div class="beta-banner">
        🧙‍♂️ Beta Preview (v0.1) - URL Migration Wizard is in beta testing. If you find issues, please report them.
    </div>
    """, 
    unsafe_allow_html=True
)

# Header with logo
try:
    logo = Image.open("images/logo.png")
    st.markdown(
        f"""
        <div class="header-container">
            <img src="data:image/png;base64,{img_to_base64(logo)}" class="header-logo" alt="Company Logo">
            <h1 class="header-title">URL Migration Wizard</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
except:
    # Fallback if logo is not found
    st.title("🧙‍♂️ URL Migration Wizard")

st.markdown(
    """
    Transform your website migration with intelligent URL matching. Upload your website crawls, 
    configure matching parameters, and get comprehensive mapping between legacy and new URLs.
    """
)

# --------------------
# Main App Tabs
# --------------------

tabs = st.tabs(["📤 Upload & Configure", "🔍 Analysis Results", "📊 Data Visualization", "❓ Help"])

with tabs[0]:
    st.header("Upload Files & Configure Settings")
    
    # File upload section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown(
            create_tooltip(
                "Legacy Website Crawl", 
                "Upload a CSV or Excel file containing the crawl data from your legacy website."
            ), 
            unsafe_allow_html=True
        )
        legacy_file = st.file_uploader(
            "Choose legacy website crawl file", 
            type=["csv", "xlsx"], 
            key="legacy_upload",
            help="CSV or Excel file with 'Address', 'Title 1', 'Meta Description 1', and 'H1-1' columns"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="css-card">', unsafe_allow_html=True)
        st.markdown(
            create_tooltip(
                "New Website Crawl", 
                "Upload a CSV or Excel file containing the crawl data from your new website."
            ),
            unsafe_allow_html=True
        )
        new_file = st.file_uploader(
            "Choose new website crawl file", 
            type=["csv", "xlsx"], 
            key="new_upload",
            help="CSV or Excel file with 'Address', 'Title 1', 'Meta Description 1', and 'H1-1' columns"
        )
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration section
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("Configuration")
    
    with st.expander("URL Processing Settings", expanded=True):
        config = st.session_state.config
        
        col1, col2 = st.columns(2)
        
        with col1:
            config['settings']['remove_query_params'] = st.toggle(
                "Remove Query Parameters",
                value=config['settings']['remove_query_params'],
                help="When enabled, query parameters (like ?id=123) will be stripped from URLs during processing."
            )
            
            config['settings']['exclude_non_html_resources'] = st.toggle(
                "Exclude Non-HTML Resources",
                value=config['settings']['exclude_non_html_resources'],
                help="When enabled, URLs that appear to be JavaScript, CSS, or API endpoints will be excluded."
            )
    
    # Custom Extraction ID Settings - Now in its own expander
    with st.expander("Custom Extraction ID Settings"):
        st.markdown(
            create_tooltip(
                "Custom Extraction ID", 
                "Define regex patterns to extract custom identifier values from URLs."
            ), 
            unsafe_allow_html=True
        )
        
        # Toggle to enable/disable Custom Extraction ID
        custom_extraction_enabled = st.toggle(
            "Enable Custom Extraction ID",
            value=config['settings']['custom_extraction_id']['enabled'],
            help="When enabled, the system will extract and match URLs based on custom identifiers defined by regex patterns."
        )
        config['settings']['custom_extraction_id']['enabled'] = custom_extraction_enabled
        
        # Only show regex input fields if enabled
        if custom_extraction_enabled:
            col1, col2 = st.columns(2)
            
            with col1:
                # Legacy website regex
                config['settings']['custom_extraction_id']['legacy_website']['regex'] = st.text_input(
                    "Legacy Website Custom ID Regex",
                    value=config['settings']['custom_extraction_id']['legacy_website']['regex'],
                    help="Regular expression pattern to extract identifiers from legacy URLs"
                )
            
            with col2:
                # New website regex
                config['settings']['custom_extraction_id']['new_website']['regex'] = st.text_input(
                    "New Website Custom ID Regex",
                    value=config['settings']['custom_extraction_id']['new_website']['regex'],
                    help="Regular expression pattern to extract identifiers from new URLs"
                )
        else:
            # If disabled, set regex fields to empty
            config['settings']['custom_extraction_id']['legacy_website']['regex'] = ''
            config['settings']['custom_extraction_id']['new_website']['regex'] = ''
    
    with st.expander("Language Settings"):
        st.markdown(
            create_tooltip(
                "Language Extraction",
                "Configure how language codes are extracted from URLs. If your site has language paths " +
                "like '/en/' or '/fr/', you can define regex patterns to extract them."
            ),
            unsafe_allow_html=True
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Legacy website language settings
            st.markdown("#### Legacy Website")
            
            legacy_regex_enabled = not (config['settings']['language_extraction']['legacy_website']['regex'] is False)
            use_legacy_regex = st.checkbox(
                "Extract Language from Legacy URLs",
                value=legacy_regex_enabled,
                key="legacy_lang_toggle"
            )
            
            config['settings']['language_extraction']['legacy_website']['regex'] = (
                st.text_input(
                    "Legacy Language Regex",
                    value="" if not legacy_regex_enabled else config['settings']['language_extraction']['legacy_website']['regex'],
                    disabled=not use_legacy_regex,
                    help="Example regex to match language code in path: '^/([a-z]{2})/' will match '/en/' and capture 'en'"
                ) if use_legacy_regex else False
            )
            
            config['settings']['language_extraction']['legacy_website']['default_language'] = st.text_input(
                "Default Legacy Language",
                value=config['settings']['language_extraction']['legacy_website']['default_language'] 
                    if config['settings']['language_extraction']['legacy_website']['default_language'] != "None" else "",
                help="Default language code if no language is detected (leave empty for None)"
            )
            
            if config['settings']['language_extraction']['legacy_website']['default_language'] == "":
                config['settings']['language_extraction']['legacy_website']['default_language'] = "None"
        
        with col2:
            # New website language settings
            st.markdown("#### New Website")
            
            new_regex_enabled = not (config['settings']['language_extraction']['new_website']['regex'] is False)
            use_new_regex = st.checkbox(
                "Extract Language from New URLs",
                value=new_regex_enabled,
                key="new_lang_toggle"
            )
            
            config['settings']['language_extraction']['new_website']['regex'] = (
                st.text_input(
                    "New Language Regex",
                    value="" if not new_regex_enabled else config['settings']['language_extraction']['new_website']['regex'],
                    disabled=not use_new_regex,
                    help="Example regex to match language code in path: '^/([a-z]{2})/' will match '/en/' and capture 'en'"
                ) if use_new_regex else False
            )
            
            config['settings']['language_extraction']['new_website']['default_language'] = st.text_input(
                "Default New Language",
                value=config['settings']['language_extraction']['new_website']['default_language']
                    if config['settings']['language_extraction']['new_website']['default_language'] != "None" else "",
                help="Default language code if no language is detected (leave empty for None)"
            )
            
            if config['settings']['language_extraction']['new_website']['default_language'] == "":
                config['settings']['language_extraction']['new_website']['default_language'] = "None"
    
    with st.expander("Product ID Settings"):
        st.markdown(
            create_tooltip(
                "Product ID Extraction",
                "Configure how product IDs are extracted from URLs. This is useful for e-commerce sites " +
                "where product IDs can be used to match pages directly."
            ),
            unsafe_allow_html=True
        )
        
        # Toggle to enable/disable Product ID extraction
        product_id_enabled = st.toggle(
            "Enable Product ID Extraction",
            value=config['settings']['product_id_extraction']['enabled'],
            help="When enabled, the system will extract and match URLs based on product IDs."
        )
        config['settings']['product_id_extraction']['enabled'] = product_id_enabled
        
        if product_id_enabled:
            # Minimum product ID length
            config['settings']['product_id_extraction']['min_product_id_length'] = st.number_input(
                "Minimum Product ID Length",
                min_value=1,
                max_value=20,
                value=config['settings']['product_id_extraction']['min_product_id_length'],
                help="Product IDs shorter than this will be ignored"
            )
            
            # Excluded product IDs
            excluded_ids = st.text_input(
                "Excluded Product IDs (comma separated)",
                value=",".join(map(str, config['settings']['product_id_extraction']['exclude_product_ids'])),
                help="These IDs will be ignored even if they match the pattern (e.g., common years like 2022)"
            )
            config['settings']['product_id_extraction']['exclude_product_ids'] = [id.strip() for id in excluded_ids.split(",")]
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Legacy website product ID regex
                config['settings']['product_id_extraction']['legacy_website']['regex'] = st.text_input(
                    "Legacy Website Product ID Regex",
                    value=config['settings']['product_id_extraction']['legacy_website']['regex'],
                    help="Regular expression pattern to extract product IDs from legacy URLs"
                )
            
            with col2:
                # New website product ID regex
                config['settings']['product_id_extraction']['new_website']['regex'] = st.text_input(
                    "New Website Product ID Regex",
                    value=config['settings']['product_id_extraction']['new_website']['regex'],
                    help="Regular expression pattern to extract product IDs from new URLs"
                )
        else:
            # If disabled, set regex fields to empty
            config['settings']['product_id_extraction']['legacy_website']['regex'] = ''
            config['settings']['product_id_extraction']['new_website']['regex'] = ''
    
    with st.expander("Fuzzy Matching Settings"):
        st.markdown(
            create_tooltip(
                "Fuzzy Matching",
                "Configure how URLs are matched when exact matches aren't found. Fuzzy matching compares titles, " +
                "H1 headings, and URL slugs to find probable matches."
            ),
            unsafe_allow_html=True
        )
        
        # Enable/disable fuzzy matching
        config['settings']['fuzzy_matching']['enabled'] = st.toggle(
            "Enable Fuzzy Matching",
            value=config['settings']['fuzzy_matching']['enabled'],
            help="When enabled, the system will attempt to match URLs based on similar content when exact matches aren't found."
        )
        
        # If fuzzy matching is enabled, show confidence sliders
        if config['settings']['fuzzy_matching']['enabled']:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Title matching
                st.markdown("#### Title Matching")
                config['settings']['fuzzy_matching']['title']['enabled'] = st.toggle(
                    "Enable Title Matching",
                    value=config['settings']['fuzzy_matching']['title']['enabled'],
                    help="Match pages based on similar page titles"
                )
                
                if config['settings']['fuzzy_matching']['title']['enabled']:
                    config['settings']['fuzzy_matching']['title']['min_confidence'] = st.slider(
                        "Title Match Confidence",
                        min_value=0.5,
                        max_value=1.0,
                        value=float(config['settings']['fuzzy_matching']['title']['min_confidence']),
                        step=0.05,
                        format="%.2f",
                        help="Minimum confidence score required for title-based matches (higher = stricter)"
                    )
            
            with col2:
                # H1 matching
                st.markdown("#### H1 Matching")
                config['settings']['fuzzy_matching']['h1']['enabled'] = st.toggle(
                    "Enable H1 Matching",
                    value=config['settings']['fuzzy_matching']['h1']['enabled'],
                    help="Match pages based on similar H1 headings"
                )
                
                if config['settings']['fuzzy_matching']['h1']['enabled']:
                    config['settings']['fuzzy_matching']['h1']['min_confidence'] = st.slider(
                        "H1 Match Confidence",
                        min_value=0.5,
                        max_value=1.0,
                        value=float(config['settings']['fuzzy_matching']['h1']['min_confidence']),
                        step=0.05,
                        format="%.2f",
                        help="Minimum confidence score required for H1-based matches (higher = stricter)"
                    )
            
            with col3:
                # Slug matching
                st.markdown("#### Slug Matching")
                config['settings']['fuzzy_matching']['slug']['enabled'] = st.toggle(
                    "Enable Slug Matching",
                    value=config['settings']['fuzzy_matching']['slug']['enabled'],
                    help="Match pages based on similar URL slugs"
                )
                
                if config['settings']['fuzzy_matching']['slug']['enabled']:
                    config['settings']['fuzzy_matching']['slug']['min_confidence'] = st.slider(
                        "Slug Match Confidence",
                        min_value=0.5,
                        max_value=1.0,
                        value=float(config['settings']['fuzzy_matching']['slug']['min_confidence']),
                        step=0.05,
                        format="%.2f",
                        help="Minimum confidence score required for slug-based matches (higher = stricter)"
                    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Run Analysis button
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("Run URL Migration Analysis")
    
    if legacy_file is not None and new_file is not None:
        analyze_button = st.button(
            "Run Analysis", 
            type="primary",
            use_container_width=True,
            key="run_analysis",
            help="Process files and perform URL matching analysis"
        )
        
        if analyze_button:
            with st.spinner("Processing files... This may take several minutes."):
                # Process legacy file
                st.text("Processing legacy website crawl...")
                progress_bar = st.progress(0)
                
                # Process the legacy website file
                legacy_data = process_uploaded_file(
                    legacy_file, 
                    config['settings']['remove_query_params'],
                    config,
                    'legacy_website'
                )
                st.session_state.legacy_data = legacy_data
                progress_bar.progress(50)
                
                # Process the new website file
                st.text("Processing new website crawl...")
                new_data = process_uploaded_file(
                    new_file, 
                    config['settings']['remove_query_params'],
                    config,
                    'new_website'
                )
                st.session_state.new_data = new_data
                progress_bar.progress(100)
                
                # Match data
                st.text("Matching URLs...")
                matches = match_data(legacy_data, new_data, config)
                st.session_state.matches = matches
                
                # Convert matches to DataFrame
                st.session_state.matches_df = pd.DataFrame(matches)
                
                # Calculate statistics
                st.session_state.statistics = calculate_match_statistics(matches)
                
                # Set results flag
                st.session_state.results_ready = True
                
                # Display success message
                match_rate = st.session_state.statistics["match_rate"] * 100
                st.markdown(
                    f"""
                    <div class="success-message">
                        Analysis completed successfully! Found matches for {match_rate:.1f}% of your URLs. 
                        Please navigate to the "Analysis Results" tab to view detailed results and download your URL mapping.
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Don't rerun automatically - let user see the success message and navigate manually
                # st.rerun()  # Removed this line
    else:
        st.info("Please upload both legacy and new website crawl files to run the analysis.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tabs[1]:
    st.header("Analysis Results")
    
    if st.session_state.results_ready:
        # Display summary statistics
        st.subheader("Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{st.session_state.statistics["total"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Total URLs</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{st.session_state.statistics["exact"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Exact Matches</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{st.session_state.statistics["fuzzy"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Fuzzy Matches</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col4:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{st.session_state.statistics["none"]}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">No Matches</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Match rate
        st.markdown('<div class="metric-container" style="margin-top: 15px;">', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value" style="color: #002FA7;">{st.session_state.statistics["match_rate"]*100:.1f}%</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Overall Match Rate</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Results table
        st.subheader("URL Mappings")
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            match_filter = st.multiselect(
                "Filter by Match Type",
                options=["Exact Match", "Fuzzy Match", "No Match"],
                default=["Exact Match", "Fuzzy Match", "No Match"],
                help="Select which types of matches to display"
            )
        
        with col2:
            confidence_range = st.slider(
                "Confidence Level Range",
                min_value=0.0,
                max_value=1.0,
                value=(0.0, 1.0),
                step=0.05,
                help="Filter matches by confidence level"
            )
        
        with col3:
            search_term = st.text_input(
                "Search URLs",
                placeholder="Enter keywords to search",
                help="Search in legacy or new URLs"
            )
        
        # Apply filters to the DataFrame
        filtered_df = st.session_state.matches_df.copy()
        
        # Match type filter
        if "Exact Match" in match_filter and "Fuzzy Match" in match_filter and "No Match" in match_filter:
            pass  # Show all
        else:
            if "Exact Match" in match_filter:
                exact_mask = filtered_df['matching_method'].str.contains('exact_match', na=False)
            else:
                exact_mask = pd.Series([False] * len(filtered_df))
                
            if "Fuzzy Match" in match_filter:
                fuzzy_mask = filtered_df['matching_method'].str.contains('fuzzy_match', na=False)
            else:
                fuzzy_mask = pd.Series([False] * len(filtered_df))
                
            if "No Match" in match_filter:
                no_match_mask = filtered_df['matching_method'] == 'no_match'
            else:
                no_match_mask = pd.Series([False] * len(filtered_df))
                
            filtered_df = filtered_df[exact_mask | fuzzy_mask | no_match_mask]
        
        # Confidence filter (only apply to rows with confidence values)
        has_confidence = filtered_df['confidence_level'].notna()
        if has_confidence.any():
            confidence_mask = (filtered_df['confidence_level'] >= confidence_range[0]) & (filtered_df['confidence_level'] <= confidence_range[1])
            filtered_df = filtered_df[confidence_mask | ~has_confidence]
        
        # Search filter
        if search_term:
            search_mask = (
                filtered_df['legacy_original_address'].str.contains(search_term, case=False, na=False) |
                filtered_df['new_original_address'].str.contains(search_term, case=False, na=False)
            )
            filtered_df = filtered_df[search_mask]
        
        # Display the filtered DataFrame
        if not filtered_df.empty:
            # Format the DataFrame for display
            display_df = filtered_df.copy()
            
            # Add visual indicators for match types
            def format_method(method):
                if pd.isna(method) or method == 'no_match':
                    return "❌ No Match"
                elif 'exact_match' in method:
                    return "✅ " + method.replace('exact_match', 'Exact Match')
                elif 'fuzzy_match' in method:
                    return "🔄 " + method.replace('fuzzy_match', 'Fuzzy Match')
                return method
            
            display_df['matching_method'] = display_df['matching_method'].apply(format_method)
            
            # Format confidence level
            display_df['confidence_level'] = display_df['confidence_level'].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
            )
            
            # Rename columns for display
            display_df = display_df.rename(columns={
                'legacy_original_address': 'Legacy URL',
                'new_original_address': 'New URL',
                'confidence_level': 'Confidence',
                'matching_method': 'Match Type',
                'legacy_language': 'Legacy Lang',
                'new_language': 'New Lang'
            })
            
            # Select columns for display
            display_df = display_df[['Legacy URL', 'New URL', 'Match Type', 'Confidence', 'Legacy Lang', 'New Lang']]
            
            # Show the table
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
            
            # Download buttons
            col1, col2 = st.columns(2)
            
            with col1:
                csv = filtered_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "Download Results as CSV",
                    data=csv,
                    file_name="url_migration_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            with col2:
                json_str = filtered_df.to_json(orient="records", indent=2)
                st.download_button(
                    "Download Results as JSON",
                    data=json_str,
                    file_name="url_migration_results.json",
                    mime="application/json",
                    use_container_width=True
                )
        else:
            st.warning("No results match your filters. Try adjusting your filter criteria.")
    else:
        st.info("No analysis results available yet. Please upload files and run the analysis first.")

with tabs[2]:
    st.header("Data Visualization")
    
    if st.session_state.results_ready:
        # Create visualizations
        fig_pie, fig_hist = create_results_visualizations(st.session_state.statistics)
        
        # Display pie chart
        st.subheader("Match Type Distribution")
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Display histogram if available
        if fig_hist:
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # Match method breakdown
        if 'matches_df' in st.session_state and not st.session_state.matches_df.empty:
            st.subheader("Match Method Breakdown")
            
            # Group by matching method and count
            method_counts = st.session_state.matches_df['matching_method'].value_counts().reset_index()
            method_counts.columns = ['Method', 'Count']
            
            # Create horizontal bar chart
            fig_bar = px.bar(
                method_counts,
                y='Method',
                x='Count',
                orientation='h',
                color='Method',
                color_discrete_map={
                    'exact_match (slug_normalized)': '#B3FF60',
                    'exact_match (product_id)': '#B3FF60',
                    'exact_match (custom_extraction_id)': '#B3FF60',
                    'fuzzy_match (title_normalized)': '#002FA7',
                    'fuzzy_match (h1_normalized)': '#002FA7',
                    'fuzzy_match (slug_normalized)': '#002FA7',
                    'no_match': '#4C4C49'
                }
            )
            fig_bar.update_layout(
                xaxis_title="Number of URLs",
                yaxis_title="",
                height=400,
                margin=dict(l=0, r=0, t=20, b=0)
            )
            
            st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No visualization data available yet. Please upload files and run the analysis first.")

with tabs[3]:
    st.header("Help & Documentation")
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("About URL Migration Wizard")
    st.markdown("""
    URL Migration Wizard helps you map URLs between an old (legacy) website and a new website. 
    This tool is designed to help with website migrations by automating the process of creating 
    URL redirects.
    
    The wizard works by:
    1. Analyzing crawl data from both the legacy and new websites
    2. Matching legacy URLs to their new counterparts using multiple strategies
    3. Providing confidence scores for each match
    4. Generating a comprehensive mapping that can be used for redirects
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("Getting Started")
    st.markdown("""
    #### Step 1: Prepare Your Crawl Data
    
    You need to crawl both your legacy website and new website to generate the required input files.
    We recommend using tools like Screaming Frog, Sitebulb, or other web crawlers.
    
    Your crawl files should be in CSV or Excel format and must include the following columns:
    - `Address` (the URL)
    - `Title 1` (the page title)
    - `Meta Description 1` (the meta description)
    - `H1-1` (the first H1 heading)
    
    #### Step 2: Configure Matching Settings
    
    Adjust the configuration settings to match your specific needs:
    - URL Processing: Control how URLs are processed and cleaned
    - Custom Extraction ID: Configure regex patterns to extract custom IDs from URLs
    - Language Settings: Configure language code extraction if your site is multilingual
    - Product ID Settings: Set up product ID extraction for e-commerce sites
    - Fuzzy Matching: Configure how URLs are matched based on content similarity
    
    #### Step 3: Run the Analysis
    
    Click "Run Analysis" to process your files and generate matches.
    
    #### Step 4: Review and Export Results
    
    Review the match results, filter as needed, and export to CSV or JSON for implementation.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("Matching Methods Explained")
    st.markdown("""
    URL Migration Wizard uses several methods to match legacy URLs to new URLs, in order of precision:
    
    #### Exact Matching
    
    1. **Product ID Matching**: If product IDs can be extracted from both legacy and new URLs and they match, this is considered a high-confidence match.
    
    2. **Custom Extraction ID Matching**: Custom IDs defined by your regex patterns are matched exactly.
    
    3. **Normalized Slug Matching**: URL slugs (the last part of the path) are normalized and compared exactly.
    
    #### Fuzzy Matching
    
    When exact matches aren't found, the tool uses fuzzy matching to find the most similar content:
    
    4. **Title Matching**: Page titles are compared for similarity using TF-IDF and cosine similarity.
    
    5. **H1 Matching**: The main headings of pages are compared for similarity.
    
    6. **Slug Fuzzy Matching**: URL slugs are compared for similarity when they don't match exactly.
    
    Each fuzzy match is given a confidence score between 0 and 1, where higher values indicate greater confidence.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="css-card">', unsafe_allow_html=True)
    st.subheader("FAQs")
    
    with st.expander("What file formats are supported?"):
        st.markdown("""
        The URL Migration Wizard supports:
        - CSV files (.csv)
        - Excel files (.xlsx, .xls)
        
        Your files must contain at minimum an 'Address' column with the URLs.
        """)
    
    with st.expander("How do I improve match accuracy?"):
        st.markdown("""
        To improve match accuracy:
        
        1. Ensure your crawl data is comprehensive and includes titles, H1s, and meta descriptions
        2. Adjust fuzzy matching confidence thresholds based on your needs
        3. If your site has product IDs or other unique identifiers, configure the regex patterns to extract them
        4. For multilingual sites, configure language extraction to ensure proper language matching
        """)
    
    with st.expander("What do the confidence scores mean?"):
        st.markdown("""
        Confidence scores range from 0 to 1:
        - 1.0: Exact match (100% confidence)
        - 0.7-0.99: High similarity, very likely to be the correct match
        - 0.5-0.69: Moderate similarity, may be correct but should be verified
        - Below 0.5: Low similarity, unlikely to be shown as a match by default
        
        Exact matches always receive a confidence score of 1.0.
        """)
    
    with st.expander("How do I implement the redirects after export?"):
        st.markdown("""
        After exporting your URL mapping:
        
        1. **For Apache**: Create 301 redirects in your .htaccess file
        2. **For Nginx**: Add redirect rules to your server configuration
        3. **For CMS platforms**: Many CMS systems have redirect management tools or plugins
        4. **For CDNs**: Services like Cloudflare, Fastly, etc. allow you to implement redirect rules
        
        Always test your redirects thoroughly before deploying to production.
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)
