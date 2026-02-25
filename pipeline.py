import os
import re
from dotenv import load_dotenv

load_dotenv()
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions

# For our LLM agent
try:
    from google import genai
except ImportError:
    genai = None

# 1. Initialize Vector Database connection
DB_PATH = "./chroma_hs_db"
COLLECTION_NAME = "saso_hs_codes_gemini"
EXCEL_FILE = "Tariff.xlsx"

class GeminiEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, api_key: str):
        from google import genai
        self.client = genai.Client(api_key=api_key)
        
    def __call__(self, input: list) -> list:
        import time
        embeddings = []
        batch_size = 100
        for i in range(0, len(input), batch_size):
            batch = input[i:i+batch_size]
            retries = 3
            while retries > 0:
                try:
                    res = self.client.models.embed_content(
                        model='gemini-embedding-001',
                        contents=batch
                    )
                    embeddings.extend([e.values for e in res.embeddings])
                    break
                except Exception as e:
                    print(f"Error embedding batch {i}, retries left {retries-1}: {e}")
                    time.sleep(10)
                    retries -= 1
        return embeddings

try:
    client = chromadb.PersistentClient(path=DB_PATH)
    gemini_ef = GeminiEmbeddingFunction(api_key=os.environ.get("GEMINI_API_KEY", ""))
    collection = client.get_collection(
        name=COLLECTION_NAME, 
        embedding_function=gemini_ef
    )
    print("Successfully connected to ChromaDB")
except Exception as e:
    print(f"Failed to connect to ChromaDB or get collection: {e}")
    collection = None


# =============================================================================
# 2. HS Code Hierarchy & Metadata Lookup
#    Loaded once at startup from the Excel file.
#    Provides breadcrumb paths, duty rates, and procedures for each HS code.
# =============================================================================

def _compute_dash_level(desc: str) -> int:
    """Count leading dashes in description to determine hierarchy depth."""
    match = re.match(r'^([-\s]+)', desc)
    if match:
        return match.group(1).replace(' ', '').count('-')
    return 0

def _clean_description(desc: str) -> str:
    """Strip leading dashes/spaces and trailing colons."""
    cleaned = desc.lstrip('- ').strip()
    if cleaned.endswith(':'):
        cleaned = cleaned[:-1].strip()
    return cleaned

def build_hs_lookup(excel_path: str) -> dict:
    """
    Build a lookup dictionary keyed by HS code with:
      - hierarchy_path: breadcrumb string (e.g. "Fresh vegetables > Carrots and turnips > Carrots")
      - duty_rate: the English duty rate
      - procedures: the procedures codes
    
    The hierarchy is derived from the dash-level encoding in the English descriptions.
    
    IMPORTANT: The dataset contains SASO-specific subcode entries (e.g. HS 200510120001,
    200510120002) that have NO leading dashes despite being children of deeper-level items.
    These appear as "level 0" from the dash count, but they are NOT new section headings.
    If they overwrote hierarchy[0], all subsequent items in the same section would get
    corrupted paths. We detect them by checking: if a level-0 item shares the same 4-digit
    HS prefix as the current heading, it's a subcode variant — not a new heading — so we
    preserve the existing hierarchy state and just give the subcode the parent's path.
    """
    print(f"Building HS code hierarchy lookup from {excel_path}...")
    df = pd.read_excel(excel_path, dtype=str)
    df = df.rename(columns={
        'رمز النظام المنسق \n Harmonized Code': 'HS_CODE',
        'الصنف باللغة الانجليزية \n Item English Name': 'DESC_EN',
        'الصنف باللغة العربية \n Item Arabic Name': 'DESC_AR',
        'فئة الرسم باللغة الانجليزية \n English Duty Rate': 'DUTY_RATE',
        "الاجراءات '\n Procedures": 'PROCEDURES',
    })
    
    lookup = {}
    hierarchy = {}  # level -> cleaned description text
    current_heading_prefix = None  # First 4 digits of the current level-0 heading's HS code
    
    for _, row in df.iterrows():
        hs_code = str(row.get('HS_CODE', '')).strip()
        desc_en = str(row.get('DESC_EN', ''))
        duty_rate = str(row.get('DUTY_RATE', '')) if pd.notna(row.get('DUTY_RATE')) else ''
        procedures = str(row.get('PROCEDURES', '')) if pd.notna(row.get('PROCEDURES')) else ''
        
        if not hs_code or hs_code == 'nan':
            continue
        
        level = _compute_dash_level(desc_en)
        clean = _clean_description(desc_en)
        
        hs_prefix = hs_code[:4]
        
        if level == 0:
            if current_heading_prefix is not None and hs_prefix == current_heading_prefix:
                # This is a SASO subcode variant (no dashes but same heading prefix).
                # Don't modify the hierarchy tracker — just use the current hierarchy path.
                path_parts = [hierarchy[l] for l in sorted(hierarchy.keys())]
                hierarchy_path = ' > '.join(path_parts)
                
                lookup[hs_code] = {
                    'hierarchy_path': hierarchy_path,
                    'duty_rate': duty_rate,
                    'procedures': procedures,
                }
                continue
            else:
                # Genuine new heading — update prefix and hierarchy
                current_heading_prefix = hs_prefix
        
        # Update current level in the hierarchy tracker
        hierarchy[level] = clean
        
        # Clear any deeper (stale) levels
        for k in list(hierarchy.keys()):
            if k > level:
                del hierarchy[k]
        
        # Build the breadcrumb path from level 0 up to current level
        path_parts = [hierarchy[l] for l in sorted(hierarchy.keys()) if l <= level]
        hierarchy_path = ' > '.join(path_parts)
        
        lookup[hs_code] = {
            'hierarchy_path': hierarchy_path,
            'duty_rate': duty_rate,
            'procedures': procedures,
        }
    
    print(f"Built lookup for {len(lookup)} HS codes with hierarchy paths.")
    return lookup


# Build the lookup at module load time
_hs_lookup = {}
try:
    _excel_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), EXCEL_FILE)
    if os.path.exists(_excel_path):
        _hs_lookup = build_hs_lookup(_excel_path)
    else:
        print(f"Warning: Excel file not found at {_excel_path}, hierarchy lookup disabled")
except Exception as e:
    print(f"Warning: Could not build HS lookup: {e}")


def search_hs_code(query_text, top_k=5):
    """Search ChromaDB for potential HS codes, enriched with hierarchy metadata."""
    if collection is None:
        print("ChromaDB collection not initialized. Cannot perform search.")
        return []
    
    results = collection.query(
        query_texts=[query_text],
        n_results=top_k
    )
    
    matches = []
    if results and results.get('ids') and results['ids'][0]:
        for i in range(len(results['ids'][0])):
            hs_code = results['metadatas'][0][i]['hs_code']
            
            # Enrich with hierarchy metadata from the lookup
            meta = _hs_lookup.get(hs_code, {})
            
            matches.append({
                "hs_code": hs_code,
                "desc_en": results['metadatas'][0][i]['desc_en'],
                "desc_ar": results['metadatas'][0][i].get('desc_ar', ''),
                "distance": results['distances'][0][i],
                "hierarchy_path": meta.get('hierarchy_path', ''),
                "duty_rate": meta.get('duty_rate', ''),
                "procedures": meta.get('procedures', ''),
            })
    return matches

def step_a_extract_invoice(image_path_or_url=None):
    """
    Step A: OCR with Layout Awareness
    In a real scenario, this would send an image/PDF to GPT-4o.
    For this MVP, we return mock data that simulates the output from the LLM prompt:
    'Extract the table from this invoice into JSON. Key fields: Item Description (German), Quantity, Country of Origin.'
    """
    print("--- Step A: Extracting Invoice Data (Mocked for MVP) ---")
    
    mock_extracted_json = [
        {"item_description": "Kugellager für Industriemotor", "quantity": 100, "country_of_origin": "Germany"},
        {"item_description": "Edelstahl-Kreiselpumpe für Wasser", "quantity": 5, "country_of_origin": "Germany"},
        {"item_description": "Kupferkabel 2.5mm", "quantity": 500, "country_of_origin": "Germany"}
    ]
    print(f"Extracted {len(mock_extracted_json)} items from invoice.")
    return mock_extracted_json

def step_b_translate_and_map(extracted_items, gemini_client=None):
    """
    Step B: The 'Translator + Mapper' Agent
    """
    print("\n--- Step B: Translating and Mapping to HS Codes ---")
    mapped_results = []
    
    for item in extracted_items:
        german_desc = item["item_description"]
        print(f"\nProcessing: {german_desc}")
        
        # 1. Search ChromaDB (using the direct German query!)
        potential_matches = search_hs_code(german_desc, top_k=5)
        print(f"--- ChromaDB found {len(potential_matches)} matches for '{german_desc}' ---")
        
        matches_text = ""
        for i, match in enumerate(potential_matches):
            matches_text += f"{i+1}. HS Code: {match['hs_code']} | EN: {match['desc_en'][:80]} | AR: {match['desc_ar'][:80]}\n"
            
        print(f"Found {len(potential_matches)} candidate HS Codes in Vector DB.")

        if gemini_client:
            # 2. Ask LLM to pick the best one
            prompt = f"""
            You are an expert Saudi Customs classifier.
            Here is the German item description from an invoice: '{german_desc}'.
            
            Here are 5 potential Saudi HS codes retrieved from our database:
            {matches_text}
            
            Which ONE fits best? 
            Output strictly in JSON format with exactly 4 keys: 
            "hs_code", "selected_arabic_description", "selected_english_description", and "reasoning".
            """
            
            try:
                response = gemini_client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config={
                        'response_mime_type': 'application/json',
                    }
                )
                
                decision = json.loads(response.text)
                item["hs_code"] = decision.get("hs_code", "Unknown")
                item["arabic_description"] = decision.get("selected_arabic_description", "Unknown")
                item["english_description"] = decision.get("selected_english_description", "Unknown")
                item["reasoning"] = decision.get("reasoning", "")
                print(f"LLM Selected: {item['hs_code']} - {item['arabic_description']} / {item['english_description']}")
            except Exception as e:
                print(f"LLM Mapping failed: {e}")
                item["hs_code"] = potential_matches[0]["hs_code"] if potential_matches else "Unknown"
                item["arabic_description"] = potential_matches[0]["desc_ar"] if potential_matches else "Unknown"
                item["english_description"] = potential_matches[0]["desc_en"] if potential_matches else "Unknown"
                item["reasoning"] = "Fallback to top vector match due to LLM error"
        else:
            # Fallback if no Gemini client is provided
            print("No Gemini client provided. Defaulting to Top-1 Vector DB match.")
            if potential_matches:
                best_match = potential_matches[0]
                item["hs_code"] = best_match["hs_code"]
                item["arabic_description"] = best_match["desc_ar"]
                item["reasoning"] = "Top vector search match (No LLM)"
            else:
                item["hs_code"] = "Unknown"
                item["arabic_description"] = "Unknown"
                item["reasoning"] = "No matches found"
            
        mapped_results.append(item)
        
    return mapped_results

def step_c_generate_csv(mapped_data, output_filename="processed_invoice.csv"):
    """
    Step C: Generate a CSV file.
    """
    print(f"\n--- Step C: Generating Output CSV: {output_filename} ---")
    df = pd.DataFrame(mapped_data)
    # Ensure Arabic output exports properly using utf-8-sig
    df.to_csv(output_filename, index=False, encoding='utf-8-sig') 
    print(f"Successfully saved {len(df)} records to {output_filename}")
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    print("\nPreview:")
    print(df[['item_description', 'hs_code']].head())

if __name__ == "__main__":
    # If you have a Gemini API key:
    if genai:
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
    else:
        client = None
    
    invoice_data = step_a_extract_invoice()
    mapped_data = step_b_translate_and_map(invoice_data, gemini_client=client)
    step_c_generate_csv(mapped_data)

