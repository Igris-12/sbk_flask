import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
import gc
from urllib.parse import urlparse
import random
import yake
from datetime import datetime
import argparse
from dotenv import load_dotenv

# --- COMMAND LINE ARGUMENTS ---
parser = argparse.ArgumentParser(description='Process articles with embeddings (GPU-optimized)')
parser.add_argument('--table', type=str, default='articles.csv', help='Input the table')
parser.add_argument('--file', type=str, default='SB_publication_PMC.csv', help='Input CSV file path')
parser.add_argument('--output-suffix', type=str, default='', help='Suffix for output log files')

args = parser.parse_args()

if args.output_suffix:
    FAILED_URLS_LOG = f'failed_urls_{args.output_suffix}.log'

# --- CONFIGURATION ---

load_dotenv() # Loads variables from .env file

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("✗ Error: Supabase credentials not found in .env file.")
    exit()

CSV_FILE_PATH = args.file
TABLE_NAME = args.table
ARTICLE_BATCH_SIZE = 12  # Balanced batch size
CHUNK_SIZE = 1200  # Good balance of context and speed
CHUNK_OVERLAP = 120  # 10% overlap for better continuity
EMBEDDING_BATCH_SIZE = 196  # Optimized for 4GB VRAM target
FAILED_URLS_LOG = 'failed_urls.log'

# YAKE keyword extraction parameters
YAKE_LANGUAGE = "en"
YAKE_MAX_NGRAM = 3  # Extract up to 3-word phrases
YAKE_NUM_KEYWORDS = 10  # Number of keywords to extract

# Headers to mimic a real browser and avoid 403 errors
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}


# --- INITIALIZATION ---
print("Initializing...")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

if device == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Available VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("Supabase client initialized.")
except Exception as e:
    print(f"Error initializing Supabase client: {e}")
    exit()

# Balanced model for accuracy and speed
# 'all-MiniLM-L6-v2' (384 dims, fast), 'paraphrase-MiniLM-L6-v2' (384 dims, better quality)
# 'all-MiniLM-L12-v2' (384 dims, most accurate), 'all-mpnet-base-v2' (768 dims, best quality)
try:
    # Use L12 model - better accuracy than L6, still efficient
    model = SentenceTransformer('all-MiniLM-L12-v2', device=device)
    model.max_seq_length = 384  # Balanced context window
    print(f"Model loaded on {device}.")
    
    if device == 'cuda':
        # Warm up GPU
        dummy = model.encode(["warmup"] * 10, show_progress_bar=False)
        torch.cuda.empty_cache()
        print(f"Initial VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

# Create a requests session for better connection handling
session = requests.Session()
session.headers.update(HEADERS)

# Initialize YAKE keyword extractor
kw_extractor = yake.KeywordExtractor(
    lan=YAKE_LANGUAGE,
    n=YAKE_MAX_NGRAM,
    dedupLim=0.9,
    top=YAKE_NUM_KEYWORDS,
    features=None
)

# --- HELPER FUNCTIONS ---
def log_failure(url, reason):
    """Appends a failed URL and reason to a log file."""
    with open(FAILED_URLS_LOG, "a", encoding='utf-8') as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {url}\tREASON: {reason}\n")
def fetch_article(url, max_retries=3):
    """Fetch article with retry logic and rate limiting"""
    for attempt in range(max_retries):
        try:
            # Add random delay to appear more human-like
            time.sleep(random.uniform(1.5, 3.0))
            
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"    [!] 403 Forbidden (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    # Exponential backoff
                    wait_time = (2 ** attempt) * random.uniform(2, 4)
                    print(f"    Waiting {wait_time:.1f}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise
            else:
                raise
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = random.uniform(2, 4)
                time.sleep(wait_time)
            else:
                raise
    return None

def extract_publication_date(soup):
    """Extract publication date from PMC articles"""
    # Try multiple date selectors
    date_text = None
    
    # Method 1: Meta tags
    meta_date = (
        soup.find('meta', {'name': 'citation_publication_date'}) or
        soup.find('meta', {'name': 'citation_date'}) or
        soup.find('meta', {'property': 'article:published_time'})
    )
    if meta_date:
        date_text = meta_date.get('content', '')
    
    # Method 2: Specific date elements
    if not date_text:
        date_elem = (
            soup.find('span', class_='cit') or
            soup.find('time') or
            soup.find('span', class_='date')
        )
        if date_elem:
            date_text = date_elem.get_text(strip=True)
    
    # Try to parse the date
    if date_text:
        try:
            # Try common date formats
            for fmt in ['%Y-%m-%d', '%Y/%m/%d', '%B %d, %Y', '%d %B %Y', '%Y %b %d', '%Y']:
                try:
                    parsed_date = datetime.strptime(date_text.strip(), fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue
        except Exception:
            pass
    
    return 'Unknown'

def extract_keywords(soup, text):
    """Extract keywords from metadata or using YAKE"""
    keywords = []
    
    # Method 1: From meta tags
    meta_keywords = soup.find_all('meta', {'name': 'keywords'})
    if meta_keywords:
        for meta in meta_keywords:
            kw = meta.get('content', '')
            if kw:
                keywords.extend([k.strip() for k in kw.split(',')])
    
    # Method 2: From article keywords section
    if not keywords:
        kw_section = soup.find(['section', 'div'], class_=['kwd-group', 'keywords'])
        if kw_section:
            kw_elements = kw_section.find_all(['span', 'a', 'p'])
            keywords = [elem.get_text(strip=True) for elem in kw_elements if elem.get_text(strip=True)]
    
    # Method 3: Use YAKE on first section if no metadata keywords found
    if not keywords and text:
        # Use first 3000 characters for keyword extraction
        text_sample = text[:3000]
        try:
            yake_keywords = kw_extractor.extract_keywords(text_sample)
            keywords = [kw[0] for kw in yake_keywords]  # Get keyword text only
        except Exception as e:
            print(f"    [!] YAKE extraction failed: {e}")
    
    # Clean and return
    keywords = [kw for kw in keywords if kw and len(kw) > 2][:15]  # Max 15 keywords
    return ', '.join(keywords) if keywords else 'Not found'

def clean_article_body(article_body):
    """Remove navigation, references, and other junk from article body"""
    # Selectors for common junk sections to remove
    junk_selectors = [
        'nav', 'footer', 
        '.references', '.ref-list', '#references', '#bibliography',
        '.author-notes', '.author-information', '.author-contributions',
        '.c-article-header', '.c-article-info-details',
        '.acknowledgments', '.ack',
        '.supplementary-material', '.app-group',
        '.copyright', '.license',
        'script', 'style', 'noscript'
    ]
    
    # Remove all matching elements
    for selector in junk_selectors:
        for element in article_body.select(selector):
            element.decompose()
    
    return article_body
def extract_authors(soup):
    """Extract authors from PMC articles"""
    authors = []
    
    # Try multiple common author selectors
    author_elements = (
        soup.find_all('a', class_='author-name') or
        soup.find_all('span', class_='contrib-name') or
        soup.find_all('span', class_='name') or
        soup.find('div', class_='contrib-group')
    )
    
    if author_elements:
        if isinstance(author_elements, list):
            for elem in author_elements[:10]:  # Limit to first 10 authors
                name = elem.get_text(strip=True)
                if name and len(name) > 2:
                    authors.append(name)
        else:
            # If it's a div, find all names within it
            names = author_elements.find_all(['span', 'a'])
            for name_elem in names[:10]:
                name = name_elem.get_text(strip=True)
                if name and len(name) > 2 and name not in ['and', 'et al.']:
                    authors.append(name)
    
    # Fallback: search in meta tags
    if not authors:
        meta_authors = soup.find_all('meta', {'name': 'citation_author'})
        authors = [meta.get('content', '').strip() for meta in meta_authors[:10]]
    
    return ', '.join(authors) if authors else 'Unknown'

def extract_abstract(soup):
    """Extract abstract from PMC articles"""
    # Try multiple selectors for abstract
    abstract_elem = (
        soup.find('div', class_='abstract') or
        soup.find('div', id='abstract') or
        soup.find('section', class_='abstract') or
        soup.find('abstract')
    )
    
    if abstract_elem:
        # Remove headings like "Abstract", "Background", etc.
        for heading in abstract_elem.find_all(['h1', 'h2', 'h3', 'h4', 'title']):
            heading.decompose()
        
        abstract_text = abstract_elem.get_text(separator=' ', strip=True)
        
        # Clean up common artifacts
        abstract_text = abstract_text.replace('Abstract', '').strip()
        
        return abstract_text if len(abstract_text) > 50 else 'Not found'
    
    return 'Not found'

def extract_conclusion(soup):
    """Extract conclusion/discussion from PMC articles"""
    # Try multiple selectors for conclusion
    conclusion_elem = None
    
    # Search for sections with conclusion-related titles
    for section in soup.find_all(['div', 'section']):
        section_title = section.find(['h1', 'h2', 'h3', 'h4', 'title'])
        if section_title:
            title_text = section_title.get_text(strip=True).lower()
            if any(keyword in title_text for keyword in ['conclusion', 'concluding', 'summary', 'discussion']):
                conclusion_elem = section
                break
    
    if conclusion_elem:
        # Remove the heading
        for heading in conclusion_elem.find_all(['h1', 'h2', 'h3', 'h4', 'title']):
            heading.decompose()
        
        conclusion_text = conclusion_elem.get_text(separator=' ', strip=True)
        
        # Limit length to avoid extremely long conclusions
        if len(conclusion_text) > 2000:
            conclusion_text = conclusion_text[:2000] + '...'
        
        return conclusion_text if len(conclusion_text) > 50 else 'Not found'
    
    return 'Not found'

# --- PROCESSING ---
print(f"Starting processing of {CSV_FILE_PATH}...")

try:
    csv_iterator = pd.read_csv(
        CSV_FILE_PATH,
        chunksize=ARTICLE_BATCH_SIZE,
        header=0
    )
except FileNotFoundError:
    print(f"Error: File '{CSV_FILE_PATH}' not found.")
    exit()

total_articles_processed = 0
total_chunks_created = 0

for batch_num, batch_df in enumerate(csv_iterator):
    print(f"\n--- Processing Batch {batch_num+1} ---")
    
    # Collect all chunks from this batch before embedding
    all_chunks = []
    chunk_metadata = []
    
    # Step 1: Scrape and chunk all articles in batch
    for index, row in batch_df.iterrows():
        title = row['Title']
        link = row['Link']
        
        print(f"  Scraping: {title[:50]}...")

        try:
            response = fetch_article(link)
            if not response:
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract metadata
            authors = extract_authors(soup)
            pub_date = extract_publication_date(soup)
            abstract = extract_abstract(soup)
            conclusion = extract_conclusion(soup)
            
            article_body = (soup.find('div', id='mc') or 
                          soup.find('div', class_='article') or
                          soup.find('article') or
                          soup.find('body'))

            if not article_body:
                print(f"    [!] Could not find article body")
                log_failure(link, "Article body not found")
                continue

            # Clean the article body
            article_body = clean_article_body(article_body)
            full_text = article_body.get_text(separator=' ', strip=True)

            if len(full_text) < 100:
                print(f"    [!] Text too short, skipping")
                log_failure(link, "Text too short after extraction")
                continue

            # Extract keywords (using metadata or YAKE)
            keywords = extract_keywords(soup, full_text)
            
            print(f"    Authors: {authors[:50]}...")
            print(f"    Date: {pub_date}")
            print(f"    Keywords: {keywords[:60]}...")
            print(f"    Abstract: {'Found' if abstract != 'Not found' else 'Not found'}")
            print(f"    Conclusion: {'Found' if conclusion != 'Not found' else 'Not found'}")

            chunks = text_splitter.split_text(full_text)
            print(f"    Created {len(chunks)} chunks")
            
            # Store chunks with metadata
            for j, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    'title': title,
                    'link': link,
                    'chunk_id': j + 1,
                    'authors': authors,
                    'publication_date': pub_date,
                    'keywords': keywords,
                    'abstract': abstract,
                    'conclusion': conclusion
                })
            
            total_articles_processed += 1
            
            # No additional sleep needed - already done in fetch_article()

        except requests.exceptions.RequestException as e:
            print(f"    [!] Error fetching {link}: {e}")
            log_failure(link, f"Request error: {str(e)}")
        except Exception as e:
            print(f"    [!] Unexpected error: {e}")
            log_failure(link, f"Unexpected error: {str(e)}")

    # Step 2: Generate embeddings for ALL chunks at once (GPU optimization)
    if all_chunks:
        print(f"\n  Generating embeddings for {len(all_chunks)} chunks...")
        
        if device == 'cuda':
            print(f"  VRAM before embedding: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        try:
            # Process in large batches to maximize GPU utilization
            all_embeddings = model.encode(
                all_chunks,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Enable for better search quality
            )
            
            if device == 'cuda':
                print(f"  VRAM after embedding: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
                print(f"  Peak VRAM: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
            
            total_chunks_created += len(all_chunks)
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  [!] GPU OOM. Reducing batch size and retrying...")
                torch.cuda.empty_cache()
                gc.collect()
                
                # Fallback with smaller batch
                all_embeddings = model.encode(
                    all_chunks,
                    batch_size=64,
                    show_progress_bar=True,
                    convert_to_numpy=True
                )
            else:
                raise e

        # Step 3: Prepare data for insertion
        rows_to_insert = []
        for i, (chunk, metadata, embedding) in enumerate(zip(all_chunks, chunk_metadata, all_embeddings)):
            rows_to_insert.append({
                'title': metadata['title'],
                'link': metadata['link'],
                'chunk_id': metadata['chunk_id'],
                'authors': metadata['authors'],
                'publication_date': metadata['publication_date'],
                'keywords': metadata['keywords'],
                'abstract': metadata['abstract'],
                'conclusion': metadata['conclusion'],
                'content': chunk,
                'embedding': embedding.tolist()
            })

        # Step 4: Upload to Supabase
        print(f"  Uploading {len(rows_to_insert)} chunks to Supabase...")
        try:
            # Upload in sub-batches if too large
            upload_batch_size = 100
            for i in range(0, len(rows_to_insert), upload_batch_size):
                sub_batch = rows_to_insert[i:i+upload_batch_size]
                supabase.table(TABLE_NAME).insert(sub_batch).execute()
            print("  Upload successful.")
        except Exception as e:
            print(f"  [!] Error uploading: {e}")

        # Clear memory
        if device == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

print(f"\n--- Processing Complete ---")
print(f"Total articles processed: {total_articles_processed}")
print(f"Total chunks created: {total_chunks_created}")

if device == 'cuda':
    print(f"Final VRAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
