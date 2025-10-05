import os
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from supabase import create_client, Client
from langchain.text_splitter import RecursiveCharacterTextSplitter
import gc
from urllib.parse import urlparse
import random
import yake
from datetime import datetime
import argparse
from dotenv import load_dotenv

# --- CONFIGURATION ---

load_dotenv() # Loads variables from .env file

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("✗ Error: Supabase credentials not found in .env file.")
    exit()


# CPU-optimized settings
ARTICLE_BATCH_SIZE = 5  # Smaller batch for memory efficiency
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 120
EMBEDDING_BATCH_SIZE = 32  # Smaller batch for CPU - adjust based on RAM
FAILED_URLS_LOG = 'failed_urls.log'


# YAKE keyword extraction parameters
YAKE_LANGUAGE = "en"
YAKE_MAX_NGRAM = 3
YAKE_NUM_KEYWORDS = 10

# Headers to mimic a real browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}
# --- COMMAND LINE ARGUMENTS ---

parser = argparse.ArgumentParser(description='Process articles with embeddings (CPU-optimized)')
parser.add_argument('--file', type=str, default='SB_publication_PMC.csv', help='Input CSV file path')
parser.add_argument('--table', type=str, default='articles.csv', help='Input Table')
parser.add_argument('--output-suffix', type=str, default='', help='Suffix for output log files')

args = parser.parse_args()

CSV_FILE_PATH = args.file
TABLE_NAME = args.table

# Processing range (for parallel execution)
START_ROW = 0  # Default start
END_ROW = None  # Default end (all rows)

if args.output_suffix:
    FAILED_URLS_LOG = f'failed_urls_{args.output_suffix}.log'
# --- INITIALIZATION ---
print("="*60)
print("CPU-OPTIMIZED EMBEDDING PROCESSOR")
print("="*60)
print(f"Processing rows: {START_ROW} to {END_ROW if END_ROW else 'END'}")
print(f"Embedding batch size: {EMBEDDING_BATCH_SIZE}")
print(f"Output suffix: {args.output_suffix if args.output_suffix else 'none'}")
print("="*60)

# Force CPU usage - ensures identical embeddings across runs
device = 'cpu'
print(f"Device: {device} (forced for consistency)")

try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✓ Supabase client initialized")
except Exception as e:
    print(f"✗ Error initializing Supabase: {e}")
    exit()

# Load model with explicit CPU device
try:
    print("\nLoading embedding model...")
    model = SentenceTransformer('all-MiniLM-L12-v2', device=device)
    model.max_seq_length = 384
    
    # CRITICAL: Set to eval mode and disable gradients for consistency
    model.eval()
    
    # Warm up model with deterministic test
    print("Warming up model...")
    test_embedding = model.encode(["test sentence"], show_progress_bar=False)
    print(f"✓ Model loaded successfully (embedding dim: {len(test_embedding[0])})")
    
    # Clear any cached data
    gc.collect()
except Exception as e:
    print(f"✗ Error loading model: {e}")
    exit()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

session = requests.Session()
session.headers.update(HEADERS)

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
            time.sleep(random.uniform(1.5, 3.0))
            response = session.get(url, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                print(f"    [!] 403 Forbidden (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
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
    date_text = None
    
    meta_date = (
        soup.find('meta', {'name': 'citation_publication_date'}) or
        soup.find('meta', {'name': 'citation_date'}) or
        soup.find('meta', {'property': 'article:published_time'})
    )
    if meta_date:
        date_text = meta_date.get('content', '')
    
    if not date_text:
        date_elem = (
            soup.find('span', class_='cit') or
            soup.find('time') or
            soup.find('span', class_='date')
        )
        if date_elem:
            date_text = date_elem.get_text(strip=True)
    
    if date_text:
        try:
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
    
    meta_keywords = soup.find_all('meta', {'name': 'keywords'})
    if meta_keywords:
        for meta in meta_keywords:
            kw = meta.get('content', '')
            if kw:
                keywords.extend([k.strip() for k in kw.split(',')])
    
    if not keywords:
        kw_section = soup.find(['section', 'div'], class_=['kwd-group', 'keywords'])
        if kw_section:
            kw_elements = kw_section.find_all(['span', 'a', 'p'])
            keywords = [elem.get_text(strip=True) for elem in kw_elements if elem.get_text(strip=True)]
    
    if not keywords and text:
        text_sample = text[:3000]
        try:
            yake_keywords = kw_extractor.extract_keywords(text_sample)
            keywords = [kw[0] for kw in yake_keywords]
        except Exception as e:
            print(f"    [!] YAKE extraction failed: {e}")
    
    keywords = [kw for kw in keywords if kw and len(kw) > 2][:15]
    return ', '.join(keywords) if keywords else 'Not found'

def clean_article_body(article_body):
    """Remove navigation, references, and other junk from article body"""
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
    
    for selector in junk_selectors:
        for element in article_body.select(selector):
            element.decompose()
    
    return article_body

def extract_authors(soup):
    """Extract authors from PMC articles"""
    authors = []
    
    author_elements = (
        soup.find_all('a', class_='author-name') or
        soup.find_all('span', class_='contrib-name') or
        soup.find_all('span', class_='name') or
        soup.find('div', class_='contrib-group')
    )
    
    if author_elements:
        if isinstance(author_elements, list):
            for elem in author_elements[:10]:
                name = elem.get_text(strip=True)
                if name and len(name) > 2:
                    authors.append(name)
        else:
            names = author_elements.find_all(['span', 'a'])
            for name_elem in names[:10]:
                name = name_elem.get_text(strip=True)
                if name and len(name) > 2 and name not in ['and', 'et al.']:
                    authors.append(name)
    
    if not authors:
        meta_authors = soup.find_all('meta', {'name': 'citation_author'})
        authors = [meta.get('content', '').strip() for meta in meta_authors[:10]]
    
    return ', '.join(authors) if authors else 'Unknown'

def extract_abstract(soup):
    """Extract abstract from PMC articles"""
    abstract_elem = (
        soup.find('div', class_='abstract') or
        soup.find('div', id='abstract') or
        soup.find('section', class_='abstract') or
        soup.find('abstract')
    )
    
    if abstract_elem:
        for heading in abstract_elem.find_all(['h1', 'h2', 'h3', 'h4', 'title']):
            heading.decompose()
        
        abstract_text = abstract_elem.get_text(separator=' ', strip=True)
        abstract_text = abstract_text.replace('Abstract', '').strip()
        
        return abstract_text if len(abstract_text) > 50 else 'Not found'
    
    return 'Not found'

def extract_conclusion(soup):
    """Extract conclusion/discussion from PMC articles"""
    conclusion_elem = None
    
    for section in soup.find_all(['div', 'section']):
        section_title = section.find(['h1', 'h2', 'h3', 'h4', 'title'])
        if section_title:
            title_text = section_title.get_text(strip=True).lower()
            if any(keyword in title_text for keyword in ['conclusion', 'concluding', 'summary', 'discussion']):
                conclusion_elem = section
                break
    
    if conclusion_elem:
        for heading in conclusion_elem.find_all(['h1', 'h2', 'h3', 'h4', 'title']):
            heading.decompose()
        
        conclusion_text = conclusion_elem.get_text(separator=' ', strip=True)
        
        if len(conclusion_text) > 2000:
            conclusion_text = conclusion_text[:2000] + '...'
        
        return conclusion_text if len(conclusion_text) > 50 else 'Not found'
    
    return 'Not found'

# --- PROCESSING ---
print(f"\nStarting processing of {CSV_FILE_PATH}...")

try:
    # Read the entire CSV first to handle row slicing
    df = pd.read_csv(CSV_FILE_PATH, header=0)
    total_rows = len(df)
    
    # Apply row range
    if END_ROW is None:
        END_ROW = total_rows
    
    df = df.iloc[START_ROW:END_ROW]
    print(f"Processing {len(df)} articles (rows {START_ROW} to {END_ROW})")
    
    # Create batches manually
    num_batches = (len(df) + ARTICLE_BATCH_SIZE - 1) // ARTICLE_BATCH_SIZE
    
except FileNotFoundError:
    print(f"Error: File '{CSV_FILE_PATH}' not found.")
    exit()

total_articles_processed = 0
total_chunks_created = 0
start_time = time.time()

for batch_num in range(num_batches):
    batch_start = batch_num * ARTICLE_BATCH_SIZE
    batch_end = min(batch_start + ARTICLE_BATCH_SIZE, len(df))
    batch_df = df.iloc[batch_start:batch_end]
    
    print(f"\n{'='*60}")
    print(f"Batch {batch_num+1}/{num_batches} (Articles {START_ROW + batch_start} to {START_ROW + batch_end})")
    print(f"{'='*60}")
    
    all_chunks = []
    chunk_metadata = []
    
    # Step 1: Scrape and chunk
    for index, row in batch_df.iterrows():
        title = row['Title']
        link = row['Link']
        
        print(f"\n  [{total_articles_processed + 1}] {title[:60]}...")

        try:
            response = fetch_article(link)
            if not response:
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            authors = extract_authors(soup)
            pub_date = extract_publication_date(soup)
            abstract = extract_abstract(soup)
            conclusion = extract_conclusion(soup)
            
            article_body = (soup.find('div', id='mc') or 
                          soup.find('div', class_='article') or
                          soup.find('article') or
                          soup.find('body'))

            if not article_body:
                print(f"    ✗ Article body not found")
                log_failure(link, "Article body not found")
                continue

            article_body = clean_article_body(article_body)
            full_text = article_body.get_text(separator=' ', strip=True)

            if len(full_text) < 100:
                print(f"    ✗ Text too short")
                log_failure(link, "Text too short after extraction")
                continue

            keywords = extract_keywords(soup, full_text)
            
            print(f"    ✓ Metadata extracted")
            print(f"      Authors: {authors[:40]}...")
            print(f"      Date: {pub_date}")

            chunks = text_splitter.split_text(full_text)
            print(f"    ✓ Created {len(chunks)} chunks")
            
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

        except requests.exceptions.RequestException as e:
            print(f"    ✗ Request error: {e}")
            log_failure(link, f"Request error: {str(e)}")
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")
            log_failure(link, f"Unexpected error: {str(e)}")

    # Step 2: Generate embeddings
    if all_chunks:
        print(f"\n  Generating embeddings for {len(all_chunks)} chunks...")
        print(f"  (This may take a while on CPU...)")
        
        try:
            # CPU-optimized encoding with progress bar
            all_embeddings = model.encode(
                all_chunks,
                batch_size=EMBEDDING_BATCH_SIZE,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True  # Important for consistency
            )
            
            print(f"  ✓ Embeddings generated")
            total_chunks_created += len(all_chunks)
            
        except Exception as e:
            print(f"  ✗ Embedding error: {e}")
            print(f"  Trying with smaller batch size...")
            try:
                all_embeddings = model.encode(
                    all_chunks,
                    batch_size=16,
                    show_progress_bar=True,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
                print(f"  ✓ Embeddings generated (reduced batch)")
            except Exception as e2:
                print(f"  ✗ Failed even with smaller batch: {e2}")
                continue

        # Step 3: Prepare for insertion
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
            upload_batch_size = 100
            for i in range(0, len(rows_to_insert), upload_batch_size):
                sub_batch = rows_to_insert[i:i+upload_batch_size]
                supabase.table(TABLE_NAME).insert(sub_batch).execute()
            print(f"  ✓ Upload successful")
        except Exception as e:
            print(f"  ✗ Upload error: {e}")

        # Clear memory
        gc.collect()
    
    # Progress update
    elapsed = time.time() - start_time
    avg_time = elapsed / (batch_num + 1)
    remaining = avg_time * (num_batches - batch_num - 1)
    print(f"\n  Progress: {batch_num + 1}/{num_batches} batches")
    print(f"  Elapsed: {elapsed/60:.1f}m | Est. remaining: {remaining/60:.1f}m")

print(f"\n{'='*60}")
print(f"PROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Articles processed: {total_articles_processed}")
print(f"Total chunks created: {total_chunks_created}")
print(f"Total time: {(time.time() - start_time)/60:.1f} minutes")
print(f"Failed URLs logged to: {FAILED_URLS_LOG}")
print(f"{'='*60}")
