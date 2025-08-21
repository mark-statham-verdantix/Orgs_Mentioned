"""
Organization Extraction Streamlit Application
=============================================

Interactive web application for analysts to upload documents,
extract organizations, and manage database additions.
"""

import streamlit as st
import pandas as pd
import numpy as np
import re
import time
import json
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import psycopg2
from sqlalchemy import create_engine, text
from transformers import AutoTokenizer, pipeline
from fuzzywuzzy import fuzz, process
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Import document processing libraries
import PyPDF2
import docx
from io import BytesIO

class OrganizationExtractor:
    """Core organization extraction system for Streamlit app"""
    
    def __init__(self, postgres_config: Dict[str, str]):
        self.postgres_config = postgres_config
        self.min_confidence = 0.85  # Conservative threshold for production
        self.min_org_length = 3
        
        # Initialize components
        self.master_orgs_df = pd.DataFrame()
        self.org_lookup = {}
        self.tokenizer = None
        self.ner_model = None
        
        # Initialize system
        self._initialize_system()
    
    @st.cache_resource
    def _initialize_system(_self):
        """Initialize system components with caching"""
        try:
            # Initialize tokenizer
            _self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            
            # Load NER model
            _self.ner_model = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=-1
            )
            
            # Connect to database and load organizations
            _self._load_organizations()
            _self._build_lookup()
            
            return True
            
        except Exception as e:
            st.error(f"System initialization failed: {e}")
            return False
    
    def _load_organizations(self):
        """Load organizations from database"""
        try:
            conn = psycopg2.connect(
                host=self.postgres_config['host'],
                port=self.postgres_config.get('port', 5432),
                database=self.postgres_config['database'],
                user=self.postgres_config['user'],
                password=self.postgres_config['password']
            )
            
            query = """
            SELECT org_id, org_name
            FROM verdantix.org
            WHERE org_name IS NOT NULL 
                AND LENGTH(TRIM(org_name)) > 2
            ORDER BY org_name
            """
            
            self.master_orgs_df = pd.read_sql_query(query, conn)
            conn.close()
            
        except Exception as e:
            st.error(f"Database connection failed: {e}")
            self.master_orgs_df = pd.DataFrame()
    
    def _build_lookup(self):
        """Build organization lookup dictionary"""
        self.org_lookup = {}
        
        for _, org in self.master_orgs_df.iterrows():
            org_id = org['org_id']
            org_name = str(org['org_name']).strip()
            
            if len(org_name) > 2:
                self.org_lookup[org_name.lower()] = {
                    'org_id': org_id,
                    'canonical': org_name,
                    'confidence': 1.0
                }
                
                # Add aliases
                aliases = self._generate_aliases(org_name)
                for alias in aliases:
                    if alias.lower() not in self.org_lookup and len(alias) > 2:
                        self.org_lookup[alias.lower()] = {
                            'org_id': org_id,
                            'canonical': org_name,
                            'confidence': 0.85
                        }
    
    def _generate_aliases(self, org_name: str) -> List[str]:
        """Generate aliases for organization names"""
        aliases = []
        
        # Extract acronyms from parentheses (e.g., "World Economic Forum (WEF)" -> "WEF")
        paren_pattern = r'\(([^)]+)\)'
        paren_matches = re.findall(paren_pattern, org_name)
        for match in paren_matches:
            cleaned = match.strip()
            if 2 <= len(cleaned) <= 10:
                aliases.append(cleaned)
        
        # Remove business suffixes
        suffix_patterns = [
            r'\s+(?:Inc\.?|Corporation|Corp\.?|Company|Co\.?|Limited|Ltd\.?)',
            r'\s+(?:LLC|LLP|LP|PLC|Group|Holdings?)'
        ]
        
        for pattern in suffix_patterns:
            base = re.sub(pattern + r'$', '', org_name, flags=re.IGNORECASE).strip()
            if base != org_name and len(base) > 2:
                aliases.append(base)
        
        # Create acronyms from main text (excluding parentheses)
        main_text = re.sub(paren_pattern, '', org_name).strip()
        words = main_text.split()
        if len(words) > 1:
            # Skip common words when creating acronyms
            skip_words = {'of', 'the', 'and', 'for', 'in', 'on', 'at', 'to', 'a', 'an', '&'}
            meaningful_words = [w for w in words if w.lower() not in skip_words and len(w) > 0]
            if len(meaningful_words) > 1:
                acronym = ''.join([w[0].upper() for w in meaningful_words])
                if 2 <= len(acronym) <= 8:
                    aliases.append(acronym)
        
        return list(set(aliases))
    
    def _is_generic_term(self, term: str) -> bool:
        """Filter generic business terms"""
        generic_terms = {
            'ai', 'iot', 'esg', 'api', 'cloud', 'data', 'tech', 'digital',
            'smart', 'green', 'cyber', 'auto', 'bio', 'blockchain', 'fintech',
            'saas', 'crm', 'erp', 'hr', 'it', 'covid', 'gdpr'
        }
        return term.lower() in generic_terms
    
    def _validate_context(self, text: str, start: int, end: int) -> bool:
        """Validate organizational context"""
        window = 100
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        context = text[context_start:context_end].lower()
        
        org_indicators = [
            'company', 'corporation', 'inc', 'founded', 'ceo', 'announced',
            'partnership', 'acquisition', 'investment', 'subsidiary'
        ]
        
        return any(indicator in context for indicator in org_indicators)
    
    def extract_organizations(self, text: str) -> Tuple[List[Dict], List[Dict]]:
        """Extract organizations and categorize as known vs unknown"""
        
        db_matches = []
        ner_discoveries = []
        
        # Method 1: Database matches
        text_lower = text.lower()
        sorted_terms = sorted(self.org_lookup.keys(), key=len, reverse=True)
        matched_positions = set()
        
        for term in sorted_terms:
            if (len(term) >= self.min_org_length and 
                not self._is_generic_term(term)):
                
                # Require multi-word for short terms, but allow known brands
                if len(term) < 8 and ' ' not in term:
                    # Allow well-known organization names even if short
                    known_short_orgs = {'pfizer', 'nvidia', 'volvo', 'tesla', 'apple', 'google', 'meta', 'uber', 'cips'}
                    if term.lower() not in known_short_orgs:
                        continue
                
                pattern = r'\b' + re.escape(term) + r'\b'
                
                for match in re.finditer(pattern, text_lower):
                    start, end = match.span()
                    
                    if not any(start < e and s < end for s, e in matched_positions):
                        matched_positions.add((start, end))
                        
                        org_info = self.org_lookup[term]
                        
                        # Context validation for shorter terms
                        if len(term) < 10:
                            if not self._validate_context(text, start, end):
                                continue
                        
                        db_matches.append({
                            'text': text[start:end],
                            'canonical': org_info['canonical'],
                            'confidence': org_info['confidence'],
                            'org_id': org_info['org_id'],
                            'method': 'database'
                        })
        
        # Method 2: NER for new organizations
        if self.ner_model:
            try:
                if len(text) > 4000:
                    chunks = [text[i:i+4000] for i in range(0, len(text), 3500)]
                else:
                    chunks = [text]
                
                for chunk in chunks:
                    predictions = self.ner_model(chunk)
                    
                    for pred in predictions:
                        if 'ORG' in pred.get('entity_group', ''):
                            org_text = pred['word'].strip()
                            
                            # Clean tokenization artifacts
                            org_text = re.sub(r'^##', '', org_text)
                            org_text = re.sub(r'[^\w\s&.-]', '', org_text)
                            org_text = ' '.join(org_text.split())
                            
                            if (len(org_text) >= self.min_org_length and
                                pred['score'] >= 0.8 and
                                not self._is_generic_term(org_text)):
                                
                                # Check if already in database
                                if org_text.lower() not in self.org_lookup:
                                    # Try fuzzy matching
                                    if len(self.master_orgs_df) > 0:
                                        canonical_names = self.master_orgs_df['org_name'].tolist()
                                        best_match = process.extractOne(
                                            org_text, canonical_names, scorer=fuzz.ratio
                                        )
                                        
                                        if best_match and best_match[1] >= 85:
                                            # Close match found - add to DB matches
                                            matched_row = self.master_orgs_df[
                                                self.master_orgs_df['org_name'] == best_match[0]
                                            ].iloc[0]
                                            
                                            db_matches.append({
                                                'text': org_text,
                                                'canonical': matched_row['org_name'],
                                                'confidence': pred['score'] * (best_match[1] / 100),
                                                'org_id': matched_row['org_id'],
                                                'method': 'ner_fuzzy'
                                            })
                                        else:
                                            # New organization discovery
                                            ner_discoveries.append({
                                                'text': org_text,
                                                'confidence': pred['score'],
                                                'method': 'ner_new'
                                            })
            
            except Exception as e:
                st.error(f"NER processing error: {e}")
        
        # Deduplicate
        db_matches = self._deduplicate_matches(db_matches)
        ner_discoveries = self._deduplicate_matches(ner_discoveries)
        
        return db_matches, ner_discoveries
    
    def _deduplicate_matches(self, matches: List[Dict]) -> List[Dict]:
        """Remove duplicate matches"""
        seen = set()
        deduplicated = []
        
        for match in sorted(matches, key=lambda x: x['confidence'], reverse=True):
            canonical = match.get('canonical', match['text']).lower()
            if canonical not in seen:
                seen.add(canonical)
                deduplicated.append(match)
        
        return deduplicated


def extract_text_from_file(uploaded_file) -> str:
    """Extract text from uploaded file"""
    file_type = uploaded_file.type
    
    try:
        if file_type == "text/plain":
            return str(uploaded_file.read(), "utf-8")
        
        elif file_type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(BytesIO(uploaded_file.read()))
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text
        
        else:
            st.error(f"Unsupported file type: {file_type}")
            return ""
            
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return ""


def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Organization Extraction Tool",
        page_icon="🏢",
        layout="wide"
    )
    
    st.title("Organization Extraction & Database Management")
    st.markdown("Upload documents to automatically extract and categorize organizations")
    
    # Configuration from environment variables
    postgres_config = {
        'host': os.getenv('POSTGRES_HOST', 'localhost'),
        'port': os.getenv('POSTGRES_PORT', '5432'),
        'database': os.getenv('POSTGRES_DATABASE', 'postgres'),
        'user': os.getenv('POSTGRES_USER', 'postgres'),
        'password': os.getenv('POSTGRES_PASSWORD', '')
    }
    
    # Validate configuration
    if not all([postgres_config['host'], postgres_config['user'], postgres_config['password']]):
        st.error("⚠️ **Database configuration missing!** Please check your .env file.")
        st.info("Create a .env file with your database credentials. See .env.example for the template.")
        st.stop()
    
    # Initialize extractor
    if 'extractor' not in st.session_state:
        with st.spinner("Initializing extraction system..."):
            st.session_state.extractor = OrganizationExtractor(postgres_config)
    
    # Check system status
    extractor = st.session_state.extractor
    if len(extractor.org_lookup) == 0:
        st.error("⚠️ **CRITICAL**: Lookup dictionary is empty! Database loading failed.")
        with st.spinner("Reloading database..."):
            extractor._load_organizations()
            extractor._build_lookup()
    
    extractor = st.session_state.extractor
    
    # Sidebar configuration
    st.sidebar.header("Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.5, 
        max_value=1.0, 
        value=0.85, 
        step=0.05
    )
    extractor.min_confidence = confidence_threshold
    
    # Database statistics
    st.sidebar.metric("Organizations in Database", len(extractor.master_orgs_df))
    st.sidebar.metric("Lookup Entries", len(extractor.org_lookup))
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['txt', 'pdf', 'docx'],
        help="Supported formats: TXT, PDF, DOCX"
    )
    
    if uploaded_file is not None:
        # Extract text
        with st.spinner("Extracting text from document..."):
            text = extract_text_from_file(uploaded_file)
        
        if text:
            st.success(f"Extracted {len(text):,} characters from {uploaded_file.name}")
            
            # Text preview
            with st.expander("Document Preview"):
                st.text_area("Text Content", text[:1000] + "..." if len(text) > 1000 else text, height=200)
            
            # Extract organizations
            with st.spinner("Extracting organizations..."):
                start_time = time.time()
                db_matches, ner_discoveries = extractor.extract_organizations(text)
                processing_time = time.time() - start_time
            
            # Results summary
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Database Matches", len(db_matches))
            with col2:
                st.metric("New Discoveries", len(ner_discoveries))
            with col3:
                st.metric("Processing Time", f"{processing_time:.2f}s")
            
            # Display results
            if db_matches or ner_discoveries:
                
                # Database Matches Tab
                st.subheader("✅ Organizations Found in Database")
                if db_matches:
                    db_df = pd.DataFrame(db_matches)
                    db_df['confidence'] = db_df['confidence'].round(3)
                    
                    # Allow filtering
                    selected_db = st.multiselect(
                        "Select confirmed organizations:",
                        options=range(len(db_df)),
                        default=list(range(len(db_df))),
                        format_func=lambda x: f"{db_df.iloc[x]['canonical']} ({db_df.iloc[x]['confidence']:.2f})"
                    )
                    
                    if selected_db:
                        confirmed_orgs = db_df.iloc[selected_db]
                        st.dataframe(confirmed_orgs[['canonical', 'confidence', 'method']], use_container_width=True)
                else:
                    st.info("No organizations found in database")
                
                # New Discoveries Tab
                st.subheader("🔍 Potential New Organizations (Require Review)")
                if ner_discoveries:
                    st.warning(f"Found {len(ner_discoveries)} potential new organizations that need manual review")
                    
                    for i, discovery in enumerate(ner_discoveries):
                        with st.container():
                            col1, col2, col3 = st.columns([3, 1, 1])
                            
                            with col1:
                                st.write(f"**{discovery['text']}** (confidence: {discovery['confidence']:.2f})")
                            
                            with col2:
                                approve = st.button(f"Approve", key=f"approve_{i}")
                            
                            with col3:
                                reject = st.button(f"Reject", key=f"reject_{i}")
                            
                            if approve:
                                st.success(f"✅ Approved: {discovery['text']}")
                                # Here you would add to pending database additions
                            
                            if reject:
                                st.error(f"❌ Rejected: {discovery['text']}")
                else:
                    st.info("No new organizations discovered")
                
                # Manual Addition Section
                st.subheader("➕ Manual Organization Addition")
                with st.form("manual_addition"):
                    new_org_name = st.text_input(
                        "Organization Name",
                        placeholder="Enter organization name not found by the system"
                    )
                    
                    submitted = st.form_submit_button("Add to Pending List")
                    
                    if submitted and new_org_name:
                        if new_org_name.lower() in extractor.org_lookup:
                            st.warning(f"'{new_org_name}' already exists in database")
                        else:
                            st.success(f"Added '{new_org_name}' to pending additions")
                            # Here you would add to pending database additions
                
                # Export Results
                st.subheader("📥 Export Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if db_matches:
                        csv_data = pd.DataFrame(db_matches).to_csv(index=False)
                        st.download_button(
                            label="📊 Download CSV (Database Matches)",
                            data=csv_data,
                            file_name=f"db_matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if ner_discoveries:
                        csv_data = pd.DataFrame(ner_discoveries).to_csv(index=False)
                        st.download_button(
                            label="🔍 Download CSV (New Discoveries)",
                            data=csv_data,
                            file_name=f"new_orgs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col3:
                    if db_matches or ner_discoveries:
                        # Create combined organization list for .txt export
                        all_orgs = []
                        
                        # Add database matches
                        for match in db_matches:
                            all_orgs.append(match['canonical'])
                        
                        # Add new discoveries
                        for discovery in ner_discoveries:
                            all_orgs.append(discovery['text'])
                        
                        # Remove duplicates and sort
                        unique_orgs = sorted(list(set(all_orgs)))
                        
                        # Create comma-separated string
                        txt_content = ", ".join(unique_orgs)
                        
                        st.download_button(
                            label="📝 Download TXT (All Organizations)",
                            data=txt_content,
                            file_name=f"organizations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
            
            else:
                st.warning("No organizations found in the document")
    
    # Footer
    st.markdown("---")
    st.markdown("**Note:** This tool uses BERT-based NER combined with database matching for organization extraction.")


if __name__ == "__main__":
    main()