# Organization Named Entity Recognition (NER) Streamlit App

A sophisticated web application for extracting and identifying organizations from documents using BERT-based Named Entity Recognition combined with database matching.

## Features

### 🔍 **Dual Organization Detection**
- **Database Matching**: Identifies known organizations from a PostgreSQL database containing 18,000+ organizations
- **NER Discovery**: Uses BERT-large-cased-finetuned-conll03-english to discover new organizations not in the database
- **Smart Alias Generation**: Automatically creates acronyms and aliases (e.g., "CIPS" from "Chartered Institute of Procurement & Supply (CIPS)")

### 📄 **Document Support**
- **Multiple Formats**: TXT, PDF, DOCX
- **Large Document Handling**: Processes documents up to any size with intelligent chunking
- **Text Preview**: Shows document content before processing

### 🎯 **Advanced Filtering**
- **Context Validation**: Ensures organizations are mentioned in appropriate business contexts
- **Overlap Detection**: Prevents duplicate matches within the same text span
- **Generic Term Filtering**: Filters out common business terms that aren't organizations
- **Confidence Thresholds**: Adjustable confidence levels for both database and NER matches

### 📊 **Export Options**
- **CSV Export**: Separate downloads for database matches and new discoveries
- **TXT Export**: Comma-separated list of all unique organizations found
- **Detailed Metadata**: Includes confidence scores, organization IDs, and detection methods

### ⚙️ **Configuration**
- **Adjustable Confidence**: Slider to control detection sensitivity
- **Real-time Statistics**: Shows database size and lookup entries
- **Manual Addition**: Add organizations missed by automated detection

## Installation

### Prerequisites
```bash
pip install -r requirements.txt
```

### Database Configuration
1. **Copy the environment template**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` with your database credentials**:
   ```bash
   POSTGRES_HOST=your-postgres-host
   POSTGRES_PORT=5432
   POSTGRES_DATABASE=your-database-name
   POSTGRES_USER=your-username
   POSTGRES_PASSWORD=your-password
   ```

3. **The .env file is automatically excluded from git** to keep your credentials secure.

## Usage

### Run the Application
```bash
streamlit run ner_streamlit.py
```

### Process Documents
1. **Upload**: Select TXT, PDF, or DOCX file
2. **Review**: Check extracted text in the preview
3. **Extract**: Automatic organization detection begins
4. **Review Results**: 
   - ✅ **Database Matches**: Known organizations with high confidence
   - 🔍 **New Discoveries**: Potential new organizations requiring review
5. **Export**: Download results in CSV or TXT format

## Architecture

### Core Components

#### `OrganizationExtractor`
- **Database Integration**: Loads and caches organization data
- **Lookup Dictionary**: 18,000+ entries with aliases and acronyms
- **NER Pipeline**: BERT-based entity recognition
- **Fuzzy Matching**: Handles variations in organization names

#### **Detection Methods**
1. **Database Lookup**: Fast exact and fuzzy matching against known organizations
2. **NER Detection**: BERT model identifies organization entities in text
3. **Fuzzy Matching**: Connects NER discoveries to similar database entries

#### **Quality Filters**
- Minimum length requirements (configurable)
- Generic term exclusion
- Context validation for business relevance
- Confidence scoring and thresholds

## Test Results

Successfully identifies organizations from complex business documents:

**Example: P5 Text.docx (Supply Chain Risk Report)**
- **Input**: 13,773 character document
- **Found**: 10/11 expected organizations as database matches
- **Discovered**: 1 new organization (WuXi AppTec) via NER
- **Processing Time**: ~2-3 seconds

### Expected Organizations:
✅ AstraZeneca, CIPS, Deloitte, GAN Integrity, NVIDIA, OneTrust, Pfizer, Taiwan Semiconductor Manufacturing Company, Volvo, World Economic Forum  
🔍 WuXi AppTec (detected as new discovery)

## Technical Specifications

- **NER Model**: `dbmdz/bert-large-cased-finetuned-conll03-english`
- **Database**: PostgreSQL with `verdantix.org` table
- **Text Processing**: 4,000 character chunks with 500 character overlap
- **Confidence Threshold**: 0.85 (configurable)
- **Performance**: Processes ~1,000 words per second

## File Structure

```
streamlit/
├── ner_streamlit.py           # Main Streamlit application
├── organizations.db           # SQLite backup (if needed)
├── Test Files/
│   └── Test1/
│       ├── P5 Text.docx       # Sample test document
│       └── Orgs.txt           # Expected organizations list
├── .gitignore                 # Git ignore patterns
└── README.md                  # This file
```

## Recent Improvements

- **Enhanced Alias Generation**: Extracts acronyms from parentheses
- **Smart Short Name Handling**: Allows known short organizations (CIPS, NVIDIA, etc.)
- **Improved Context Validation**: Better business context detection
- **Multiple Export Formats**: CSV and TXT options
- **Cleaner Interface**: Removed debug logging for production use

## Contributing

This tool is designed for business document analysis and organization extraction. Contributions welcome for:
- Additional document format support
- Enhanced NER model integration
- Database schema improvements
- UI/UX enhancements

## License

Developed for Verdantix organization analysis and research purposes.