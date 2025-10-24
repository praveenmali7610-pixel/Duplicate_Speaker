from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import io
import requests
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import librosa
import soundfile as sf
from datetime import datetime

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max for CSV
app.config['ALLOWED_SHEET_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Lazy loading for encoders
pyannote_pipeline = None

# Environment variables
PYANNOTE_API_KEY = os.environ.get('PYANNOTE_API_KEY', None)

def get_pyannote_pipeline():
    """Lazy load Pyannote pipeline"""
    global pyannote_pipeline
    if pyannote_pipeline is None:
        if not PYANNOTE_API_KEY:
            raise ValueError("Pyannote API key not configured")
        from pyannote.audio import Inference
        pyannote_pipeline = Inference(
            "pyannote/embedding",
            use_auth_token=PYANNOTE_API_KEY
        )
    return pyannote_pipeline

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_SHEET_EXTENSIONS']

def download_audio_to_memory(url, timeout=30):
    """
    Download audio from URL directly to memory without saving to disk
    Returns: BytesIO object containing audio data
    """
    try:
        # Stream download to avoid loading entire file at once
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Store in memory buffer
        audio_buffer = io.BytesIO()
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                audio_buffer.write(chunk)
        
        audio_buffer.seek(0)
        return audio_buffer
    except Exception as e:
        print(f"Error downloading {url}: {str(e)}")
        return None

def process_audio_in_memory(audio_buffer, target_sr=16000):
    """
    Process audio from memory buffer without writing to disk
    Returns: numpy array of audio samples
    """
    try:
        # Load audio from memory buffer using soundfile
        audio_data, sr = sf.read(audio_buffer)
        
        # Convert to mono if stereo
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Resample if needed
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        
        return audio_data
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return None

def extract_embedding_pyannote_memory(audio_buffer):
    """Extract speaker embedding using Pyannote from memory buffer"""
    try:
        import tempfile
        
        # Get pipeline
        model = get_pyannote_pipeline()
        
        # Pyannote requires file path, use temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=True) as tmp_file:
            audio_buffer.seek(0)
            tmp_file.write(audio_buffer.read())
            tmp_file.flush()
            
            # Extract embedding
            embedding = model(tmp_file.name)
            
            # Average if multiple segments
            if len(embedding.shape) > 1:
                embedding = np.mean(embedding, axis=0)
            
            # File automatically deleted when exiting with block
            return embedding
    except Exception as e:
        print(f"Error extracting Pyannote embedding: {str(e)}")
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings"""
    return 1 - cosine(embedding1, embedding2)

def process_sheet_file(file):
    """Read CSV or Excel file and return DataFrame"""
    try:
        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif filename.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file)
        else:
            return None
        
        return df
    except Exception as e:
        print(f"Error reading sheet: {str(e)}")
        return None

def find_duplicates_from_urls(audio_urls, labels, threshold=0.30):
    """
    Find duplicate speakers from audio URLs using Pyannote (no disk storage)
    
    Args:
        audio_urls: List of audio file URLs
        labels: List of labels/identifiers for each audio
        threshold: Similarity threshold (default 0.30 for same-script scenarios)
    """
    results = []
    embeddings = []
    
    total = len(audio_urls)
    
    # Extract embeddings for all audio files (in-memory only)
    for idx, (url, label) in enumerate(zip(audio_urls, labels)):
        print(f"Processing {idx+1}/{total}: {label}")
        
        # Download audio to memory
        audio_buffer = download_audio_to_memory(url)
        if audio_buffer is None:
            embeddings.append({
                'label': label,
                'url': url,
                'embedding': None,
                'error': 'Download failed'
            })
            continue
        
        # Extract embedding using Pyannote
        embedding = extract_embedding_pyannote_memory(audio_buffer)
        
        # Clear memory buffer
        audio_buffer.close()
        del audio_buffer
        
        embeddings.append({
            'label': label,
            'url': url,
            'embedding': embedding,
            'error': None if embedding is not None else 'Processing failed'
        })
    
    # Compare all pairs
    duplicate_groups = []
    processed = set()
    all_comparisons = []
    
    for i, file1 in enumerate(embeddings):
        if i in processed or file1['embedding'] is None:
            continue
        
        group = [file1['label']]
        
        for j, file2 in enumerate(embeddings):
            if i >= j or j in processed or file2['embedding'] is None:
                continue
            
            similarity = calculate_similarity(file1['embedding'], file2['embedding'])
            
            if similarity >= threshold:
                group.append(file2['label'])
                processed.add(j)
            
            # Store comparison if similarity is significant
            if similarity >= (threshold - 0.1):
                all_comparisons.append({
                    'file1': file1['label'],
                    'file2': file2['label'],
                    'similarity': round(float(similarity), 4),
                    'is_duplicate': similarity >= threshold
                })
        
        if len(group) > 1:
            duplicate_groups.append({
                'speaker_id': f"Speaker_{len(duplicate_groups) + 1}",
                'files': group,
                'count': len(group)
            })
        
        processed.add(i)
    
    # Count unique speakers
    unique_speakers = total - sum(group['count'] - 1 for group in duplicate_groups)
    
    return {
        'total_files': total,
        'unique_speakers': unique_speakers,
        'duplicate_groups': duplicate_groups,
        'all_comparisons': sorted(all_comparisons, key=lambda x: x['similarity'], reverse=True),
        'method': 'Pyannote',
        'threshold': threshold,
        'errors': [e for e in embeddings if e['error'] is not None]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-sheet', methods=['POST'])
def upload_sheet():
    """Handle CSV/Excel upload and validate"""
    if 'sheet_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['sheet_file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Use CSV or Excel'}), 400
    
    # Read sheet into DataFrame (memory only)
    df = process_sheet_file(file)
    
    if df is None:
        return jsonify({'error': 'Failed to read file'}), 400
    
    # Validate required columns
    columns = df.columns.tolist()
    
    # Check for audio URL column (flexible naming)
    url_column = None
    for col in columns:
        if any(keyword in col.lower() for keyword in ['url', 'link', 'audio', 'file']):
            url_column = col
            break
    
    if url_column is None:
        return jsonify({
            'error': 'No URL column found. Sheet must contain a column with "url", "link", "audio", or "file" in the name',
            'columns': columns
        }), 400
    
    # Check for label column (optional)
    label_column = None
    for col in columns:
        if any(keyword in col.lower() for keyword in ['label', 'name', 'id', 'speaker']):
            label_column = col
            break
    
    # Extract data
    audio_urls = df[url_column].dropna().tolist()
    
    if label_column:
        labels = df[label_column].dropna().tolist()
    else:
        labels = [f"Audio_{i+1}" for i in range(len(audio_urls))]
    
    if len(audio_urls) == 0:
        return jsonify({'error': 'No audio URLs found in sheet'}), 400
    
    return jsonify({
        'message': f'{len(audio_urls)} audio URLs loaded successfully',
        'total_files': len(audio_urls),
        'url_column': url_column,
        'label_column': label_column or 'Auto-generated',
        'sample_urls': audio_urls[:3]  # Show first 3 as preview
    })

@app.route('/analyze-from-sheet', methods=['POST'])
def analyze_from_sheet():
    """Analyze audio from sheet URLs using Pyannote (no disk storage)"""
    if 'sheet_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['sheet_file']
    threshold = float(request.form.get('threshold', 0.30))
    
    # Read sheet
    df = process_sheet_file(file)
    if df is None:
        return jsonify({'error': 'Failed to read file'}), 400
    
    # Find URL column
    url_column = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['url', 'link', 'audio', 'file']):
            url_column = col
            break
    
    if url_column is None:
        return jsonify({'error': 'No URL column found'}), 400
    
    # Find label column
    label_column = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['label', 'name', 'id', 'speaker']):
            label_column = col
            break
    
    # Extract data
    audio_urls = df[url_column].dropna().tolist()
    labels = df[label_column].dropna().tolist() if label_column else [f"Audio_{i+1}" for i in range(len(audio_urls))]
    
    if len(audio_urls) < 2:
        return jsonify({'error': 'At least 2 audio URLs required for comparison'}), 400
    
    # Validate Pyannote API key
    if not PYANNOTE_API_KEY:
        return jsonify({'error': 'Pyannote API key not configured in environment variables'}), 400
    
    try:
        # Process all audio (in-memory only)
        results = find_duplicates_from_urls(audio_urls, labels, threshold)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'pyannote_available': PYANNOTE_API_KEY is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
