from flask import Flask, render_template, request, jsonify
import os
import io
import requests
import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
import librosa
import soundfile as sf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
app.config['ALLOWED_SHEET_EXTENSIONS'] = {'csv', 'xlsx', 'xls'}

# Environment variables
PYANNOTE_API_KEY = os.environ.get('PYANNOTE_API_KEY', None)

# Pyannote.ai official API endpoints
PYANNOTE_EMBEDDING_URL = "https://api.pyannote.ai/v1/embedding"

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_SHEET_EXTENSIONS']

def download_audio_to_memory(url, timeout=30):
    """Download audio from URL to memory"""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
        
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
    """Process audio from memory buffer"""
    try:
        audio_data, sr = sf.read(audio_buffer)
        
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        if sr != target_sr:
            audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=target_sr)
        
        return audio_data
    except Exception as e:
        print(f"Error processing audio: {str(e)}")
        return None

def save_audio_buffer_as_wav(audio_buffer):
    """Convert audio buffer to WAV format in memory"""
    try:
        # Read audio
        audio_data, sr = sf.read(audio_buffer)
        
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Create new WAV buffer
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio_data, sr, format='WAV')
        wav_buffer.seek(0)
        
        return wav_buffer
    except Exception as e:
        print(f"Error converting to WAV: {str(e)}")
        return None

def extract_embedding_pyannote_api(audio_buffer):
    """Extract embedding using official Pyannote.ai API"""
    try:
        if not PYANNOTE_API_KEY:
            raise ValueError("Pyannote API key not configured")
        
        # Convert to WAV format
        wav_buffer = save_audio_buffer_as_wav(audio_buffer)
        if wav_buffer is None:
            return None
        
        # Prepare request
        headers = {
            "Authorization": f"Bearer {PYANNOTE_API_KEY}"
        }
        
        files = {
            'audio': ('audio.wav', wav_buffer, 'audio/wav')
        }
        
        # Call Pyannote API
        response = requests.post(
            PYANNOTE_EMBEDDING_URL,
            headers=headers,
            files=files,
            timeout=60
        )
        
        wav_buffer.close()
        
        if response.status_code == 200:
            result = response.json()
            
            # Extract embedding from response
            # Pyannote API returns embeddings in their response format
            if 'embedding' in result:
                embedding = np.array(result['embedding'])
            elif 'embeddings' in result:
                # If multiple embeddings, average them
                embeddings = np.array(result['embeddings'])
                embedding = np.mean(embeddings, axis=0)
            else:
                # Fallback: try to parse the entire response as embedding
                embedding = np.array(result)
            
            return embedding
        else:
            print(f"Pyannote API Error: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"Error calling Pyannote API: {str(e)}")
        return None

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity"""
    # Flatten embeddings if multi-dimensional
    if len(embedding1.shape) > 1:
        embedding1 = np.mean(embedding1, axis=0)
    if len(embedding2.shape) > 1:
        embedding2 = np.mean(embedding2, axis=0)
    
    # Ensure 1D arrays
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    
    return 1 - cosine(embedding1, embedding2)

def process_sheet_file(file):
    """Read CSV or Excel file"""
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
    """Find duplicate speakers from audio URLs using Pyannote API"""
    embeddings = []
    total = len(audio_urls)
    
    # Extract embeddings
    for idx, (url, label) in enumerate(zip(audio_urls, labels)):
        print(f"Processing {idx+1}/{total}: {label}")
        
        audio_buffer = download_audio_to_memory(url)
        if audio_buffer is None:
            embeddings.append({
                'label': label,
                'url': url,
                'embedding': None,
                'error': 'Download failed'
            })
            continue
        
        embedding = extract_embedding_pyannote_api(audio_buffer)
        
        audio_buffer.close()
        del audio_buffer
        
        embeddings.append({
            'label': label,
            'url': url,
            'embedding': embedding,
            'error': None if embedding is not None else 'API processing failed'
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
    
    unique_speakers = total - sum(group['count'] - 1 for group in duplicate_groups)
    
    return {
        'total_files': total,
        'unique_speakers': unique_speakers,
        'duplicate_groups': duplicate_groups,
        'all_comparisons': sorted(all_comparisons, key=lambda x: x['similarity'], reverse=True),
        'method': 'Pyannote.ai API',
        'threshold': threshold,
        'errors': [e for e in embeddings if e['error'] is not None]
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload-sheet', methods=['POST'])
def upload_sheet():
    """Handle CSV/Excel upload"""
    if 'sheet_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['sheet_file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format'}), 400
    
    df = process_sheet_file(file)
    
    if df is None:
        return jsonify({'error': 'Failed to read file'}), 400
    
    columns = df.columns.tolist()
    
    url_column = None
    for col in columns:
        if any(keyword in col.lower() for keyword in ['url', 'link', 'audio', 'file']):
            url_column = col
            break
    
    if url_column is None:
        return jsonify({
            'error': 'No URL column found',
            'columns': columns
        }), 400
    
    label_column = None
    for col in columns:
        if any(keyword in col.lower() for keyword in ['label', 'name', 'id', 'speaker']):
            label_column = col
            break
    
    audio_urls = df[url_column].dropna().tolist()
    
    if label_column:
        labels = df[label_column].dropna().tolist()
    else:
        labels = [f"Audio_{i+1}" for i in range(len(audio_urls))]
    
    if len(audio_urls) == 0:
        return jsonify({'error': 'No audio URLs found'}), 400
    
    return jsonify({
        'message': f'{len(audio_urls)} audio URLs loaded successfully',
        'total_files': len(audio_urls),
        'url_column': url_column,
        'label_column': label_column or 'Auto-generated',
        'sample_urls': audio_urls[:3]
    })

@app.route('/analyze-from-sheet', methods=['POST'])
def analyze_from_sheet():
    """Analyze audio from sheet URLs using Pyannote API"""
    if 'sheet_file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['sheet_file']
    threshold = float(request.form.get('threshold', 0.30))
    
    df = process_sheet_file(file)
    if df is None:
        return jsonify({'error': 'Failed to read file'}), 400
    
    url_column = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['url', 'link', 'audio', 'file']):
            url_column = col
            break
    
    if url_column is None:
        return jsonify({'error': 'No URL column found'}), 400
    
    label_column = None
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['label', 'name', 'id', 'speaker']):
            label_column = col
            break
    
    audio_urls = df[url_column].dropna().tolist()
    labels = df[label_column].dropna().tolist() if label_column else [f"Audio_{i+1}" for i in range(len(audio_urls))]
    
    if len(audio_urls) < 2:
        return jsonify({'error': 'At least 2 audio URLs required'}), 400
    
    if not PYANNOTE_API_KEY:
        return jsonify({'error': 'Pyannote API key not configured'}), 400
    
    try:
        results = find_duplicates_from_urls(audio_urls, labels, threshold)
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'pyannote_api_available': PYANNOTE_API_KEY is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
