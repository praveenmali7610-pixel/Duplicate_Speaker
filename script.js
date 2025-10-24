let sheetData = null;

document.getElementById('sheetInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    
    if (!file) return;
    
    const formData = new FormData();
    formData.append('sheet_file', file);
    
    try {
        const response = await fetch('/upload-sheet', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            sheetData = file;
            displaySheetInfo(data);
            document.getElementById('analyzeBtn').disabled = false;
        } else {
            alert('Error: ' + data.error);
            if (data.columns) {
                alert('Available columns: ' + data.columns.join(', '));
            }
        }
    } catch (error) {
        alert('Upload failed: ' + error.message);
    }
});

document.getElementById('threshold').addEventListener('input', (e) => {
    document.getElementById('thresholdValue').textContent = e.target.value;
});

document.getElementById('analyzeBtn').addEventListener('click', analyzeSheet);

function displaySheetInfo(data) {
    const infoDiv = document.getElementById('sheetInfo');
    infoDiv.innerHTML = `
        <h3>✅ Sheet Loaded Successfully</h3>
        <p><strong>Total Audio Files:</strong> ${data.total_files}</p>
        <p><strong>URL Column:</strong> ${data.url_column}</p>
        <p><strong>Label Column:</strong> ${data.label_column}</p>
        <div style="margin-top: 10px;">
            <strong>Sample URLs:</strong>
            <ul>
                ${data.sample_urls.map(url => `<li>${url}</li>`).join('')}
            </ul>
        </div>
    `;
}

async function analyzeSheet() {
    if (!sheetData) {
        alert('Please upload a CSV/Excel file first');
        return;
    }
    
    const method = document.querySelector('input[name="method"]:checked').value;
    const threshold = parseFloat(document.getElementById('threshold').value);
    
    const formData = new FormData();
    formData.append('sheet_file', sheetData);
    formData.append('method', method);
    formData.append('threshold', threshold);
    
    document.getElementById('loading').style.display = 'block';
    document.getElementById('results').innerHTML = '';
    document.getElementById('analyzeBtn').disabled = true;
    
    try {
        const response = await fetch('/analyze-from-sheet', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            displayResults(data);
        } else {
            alert('Analysis failed: ' + data.error);
        }
    } catch (error) {
        alert('Analysis error: ' + error.message);
    } finally {
        document.getElementById('loading').style.display = 'none';
        document.getElementById('analyzeBtn').disabled = false;
    }
}

function displayResults(data) {
    const resultsDiv = document.getElementById('results');
    
    let html = `
        <div class="result-card">
            <h2>📊 Analysis Results</h2>
            <p><strong>Detection Method:</strong> ${data.method}</p>
            <p><strong>Total Files Analyzed:</strong> ${data.total_files}</p>
            <p><strong>Unique Speakers:</strong> ${data.unique_speakers}</p>
            <p><strong>Duplicate Groups Found:</strong> ${data.duplicate_groups.length}</p>
            <p><strong>Threshold Used:</strong> ${data.threshold}</p>
            <p style="color: #48bb78; font-weight: 600;">✅ All processing done in-memory - no files stored on server</p>
        </div>
    `;
    
    if (data.duplicate_groups.length > 0) {
        html += '<div class="result-card"><h3>🔴 Duplicate Speaker Groups</h3>';
        
        data.duplicate_groups.forEach(group => {
            html += `
                <div class="duplicate-group">
                    <h4>${group.speaker_id} - ${group.count} recordings detected</h4>
                    <ul>
                        ${group.files.map(file => `<li>${file}</li>`).join('')}
                    </ul>
                </div>
            `;
        });
        
        html += '</div>';
    } else {
        html += `
            <div class="result-card">
                <h3>✅ No Duplicates Found</h3>
                <p>All audio samples appear to be from different speakers.</p>
            </div>
        `;
    }
    
    if (data.all_comparisons.length > 0) {
        html += '<div class="result-card"><h3>📈 Top Similarity Scores</h3>';
        
        // Show top 10 comparisons
        data.all_comparisons.slice(0, 10).forEach(comp => {
            const similarityPercent = (comp.similarity * 100).toFixed(1);
            const badgeClass = comp.is_duplicate ? 'similarity-high' : 'similarity-medium';
            
            html += `
                <div class="comparison-item">
                    <strong>${comp.file1}</strong> ↔️ <strong>${comp.file2}</strong>
                    <span class="similarity-badge ${badgeClass}">
                        ${similarityPercent}% ${comp.is_duplicate ? '(Duplicate)' : ''}
                    </span>
                </div>
            `;
        });
        
        html += '</div>';
    }
    
    if (data.errors && data.errors.length > 0) {
        html += `
            <div class="error-section">
                <h3>⚠️ Processing Errors (${data.errors.length})</h3>
                <ul>
                    ${data.errors.map(err => `<li>${err.label}: ${err.error}</li>`).join('')}
                </ul>
            </div>
        `;
    }
    
    resultsDiv.innerHTML = html;
}
