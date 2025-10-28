// ========================================
// Configuration API
// ========================================
const API_BASE_URL = 'http://localhost:8000';
const API_VERSION = '/api/v1';

// ========================================
// Page: Index - Image Upload & Detection
// ========================================

// Handle file input change
if (document.getElementById('imageInput')) {
    document.getElementById('imageInput').addEventListener('change', function(e) {
        const file = e.target.files[0];
        if (file) {
            displayImagePreview(file);
        }
    });
}

// Display image preview
function displayImagePreview(file) {
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById('uploadBox').style.display = 'none';
        document.getElementById('imagePreview').style.display = 'block';
        document.getElementById('previewImg').src = e.target.result;
    };
    reader.readAsDataURL(file);
}

// Reset upload section
function resetUpload() {
    document.getElementById('imageInput').value = '';
    document.getElementById('uploadBox').style.display = 'block';
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';
}

// Analyze image
async function analyzeImage() {
    const fileInput = document.getElementById('imageInput');
    const file = fileInput.files[0];
    
    if (!file) {
        showError('Veuillez sélectionner une image');
        return;
    }

    // Show loading
    document.getElementById('imagePreview').style.display = 'none';
    document.getElementById('loadingSection').style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorSection').style.display = 'none';

    try {
        const formData = new FormData();
        formData.append('file', file);

        const response = await fetch(`${API_BASE_URL}${API_VERSION}/detect-disease`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }

        const data = await response.json();
        displayResults(data);
    } catch (error) {
        console.error('Error:', error);
        showError('Erreur lors de l\'analyse de l\'image. Veuillez réessayer.');
    } finally {
        document.getElementById('loadingSection').style.display = 'none';
    }
}

// Display detection results
function displayResults(data) {
    document.getElementById('diseaseName').textContent = data.disease_name || 'Non identifiée';
    document.getElementById('confidence').textContent = `${(data.confidence * 100).toFixed(1)}%`;
    document.getElementById('severity').textContent = data.severity || 'Non déterminée';
    document.getElementById('affectedCrop').textContent = data.affected_crop || 'Non spécifié';

    // Color code confidence
    const confidenceElement = document.getElementById('confidence');
    if (data.confidence >= 0.8) {
        confidenceElement.style.color = 'var(--success-color)';
    } else if (data.confidence >= 0.6) {
        confidenceElement.style.color = 'var(--warning-color)';
    } else {
        confidenceElement.style.color = 'var(--danger-color)';
    }

    // Display treatments
    const treatmentsContainer = document.getElementById('treatments');
    treatmentsContainer.innerHTML = '';
    
    if (data.treatments && data.treatments.length > 0) {
        data.treatments.forEach(treatment => {
            const treatmentDiv = document.createElement('div');
            treatmentDiv.className = 'treatment-item';
            treatmentDiv.innerHTML = `
                <h4>${treatment.name}</h4>
                <p><strong>Description:</strong> ${treatment.description}</p>
                ${treatment.organic ? '<span style="color: var(--success-color);">✓ Traitement biologique</span>' : ''}
            `;
            treatmentsContainer.appendChild(treatmentDiv);
        });
    } else {
        treatmentsContainer.innerHTML = '<p>Aucun traitement spécifique disponible.</p>';
    }

    // Display prevention tips
    const preventionList = document.getElementById('preventionTips');
    preventionList.innerHTML = '';
    
    if (data.prevention_tips && data.prevention_tips.length > 0) {
        data.prevention_tips.forEach(tip => {
            const li = document.createElement('li');
            li.textContent = tip;
            preventionList.appendChild(li);
        });
    } else {
        preventionList.innerHTML = '<li>Aucun conseil de prévention disponible.</li>';
    }

    document.getElementById('resultsSection').style.display = 'block';
}

// Show error message
function showError(message) {
    document.getElementById('errorMessage').textContent = message;
    document.getElementById('errorSection').style.display = 'block';
    document.getElementById('loadingSection').style.display = 'none';
    document.getElementById('imagePreview').style.display = 'none';
}

// ========================================
// Page: Chat - Bot Interaction
// ========================================

// Send message when Enter is pressed
function handleKeyPress(event) {
    if (event.key === 'Enter') {
        sendMessage();
    }
}

// Send user message
async function sendMessage() {
    const input = document.getElementById('chatInput');
    const message = input.value.trim();
    
    if (!message) return;

    // Display user message
    displayMessage(message, 'user');
    input.value = '';

    try {
        const response = await fetch(`${API_BASE_URL}${API_VERSION}/chat`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                session_id: sessionId
            })
        });

        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }

        const data = await response.json();
        
        // Display bot response
        setTimeout(() => {
            displayMessage(data.response, 'bot');
            
            // Display suggestions if available
            if (data.suggestions && data.suggestions.length > 0) {
                updateSuggestions(data.suggestions);
            }
        }, 500);
    } catch (error) {
        console.error('Error:', error);
        displayMessage('Désolé, une erreur s\'est produite. Veuillez réessayer.', 'bot');
    }
}

// Send suggestion
function sendSuggestion(text) {
    document.getElementById('chatInput').value = text;
    sendMessage();
}

// Display message in chat
function displayMessage(text, type) {
    const messagesContainer = document.getElementById('chatMessages');
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${type}-message`;
    
    const avatar = type === 'user' ? '👤' : '🤖';
    const time = new Date().toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' });
    
    messageDiv.innerHTML = `
        <div class="message-avatar">${avatar}</div>
        <div class="message-content">
            <p>${text}</p>
            <span class="message-time">${time}</span>
        </div>
    `;
    
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Update suggestions
function updateSuggestions(suggestions) {
    const suggestionsList = document.querySelector('.suggestions-list');
    suggestionsList.innerHTML = '';
    
    suggestions.forEach(suggestion => {
        const button = document.createElement('button');
        button.className = 'suggestion-btn';
        button.textContent = suggestion;
        button.onclick = () => sendSuggestion(suggestion);
        suggestionsList.appendChild(button);
    });
}

// ========================================
// Page: Dashboard - Statistics
// ========================================

// Load dashboard statistics
async function loadDashboardStats() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_VERSION}/statistics/dashboard`);
        
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        
        const data = await response.json();
        
        // Update statistics
        document.getElementById('totalDetections').textContent = data.total_detections || '0';
        document.getElementById('activeUsers').textContent = data.active_users || '0';
        document.getElementById('diseaseTypes').textContent = Object.keys(data.diseases_detected || {}).length || '0';
        document.getElementById('successRate').textContent = `${((data.success_rate || 0) * 100).toFixed(1)}%`;
        
        // Display top diseases chart
        displayTopDiseases(data.diseases_detected || {});
    } catch (error) {
        console.error('Error loading dashboard stats:', error);
        // Show default values on error
        document.getElementById('totalDetections').textContent = '0';
        document.getElementById('activeUsers').textContent = '0';
        document.getElementById('diseaseTypes').textContent = '0';
        document.getElementById('successRate').textContent = '0%';
    }
}

// Display top diseases chart
function displayTopDiseases(diseases) {
    const container = document.getElementById('topDiseases');
    container.innerHTML = '';
    
    if (Object.keys(diseases).length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-light);">Aucune donnée disponible</p>';
        return;
    }
    
    // Sort diseases by count
    const sortedDiseases = Object.entries(diseases).sort((a, b) => b[1] - a[1]);
    const maxCount = sortedDiseases[0][1];
    
    // Display top 5 diseases
    sortedDiseases.slice(0, 5).forEach(([name, count]) => {
        const percentage = (count / maxCount) * 100;
        
        const chartBar = document.createElement('div');
        chartBar.className = 'chart-bar';
        chartBar.innerHTML = `
            <div class="chart-label">
                <span>${name}</span>
                <span>${count} détections</span>
            </div>
            <div class="progress-bar">
                <div class="progress-fill" style="width: ${percentage}%"></div>
            </div>
        `;
        
        container.appendChild(chartBar);
    });
}

// Load common diseases
async function loadCommonDiseases() {
    try {
        const response = await fetch(`${API_BASE_URL}${API_VERSION}/diseases/common`);
        
        if (!response.ok) {
            throw new Error(`Erreur HTTP: ${response.status}`);
        }
        
        const data = await response.json();
        displayCommonDiseases(data.diseases || []);
    } catch (error) {
        console.error('Error loading common diseases:', error);
        const container = document.getElementById('diseasesList');
        container.innerHTML = '<p style="text-align: center; color: var(--text-light);">Erreur de chargement des données</p>';
    }
}

// Display common diseases
function displayCommonDiseases(diseases) {
    const container = document.getElementById('diseasesList');
    container.innerHTML = '';
    
    if (diseases.length === 0) {
        container.innerHTML = '<p style="text-align: center; color: var(--text-light);">Aucune maladie disponible</p>';
        return;
    }
    
    diseases.forEach(disease => {
        const diseaseDiv = document.createElement('div');
        diseaseDiv.className = 'disease-item';
        diseaseDiv.innerHTML = `
            <div>
                <div class="disease-name">${disease.name}</div>
                <small style="color: var(--text-light);">${disease.crop || 'Toutes cultures'} - ${disease.season || 'Toute saison'}</small>
            </div>
            <div class="disease-count">${disease.frequency || 'Rare'}</div>
        `;
        container.appendChild(diseaseDiv);
    });
}

// ========================================
// Drag & Drop for Image Upload
// ========================================
if (document.getElementById('uploadBox')) {
    const uploadBox = document.getElementById('uploadBox');
    
    uploadBox.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--primary-color)';
        uploadBox.style.backgroundColor = '#f0fff4';
    });
    
    uploadBox.addEventListener('dragleave', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--border-color)';
        uploadBox.style.backgroundColor = 'white';
    });
    
    uploadBox.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadBox.style.borderColor = 'var(--border-color)';
        uploadBox.style.backgroundColor = 'white';
        
        const file = e.dataTransfer.files[0];
        if (file && file.type.startsWith('image/')) {
            document.getElementById('imageInput').files = e.dataTransfer.files;
            displayImagePreview(file);
        }
    });
}

// ========================================
// Console Message
// ========================================
console.log('%c🌾 AgriDetect v1.0.0', 'color: #2ecc71; font-size: 20px; font-weight: bold;');
console.log('%cAPI Base URL:', 'color: #3498db; font-weight: bold;', API_BASE_URL);
console.log('%cProjet de Fin d\'Étude - 2025', 'color: #95a5a6;');
