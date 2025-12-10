document.getElementById('predictionForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    // Get form values
    const formData = {
        primary_offence: document.getElementById('primary_offence').value,
        location_type: document.getElementById('location_type').value,
        neighbourhood: document.getElementById('neighbourhood').value,
        religion_bias: document.getElementById('religion_bias').value
    };
    
    // Hide error and results
    document.getElementById('error').style.display = 'none';
    document.getElementById('results').style.display = 'none';
    
    // Show loading state
    const submitBtn = document.querySelector('.btn-primary');
    const originalText = submitBtn.textContent;
    submitBtn.innerHTML = '<span class="loading"></span> Predicting...';
    submitBtn.disabled = true;
    
    try {
        // Send POST request to Flask backend
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(formData)
        });
        
        if (!response.ok) {
            throw new Error('Prediction failed');
        }
        
        const data = await response.json();
        
        // Display results
        displayResults(data);
        
    } catch (error) {
        // Show error message
        const errorDiv = document.getElementById('error');
        errorDiv.textContent = 'An error occurred while making the prediction. Please try again.';
        errorDiv.style.display = 'block';
        console.error('Error:', error);
    } finally {
        // Reset button
        submitBtn.textContent = originalText;
        submitBtn.disabled = false;
    }
});

function displayResults(data) {
    // Logistic Regression results
    const logRegPred = data.logistic_regression.prediction;
    const logRegProb = data.logistic_regression.probability;
    
    document.getElementById('log-reg-prediction').textContent = logRegPred;
    document.getElementById('log-reg-prediction').className = 'prediction-value ' + (logRegPred === 'Yes' ? 'yes' : 'no');
    document.getElementById('log-reg-probability').textContent = logRegProb.toFixed(1) + '%';
    document.getElementById('log-reg-probability-bar').style.width = logRegProb + '%';
    
    // Decision Tree results
    const treePred = data.decision_tree.prediction;
    const treeProb = data.decision_tree.probability;
    
    document.getElementById('tree-prediction').textContent = treePred;
    document.getElementById('tree-prediction').className = 'prediction-value ' + (treePred === 'Yes' ? 'yes' : 'no');
    document.getElementById('tree-probability').textContent = treeProb.toFixed(1) + '%';
    document.getElementById('tree-probability-bar').style.width = treeProb + '%';
    
    // Show results container with smooth scroll
    const resultsDiv = document.getElementById('results');
    resultsDiv.style.display = 'block';
    resultsDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Add smooth animations when page loads
document.addEventListener('DOMContentLoaded', function() {
    const form = document.querySelector('.card');
    form.style.opacity = '0';
    form.style.transform = 'translateY(20px)';
    
    setTimeout(() => {
        form.style.transition = 'all 0.5s ease';
        form.style.opacity = '1';
        form.style.transform = 'translateY(0)';
    }, 100);
});
