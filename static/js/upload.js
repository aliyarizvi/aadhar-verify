// File Upload Handling
document.getElementById('zipUpload').addEventListener('click', function() {
    document.getElementById('zipFileInput').click();
});

document.getElementById('excelUpload').addEventListener('click', function() {
    document.getElementById('excelFileInput').click();
});

document.getElementById('zipFileInput').addEventListener('change', function() {
    const fileName = this.files[0] ? this.files[0].name : 'No file selected';
    document.getElementById('zipFileName').textContent = fileName;
    checkEnableVerifyButton();
});

document.getElementById('excelFileInput').addEventListener('change', function() {
    const fileName = this.files[0] ? this.files[0].name : 'No file selected';
    document.getElementById('excelFileName').textContent = fileName;
    checkEnableVerifyButton();
});

function checkEnableVerifyButton() {
    const zipFile = document.getElementById('zipFileInput').files[0];
    const excelFile = document.getElementById('excelFileInput').files[0];
    
    document.getElementById('verifyBtn').disabled = !(zipFile && excelFile);
}

// Verification Process Handling
document.getElementById('verifyBtn').addEventListener('click', function() {
    // Show progress section
    document.getElementById('verificationProgress').style.display = 'block';
    
    // Hide results section initially
    document.getElementById('resultsSection').style.display = 'none';
    
    // Reset progress
    document.getElementById('progressBar').style.width = '0%';
    document.getElementById('step1Status').innerHTML = '<span class="badge bg-secondary">Pending</span>';
    document.getElementById('step2Status').innerHTML = '<span class="badge bg-secondary">Pending</span>';
    document.getElementById('step3Status').innerHTML = '<span class="badge bg-secondary">Pending</span>';
    document.getElementById('step4Status').innerHTML = '<span class="badge bg-secondary">Pending</span>';
    
    // Start the upload process
    uploadFiles();
});

function uploadFiles() {
    const zipFile = document.getElementById('zipFileInput').files[0];
    const excelFile = document.getElementById('excelFileInput').files[0];
    
    if (!zipFile || !excelFile) {
        alert('Please select both ZIP and Excel files.');
        return;
    }
    
    const formData = new FormData();
    formData.append('zip_file', zipFile);
    formData.append('excel_file', excelFile);
    
    // Update progress UI - Step 1: Uploading Files
    document.getElementById('progressBar').style.width = '25%';
    document.getElementById('step1Status').innerHTML = '<span class="badge bg-warning">In Progress</span>';
    
    fetch('/api/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('File upload failed');
        }
        return response.json();
    })
    .then(data => {
        // Update progress UI - Step 1: Completed
        document.getElementById('step1Status').innerHTML = '<span class="badge bg-success">Completed</span>';
        
        // Start processing the files
        processFiles(data);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('step1Status').innerHTML = '<span class="badge bg-danger">Failed</span>';
        alert('Upload failed: ' + error.message);
    });
}

function processFiles(uploadData) {
    // Update progress UI - Step 2: Processing Documents
    document.getElementById('progressBar').style.width = '50%';
    document.getElementById('step2Status').innerHTML = '<span class="badge bg-warning">In Progress</span>';
    
    fetch('/api/process', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            zip_path: uploadData.zip_path,
            excel_path: uploadData.excel_path
        })
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Processing failed');
        }
        return response.json();
    })
    .then(data => {
        // Update progress UI - Step 2 & 3: Completed
        document.getElementById('progressBar').style.width = '75%';
        document.getElementById('step2Status').innerHTML = '<span class="badge bg-success">Completed</span>';
        document.getElementById('step3Status').innerHTML = '<span class="badge bg-success">Completed</span>';
        
        // Generate Report
        generateReport(data, uploadData.batch_id);
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('step2Status').innerHTML = '<span class="badge bg-danger">Failed</span>';
        alert('Processing failed: ' + error.message);
    });
}

function generateReport(processData, batchId) {
    // Update progress UI - Step 4: Generating Report
    document.getElementById('progressBar').style.width = '100%';
    document.getElementById('step4Status').innerHTML = '<span class="badge bg-success">Completed</span>';
    
    // Show results section
    document.getElementById('resultsSection').style.display = 'flex';
    
    // Process and display results
    displayResults(processData);
}

function displayResults(data) {
    // Update summary counts
    const verifiedCount = data.results.filter(item => item.is_aadhaar && item.overall_score >= 80).length;
    const suspiciousCount = data.results.filter(item => item.is_aadhaar && item.overall_score >= 50 && item.overall_score < 80).length;
    const fraudulentCount = data.results.filter(item => item.is_aadhaar && item.overall_score < 50).length;
    
    document.getElementById('totalCards').textContent = data.total_count;
    document.getElementById('verifiedCards').textContent = verifiedCount;
    document.getElementById('suspiciousCards').textContent = suspiciousCount;
    document.getElementById('fraudulentCards').textContent = fraudulentCount;
    
    // Populate table with results
    const tableBody = document.getElementById('resultsTable');
    tableBody.innerHTML = '';
    
    if (data.results.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="4" class="text-center">No data available</td></tr>';
        return;
    }
    
    data.results.forEach(item => {
        if (!item.is_aadhaar) {
            return; // Skip non-Aadhaar cards in the table
        }
        
        let status, statusClass;
        if (item.overall_score >= 80) {
            status = 'Verified';
            statusClass = 'success';
        } else if (item.overall_score >= 50) {
            status = 'Suspicious';
            statusClass = 'warning';
        } else {
            status = 'Fraudulent';
            statusClass = 'danger';
        }
        
        // Mask UID for privacy (show last 4 digits)
        const maskedUID = item.uid ? `XXXX XXXX ${item.uid.replace(/\s/g, '').slice(-4)}` : 'N/A';
        
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${maskedUID}</td>
            <td>${item.name || 'N/A'}</td>
            <td><span class="badge bg-${statusClass}">${status}</span></td>
            <td>${item.overall_score.toFixed(1)}%</td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Setup pagination (if needed for large result sets)
    setupPagination(data.results.filter(item => item.is_aadhaar).length);
    
    // Setup download report button
    document.getElementById('downloadReport').addEventListener('click', function(e) {
        e.preventDefault();
        downloadReport(data);
    });
}

function setupPagination(totalItems) {
    const itemsPerPage = 10;
    const totalPages = Math.ceil(totalItems / itemsPerPage);
    
    const pagination = document.getElementById('pagination');
    pagination.innerHTML = '';
    
    // Only show pagination if we have more than one page
    if (totalPages <= 1) {
        pagination.style.display = 'none';
        return;
    }
    
    pagination.style.display = 'flex';
    
    // Previous button
    const prevLi = document.createElement('li');
    prevLi.className = 'page-item disabled';
    prevLi.innerHTML = '<a class="page-link" href="#">Previous</a>';
    pagination.appendChild(prevLi);
    
    // Page numbers
    for (let i = 1; i <= totalPages; i++) {
        const pageLi = document.createElement('li');
        pageLi.className = i === 1 ? 'page-item active' : 'page-item';
        pageLi.innerHTML = `<a class="page-link" href="#">${i}</a>`;
        pagination.appendChild(pageLi);
    }
    
    // Next button
    const nextLi = document.createElement('li');
    nextLi.className = 'page-item';
    nextLi.innerHTML = '<a class="page-link" href="#">Next</a>';
    pagination.appendChild(nextLi);
}

function downloadReport(data) {
    // Create a CSV report
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Aadhar Number,Name,Status,Overall Score,Name Score,Address Score,UID Score\n";
    
    data.results.forEach(item => {
        if (!item.is_aadhaar) return; // Skip non-Aadhaar cards
        
        let status;
        if (item.overall_score >= 80) {
            status = 'Verified';
        } else if (item.overall_score >= 50) {
            status = 'Suspicious';
        } else {
            status = 'Fraudulent';
        }
        
        csvContent += `${item.uid || 'N/A'},${item.name || 'N/A'},${status},${item.overall_score.toFixed(1)},`;
        csvContent += `${item.name_score.toFixed(1)},${item.address_score.toFixed(1)},${item.uid_score.toFixed(1)}\n`;
    });
    
    // Create download link
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "aadhar_verification_report.csv");
    document.body.appendChild(link);
    
    // Trigger download
    link.click();
    
    // Clean up
    document.body.removeChild(link);
}