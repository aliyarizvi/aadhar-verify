// Load latest results when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadLatestBatchResults();

    // Set up event listener for export button
    const exportButton = document.getElementById('exportResults');
    if (exportButton) {
        exportButton.addEventListener('click', exportResults);
    }
});

// Load the latest batch results
function loadLatestBatchResults() {
    fetch('/api/results')
        .then(response => {
            if (!response.ok) {
                // If no recent batch is found, just return without error
                if (response.status === 404) {
                    document.getElementById('noBatchAlert').style.display = 'block';
                    document.getElementById('resultsContainer').style.display = 'none';
                    return null;
                }
                throw new Error('Failed to fetch latest results');
            }
            return response.json();
        })
        .then(data => {
            if (data) {
                document.getElementById('noBatchAlert').style.display = 'none';
                document.getElementById('resultsContainer').style.display = 'block';
                
                // No batch ID display on frontend
                
                // Display results
                displayAnalytics(data);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            document.getElementById('noBatchAlert').style.display = 'block';
            document.getElementById('resultsContainer').style.display = 'none';
        });
}

// Main function to display analytics across all components
function displayAnalytics(data) {
    // Update summary numbers
    updateSummaryNumbers(data);
    
    // Update charts
    updateCharts(data);
    
    // Update tables
    updateTables(data);
}

// Update the summary statistics cards
function updateSummaryNumbers(data) {
    // Calculate summary statistics
    const totalCards = data.results.length;
    const aadhaarCards = data.results.filter(item => item.is_aadhaar).length;
    const nonAadhaarCards = data.results.filter(item => !item.is_aadhaar).length;
    
    // For Aadhaar cards only
    const aadhaarResults = data.results.filter(item => item.is_aadhaar);
    const verifiedCount = aadhaarResults.filter(item => item.overall_score >= 80).length;
    const suspiciousCount = aadhaarResults.filter(item => item.overall_score >= 50 && item.overall_score < 80).length;
    const fraudulentCount = aadhaarResults.filter(item => item.overall_score < 50).length;
    
    // Calculate percentages
    const verifiedPercent = aadhaarCards > 0 ? (verifiedCount / aadhaarCards * 100).toFixed(1) : 0;
    const suspiciousPercent = aadhaarCards > 0 ? (suspiciousCount / aadhaarCards * 100).toFixed(1) : 0;
    const fraudulentPercent = aadhaarCards > 0 ? (fraudulentCount / aadhaarCards * 100).toFixed(1) : 0;
    
    // Update the DOM elements
    document.getElementById('totalCardsCount').textContent = totalCards;
    document.getElementById('verifiedCount').textContent = verifiedCount;
    document.getElementById('verifiedPercent').textContent = `(${verifiedPercent}%)`;
    document.getElementById('suspiciousCount').textContent = suspiciousCount;
    document.getElementById('suspiciousPercent').textContent = `(${suspiciousPercent}%)`;
    document.getElementById('fraudulentCount').textContent = fraudulentCount;
    document.getElementById('fraudulentPercent').textContent = `(${fraudulentPercent}%)`;
    document.getElementById('nonAadhaarCount').textContent = nonAadhaarCards;
}

// Update all chart visualizations
function updateCharts(data) {
    // Prepare data
    const aadhaarResults = data.results.filter(item => item.is_aadhaar);
    const verifiedCount = aadhaarResults.filter(item => item.overall_score >= 80).length;
    const suspiciousCount = aadhaarResults.filter(item => item.overall_score >= 50 && item.overall_score < 80).length;
    const fraudulentCount = aadhaarResults.filter(item => item.overall_score < 50).length;
    
    // Verification Status Chart (Bar Chart)
    const statusChartCanvas = document.getElementById('statusChart');
    if (statusChartCanvas) {
        const ctx = statusChartCanvas.getContext('2d');
        ctx.clearRect(0, 0, statusChartCanvas.width, statusChartCanvas.height);
        
        // Chart dimensions
        const chartWidth = statusChartCanvas.width;
        const chartHeight = statusChartCanvas.height;
        const barWidth = chartWidth / 4; // 4 bars with padding
        const maxCount = Math.max(verifiedCount, suspiciousCount, fraudulentCount, 1); // Avoid division by zero
        
        // Calculate bar height based on counts
        const getBarHeight = (count) => (count / maxCount) * (chartHeight - 60);
        
        // Draw bars
        // Verified bar (green)
        ctx.fillStyle = '#10B981';
        const verifiedHeight = getBarHeight(verifiedCount);
        ctx.fillRect(barWidth * 1 - 20, chartHeight - verifiedHeight - 30, barWidth - 20, verifiedHeight);
        
        // Suspicious bar (yellow)
        ctx.fillStyle = '#F59E0B';
        const suspiciousHeight = getBarHeight(suspiciousCount);
        ctx.fillRect(barWidth * 2 - 20, chartHeight - suspiciousHeight - 30, barWidth - 20, suspiciousHeight);
        
        // Fraudulent bar (red)
        ctx.fillStyle = '#EF4444';
        const fraudulentHeight = getBarHeight(fraudulentCount);
        ctx.fillRect(barWidth * 3 - 20, chartHeight - fraudulentHeight - 30, barWidth - 20, fraudulentHeight);
        
        // Add labels and values
        ctx.fillStyle = '#000000';
        ctx.font = '12px Arial';
        ctx.textAlign = 'center';
        
        // Verified label and value
        ctx.fillText('Verified', barWidth * 1 - 20 + (barWidth - 20) / 2, chartHeight - 10);
        ctx.fillText(verifiedCount, barWidth * 1 - 20 + (barWidth - 20) / 2, chartHeight - verifiedHeight - 40);
        
        // Suspicious label and value
        ctx.fillText('Suspicious', barWidth * 2 - 20 + (barWidth - 20) / 2, chartHeight - 10);
        ctx.fillText(suspiciousCount, barWidth * 2 - 20 + (barWidth - 20) / 2, chartHeight - suspiciousHeight - 40);
        
        // Fraudulent label and value
        ctx.fillText('Fraudulent', barWidth * 3 - 20 + (barWidth - 20) / 2, chartHeight - 10);
        ctx.fillText(fraudulentCount, barWidth * 3 - 20 + (barWidth - 20) / 2, chartHeight - fraudulentHeight - 40);
    }

    // Score Distribution Chart (Bar Chart)
    const scoreDistributionCanvas = document.getElementById('scoreDistributionChart');
    if (scoreDistributionCanvas && aadhaarResults.length > 0) {
        const ctx = scoreDistributionCanvas.getContext('2d');
        ctx.clearRect(0, 0, scoreDistributionCanvas.width, scoreDistributionCanvas.height);
        
        const scoreBins = Array(11).fill(0); // Extra bin to round nearest 10
        aadhaarResults.forEach(item => {
            const binIndex = Math.round(item.overall_score / 10); // Round to nearest 10
            scoreBins[binIndex]++;
        });
        
        const chartWidth = scoreDistributionCanvas.width;
        const chartHeight = scoreDistributionCanvas.height;
        
        // Reduced margins to make chart bigger
        const margin = { top: 30, right: 15, bottom: 50, left: 40 };
        const plotWidth = chartWidth - margin.left - margin.right;
        const plotHeight = chartHeight - margin.top - margin.bottom;
        
        // Adjusted bar width
        const barWidth = plotWidth / 12;
        const maxCount = Math.max(...scoreBins, 1);
        const getBarHeight = (count) => (count / maxCount) * plotHeight; // Removed the -20 to give more height
        
        // Draw axes
        ctx.beginPath();
        ctx.moveTo(margin.left, margin.top);
        ctx.lineTo(margin.left, chartHeight - margin.bottom);
        ctx.lineTo(chartWidth - margin.right, chartHeight - margin.bottom);
        ctx.strokeStyle = '#666';
        ctx.stroke();
        
        // Y-axis labels
        ctx.fillStyle = '#000';
        ctx.font = '12px Arial';
        ctx.textAlign = 'right';
        
        for (let i = 0; i <= maxCount; i++) { // Show all values up to max count
            if (i > 3) break; // Only show up to 3 for readability
            const y = chartHeight - margin.bottom - ((i / maxCount) * plotHeight);
            ctx.fillText(i.toString(), margin.left - 5, y + 3);
            
            // Grid lines
            ctx.beginPath();
            ctx.moveTo(margin.left, y);
            ctx.lineTo(chartWidth - margin.right, y);
            ctx.strokeStyle = '#ddd';
            ctx.setLineDash([4, 2]);
            ctx.stroke();
            ctx.setLineDash([]);
        }
        
        // X-axis labels
        ctx.textAlign = 'center';
        ctx.font = '12px Arial';
        for (let i = 0; i <= 10; i++) {
            const xPos = margin.left + ((i + 0.5) * barWidth);
            ctx.fillText(`${i * 10}`, xPos, chartHeight - margin.bottom + 15);
        }
        
        // Draw bars
        const colors = ['#FFD700', '#ADFF2F', '#32CD32', '#008000'];
        scoreBins.forEach((count, index) => {
            if (count === 0) return;
            
            ctx.fillStyle = colors[Math.min(Math.floor(index / 3), colors.length - 1)];
            
            const barHeight = getBarHeight(count);
            const xPos = margin.left + (index * barWidth);
            const yPos = chartHeight - margin.bottom - barHeight;
            
            // Wider bars (0.9 instead of 0.8)
            ctx.fillRect(xPos, yPos, barWidth * 0.9, barHeight);
            
            // Add value on top of bar
            if (barHeight > 15) {
                ctx.fillStyle = '#000';
                ctx.fillText(count.toString(), xPos + (barWidth * 0.45), yPos - 5);
            }
        });
        
        // Add axis titles
        ctx.fillStyle = '#000';
        ctx.font = '14px Arial';
        ctx.textAlign = 'center';
        ctx.fillText('Score Ranges (%)', chartWidth / 2, chartHeight - 10);
        
        ctx.save();
        ctx.translate(15, chartHeight / 2);
        ctx.rotate(-Math.PI / 2);
        ctx.fillText('Number of Cards', 0, 0);
        ctx.restore();
    }  
    
    // Document Type Distribution Chart (Pie Chart)
    const documentTypeCanvas = document.getElementById('documentTypeChart');
    if (documentTypeCanvas) {
        const ctx = documentTypeCanvas.getContext('2d');
        ctx.clearRect(0, 0, documentTypeCanvas.width, documentTypeCanvas.height);
        
        const aadhaarCount = data.results.filter(item => item.is_aadhaar).length;
        const nonAadhaarCount = data.results.filter(item => !item.is_aadhaar).length;
        const total = data.results.length;
        
        if (total === 0) {
            // No data to display
            ctx.fillStyle = '#000000';
            ctx.font = '14px Arial';
            ctx.textAlign = 'center';
            ctx.fillText('No data available', documentTypeCanvas.width / 2, documentTypeCanvas.height / 2);
            return;
        }
        
        // Pie chart dimensions
        const centerX = documentTypeCanvas.width / 2;
        const centerY = documentTypeCanvas.height / 2 - 30; // Move chart up further
        const radius = Math.min(centerX, centerY) - 30;
        
        // Calculate angles
        const aadhaarAngle = (aadhaarCount / total) * Math.PI * 2;
        
        // Draw Aadhaar slice
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, 0, aadhaarAngle);
        ctx.fillStyle = '#3F4259'; // Primary color
        ctx.fill();
        
        // Draw non-Aadhaar slice
        ctx.beginPath();
        ctx.moveTo(centerX, centerY);
        ctx.arc(centerX, centerY, radius, aadhaarAngle, Math.PI * 2);
        ctx.fillStyle = '#6C7293'; // Secondary color
        ctx.fill();
        
        // Position the legend at the bottom with more space
        const legendY = centerY + radius + 40;
        
        // Create a proper legend with more space
        // First legend item (Aadhaar)
        ctx.fillStyle = '#3F4259';
        ctx.fillRect(70, legendY, 16, 16);
        ctx.fillStyle = '#000000';
        ctx.font = '14px Arial';
        ctx.textAlign = 'left';
        ctx.fillText(`Aadhaar: ${aadhaarCount} (${Math.round(aadhaarCount / total * 100)}%)`, 95, legendY + 13);
    
        // Second legend item (Non-Aadhaar) - place it below first item
        ctx.fillStyle = '#6C7293';
        ctx.fillRect(70, legendY + 30, 16, 16);
        ctx.fillStyle = '#000000';
        ctx.fillText(`Non-Aadhaar: ${nonAadhaarCount} (${Math.round(nonAadhaarCount / total * 100)}%)`, 95, legendY + 43);
    }
}

// Update all data tables
function updateTables(data) {
    // Update the detailed results table
    const tableBody = document.getElementById('detailedResultsTable');
    if (!tableBody) return;
    
    tableBody.innerHTML = '';
    
    // Only show Aadhar cards in the detailed results
    const aadhaarResults = data.results.filter(item => item.is_aadhaar);
    
    if (aadhaarResults.length === 0) {
        tableBody.innerHTML = '<tr><td colspan="6" class="text-center">No data available</td></tr>';
        return;
    }
    
    // Sort by overall score (descending)
    aadhaarResults.sort((a, b) => b.overall_score - a.overall_score);
    
    aadhaarResults.forEach((item, index) => {
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
            <td>${index + 1}</td>
            <td>${maskedUID}</td>
            <td>${item.name || 'N/A'}</td>
            <td><span class="badge bg-${statusClass}">${status}</span></td>
            <td>
                <div class="progress" style="height: 20px;">
                    <div class="progress-bar bg-${statusClass}" role="progressbar" 
                         style="width: ${item.overall_score}%;" 
                         aria-valuenow="${item.overall_score}" aria-valuemin="0" aria-valuemax="100">
                        ${item.overall_score.toFixed(1)}%
                    </div>
                </div>
            </td>
            <td>
                <button class="btn btn-sm btn-primary view-details" data-id="${item._id}">
                    <i class="fas fa-eye"></i>
                </button>
            </td>
        `;
        
        tableBody.appendChild(row);
    });
    
    // Set up event listeners for view details buttons
    document.querySelectorAll('.view-details').forEach(button => {
        button.addEventListener('click', function() {
            const userId = this.getAttribute('data-id');
            const userDetails = aadhaarResults.find(user => user._id === userId);
            if (userDetails) {
                showUserDetailsModal(userDetails);
            }
        });
    });
    
    // Update the non-Aadhar files table
    const nonAadharTableBody = document.getElementById('nonAadharFilesTable');
    if (!nonAadharTableBody) return;
    
    nonAadharTableBody.innerHTML = '';
    
    const nonAadharFiles = data.results.filter(item => !item.is_aadhaar);
    
    if (nonAadharFiles.length === 0) {
        nonAadharTableBody.innerHTML = '<tr><td colspan="2" class="text-center">No non-Aadhar files found</td></tr>';
        return;
    }
    
    nonAadharFiles.forEach((item, index) => {
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${index + 1}</td>
            <td>${item.filename}</td>
        `;
        
        nonAadharTableBody.appendChild(row);
    });
}

// User details modal function
function showUserDetailsModal(user) {
    // Create modal elements
    const modalId = 'userDetailsModal';
    
    // Remove existing modal if any
    const existingModal = document.getElementById(modalId);
    if (existingModal) {
        existingModal.remove();
    }
    
    // Determine verification status and color
    let status, statusClass;
    if (user.overall_score >= 80) {
        status = 'Verified';
        statusClass = 'success';
    } else if (user.overall_score >= 50) {
        status = 'Suspicious';
        statusClass = 'warning';
    } else {
        status = 'Fraudulent';
        statusClass = 'danger';
    }
    
    // Create modal HTML
    const modalHTML = `
        <div class="modal fade" id="${modalId}" tabindex="-1" aria-labelledby="${modalId}Label" aria-hidden="true">
            <div class="modal-dialog modal-dialog-centered">
                <div class="modal-content">
                    <div class="modal-header bg-${statusClass} text-white">
                        <h5 class="modal-title" id="${modalId}Label">User Details</h5>
                        <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
                    </div>
                    <div class="modal-body">
                        <div class="alert alert-${statusClass} mb-4">
                            <strong>Verification Status:</strong> ${status} (${user.overall_score.toFixed(1)}%)
                        </div>
                        
                        <div class="mb-3">
                            <label class="fw-bold">UID:</label>
                            <p>${user.uid || 'N/A'}</p>
                        </div>
                        
                        <div class="mb-3">
                            <label class="fw-bold">Name:</label>
                            <p>${user.name || 'N/A'}</p>
                        </div>
                        
                        <div class="mb-3">
                            <label class="fw-bold">Address:</label>
                            <p>${user.address || 'N/A'}</p>
                        </div>
                        
                        <div class="mb-3">
                            <label class="fw-bold">Filename:</label>
                            <p>${user.filename || 'N/A'}</p>
                        </div>
                        
                        <hr>
                        
                        <h6 class="mb-3">Match Scores</h6>
                        
                        <div class="row g-3">
                            <div class="col-md-4">
                                <div class="card h-100">
                                    <div class="card-body text-center">
                                        <h5 class="card-title">${user.name_score.toFixed(1)}%</h5>
                                        <p class="card-text small">Name Match</p>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-md-4">
                                <div class="card h-100">
                                    <div class="card-body text-center">
                                        <h5 class="card-title">${user.address_score.toFixed(1)}%</h5>
                                        <p class="card-text small">Address Match</p>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="col-md-4">
                                <div class="card h-100">
                                    <div class="card-body text-center">
                                        <h5 class="card-title">${user.uid_score.toFixed(1)}%</h5>
                                        <p class="card-text small">UID Match</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="modal-footer">
                        <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Add modal to page
    document.body.insertAdjacentHTML('beforeend', modalHTML);
    
    // Show the modal
    const modalElement = document.getElementById(modalId);
    const bsModal = new bootstrap.Modal(modalElement);
    bsModal.show();
}

// Export results function
function exportResults() {
    // Fetch the latest results
    fetch('/api/results')
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch latest results');
            }
            return response.json();
        })
        .then(data => {
            // Create CSV content
            let csvContent = "data:text/csv;charset=utf-8,";
            csvContent += "UID,Name,Address,Status,Overall Score,Name Score,Address Score,UID Score\n";
            
            const aadhaarResults = data.results.filter(item => item.is_aadhaar);
            
            aadhaarResults.forEach(item => {
                let status;
                if (item.overall_score >= 80) {
                    status = 'Verified';
                } else if (item.overall_score >= 50) {
                    status = 'Suspicious';
                } else {
                    status = 'Fraudulent';
                }
                
                // Escape fields that might contain commas
                const escapeCsv = (field) => {
                    if (field === null || field === undefined) return '';
                    return `"${String(field).replace(/"/g, '""')}"`;
                };
                
                csvContent += `${escapeCsv(item.uid)},${escapeCsv(item.name)},${escapeCsv(item.address)},`;
                csvContent += `${escapeCsv(status)},${item.overall_score.toFixed(1)},${item.name_score.toFixed(1)},`;
                csvContent += `${item.address_score.toFixed(1)},${item.uid_score.toFixed(1)}\n`;
            });
            
            // Create download link
            const encodedUri = encodeURI(csvContent);
            const link = document.createElement("a");
            link.setAttribute("href", encodedUri);
            link.setAttribute("download", `aadhar_verification_report_${data.batch_id.substring(0, 8)}.csv`);
            document.body.appendChild(link);
            
            // Trigger download
            link.click();
            
            // Clean up
            document.body.removeChild(link);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Failed to export results: ' + error.message);
        });
}