/**
 * TN AgriSense — Main JavaScript
 * Handles camera capture, file upload, drag & drop, and UI interactions
 */

document.addEventListener('DOMContentLoaded', () => {
    // ─── Tab Switching ──────────────────────────────────
    const tabBtns = document.querySelectorAll('.tab-btn');
    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => b.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
            btn.classList.add('active');
            const tabId = btn.dataset.tab + 'Tab';
            document.getElementById(tabId).classList.add('active');
            
            // Stop camera if switching away
            if (btn.dataset.tab === 'file') stopCamera();
        });
    });

    // ─── File Upload & Drag-Drop ────────────────────────
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('fileInput');
    const imagePreview = document.getElementById('imagePreview');
    const previewImg = document.getElementById('previewImg');
    const removeImage = document.getElementById('removeImage');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');

    if (dropZone) {
        dropZone.addEventListener('click', () => fileInput.click());
        
        dropZone.addEventListener('dragover', (e) => {
            e.preventDefault();
            dropZone.classList.add('drag-over');
        });
        
        dropZone.addEventListener('dragleave', () => {
            dropZone.classList.remove('drag-over');
        });
        
        dropZone.addEventListener('drop', (e) => {
            e.preventDefault();
            dropZone.classList.remove('drag-over');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type.startsWith('image/')) {
                fileInput.files = e.dataTransfer.files;
                showPreview(files[0]);
            }
        });
    }

    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                showPreview(e.target.files[0]);
            }
        });
    }

    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            imagePreview.hidden = false;
            submitBtn.disabled = false;
            dropZone.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    if (removeImage) {
        removeImage.addEventListener('click', () => {
            imagePreview.hidden = true;
            submitBtn.disabled = true;
            fileInput.value = '';
            dropZone.style.display = 'block';
            previewImg.src = '';
        });
    }

    // ─── Camera ─────────────────────────────────────────
    const cameraPreview = document.getElementById('cameraPreview');
    const cameraCanvas = document.getElementById('cameraCanvas');
    const startCameraBtn = document.getElementById('startCamera');
    const captureBtn = document.getElementById('captureBtn');
    const retakeBtn = document.getElementById('retakeBtn');
    let cameraStream = null;

    if (startCameraBtn) {
        startCameraBtn.addEventListener('click', async () => {
            try {
                cameraStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
                });
                cameraPreview.srcObject = cameraStream;
                cameraPreview.hidden = false;
                startCameraBtn.hidden = true;
                captureBtn.hidden = false;
            } catch (err) {
                alert('Camera access denied. Please allow camera permissions or use file upload.');
                console.error('Camera error:', err);
            }
        });
    }

    if (captureBtn) {
        captureBtn.addEventListener('click', () => {
            cameraCanvas.width = cameraPreview.videoWidth;
            cameraCanvas.height = cameraPreview.videoHeight;
            const ctx = cameraCanvas.getContext('2d');
            ctx.drawImage(cameraPreview, 0, 0);
            
            // Convert canvas to blob and set as file input
            cameraCanvas.toBlob((blob) => {
                const file = new File([blob], 'soil_capture.jpg', { type: 'image/jpeg' });
                const dt = new DataTransfer();
                dt.items.add(file);
                fileInput.files = dt.files;
                
                previewImg.src = cameraCanvas.toDataURL('image/jpeg');
                imagePreview.hidden = false;
                submitBtn.disabled = false;
                
                // Hide camera, show retake
                cameraPreview.hidden = true;
                captureBtn.hidden = true;
                retakeBtn.hidden = false;
            }, 'image/jpeg', 0.9);
            
            stopCamera();
        });
    }

    if (retakeBtn) {
        retakeBtn.addEventListener('click', async () => {
            imagePreview.hidden = true;
            submitBtn.disabled = true;
            retakeBtn.hidden = true;
            fileInput.value = '';
            
            try {
                cameraStream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment' }
                });
                cameraPreview.srcObject = cameraStream;
                cameraPreview.hidden = false;
                captureBtn.hidden = false;
            } catch (err) {
                startCameraBtn.hidden = false;
            }
        });
    }

    function stopCamera() {
        if (cameraStream) {
            cameraStream.getTracks().forEach(track => track.stop());
            cameraStream = null;
        }
    }

    // ─── Form Submission with Loading State ─────────────
    if (uploadForm) {
        uploadForm.addEventListener('submit', (e) => {
            if (submitBtn.disabled) {
                e.preventDefault();
                return;
            }
            const btnText = submitBtn.querySelector('.btn-text');
            const btnLoader = submitBtn.querySelector('.btn-loader');
            if (btnText) btnText.hidden = true;
            if (btnLoader) btnLoader.hidden = false;
            submitBtn.disabled = true;
        });
    }

    // ─── Smooth Scroll ──────────────────────────────────
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function(e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) target.scrollIntoView({ behavior: 'smooth', block: 'start' });
        });
    });

    // ─── Intersection Observer for Animations ───────────
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('visible');
            }
        });
    }, { threshold: 0.1 });

    document.querySelectorAll('.step-card, .stat-card').forEach(el => {
        observer.observe(el);
    });
});
