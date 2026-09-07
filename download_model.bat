@echo off
setlocal

conda run -n ocr-redaction-poc python -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id='ucaslcl/GOT-OCR2_0', local_dir='models/got_ocr')"
if errorlevel 1 (
    echo.
    echo ERROR: GOT-OCR2 download failed. Check Bosch VPN/network access.
    exit /b 1
)

echo GOT-OCR2 download completed.