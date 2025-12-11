# Script para configurar variáveis de ambiente do OpenGL/Qt
# Execute com: .\setup_env.ps1

Write-Host "🔧 Configurando variáveis de ambiente para OpenGL/Qt..." -ForegroundColor Cyan

# Variáveis de ambiente para Qt/OpenGL
$env:QT_AUTO_SCREEN_SCALE_FACTOR = "1"
$env:QT_ENABLE_HIGHDPI_SCALING = "1"

# Força o uso de software rendering se houver problemas com GPU
# Descomente a linha abaixo se tiver problemas:
# $env:QT_OPENGL = "software"

# Para PyQt6/vispy
$env:PYOPENGL_PLATFORM = "osmesa"

Write-Host "✅ Variáveis configuradas!" -ForegroundColor Green
Write-Host ""
Write-Host "Variáveis definidas:" -ForegroundColor Yellow
Write-Host "  QT_AUTO_SCREEN_SCALE_FACTOR = $env:QT_AUTO_SCREEN_SCALE_FACTOR"
Write-Host "  QT_ENABLE_HIGHDPI_SCALING = $env:QT_ENABLE_HIGHDPI_SCALING"
Write-Host ""
Write-Host "🚀 Executando o script..." -ForegroundColor Cyan
python animation_cube.py
