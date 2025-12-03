$Url = "http://localhost:8000/predict"

Write-Host "Testing Emotion API with requests..." -ForegroundColor Cyan
Write-Host "========================================"

# Список векторов для теста
$Vectors = @(
    @(0.85, 0.05, 0.10, 0.60, 0.15, 0.10),
    @(0.78, 0.45, 0.15, 0.10, 0.20, 0.10),
    @(0.65, 0.10, 0.50, 0.05, 0.15, 0.20),
    @(0.45, 0.08, 0.12, 0.05, 0.10, 0.65),
    @(0.70, 0.20, 0.20, 0.20, 0.20, 0.20),
    @(0.60, 0.10, 0.10, 0.10, 0.50, 0.20),
    @(0.55, 0.60, 0.10, 0.05, 0.15, 0.10),
    @(0.50, 0.15, 0.35, 0.05, 0.10, 0.35),
    @(0.90, 0.02, 0.03, 0.70, 0.20, 0.05),
    @(0.75, 0.25, 0.25, 0.15, 0.20, 0.15)
)

$i = 1
foreach ($Vector in $Vectors) {
    # Формируем JSON тело
    $Body = @{
        emotions = $Vector
    } | ConvertTo-Json -Compress

    Write-Host "Request $i : $Body"

    try {
        # Отправляем запрос
        $Response = Invoke-RestMethod -Uri $Url -Method Post -Body $Body -ContentType "application/json"
        
        # Выводим красивый ответ
        $Preset = $Response.preset
        $Conf = $Response.confidence
        Write-Host "Response  : PRESET=$Preset (Conf: $Conf)" -ForegroundColor Green
    }
    catch {
        Write-Host "Error: $_" -ForegroundColor Red
    }
    
    Write-Host "---"
    $i++
}

Write-Host "Done!"