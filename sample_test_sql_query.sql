SELECT 
    model,
    COUNT(*) as total_predictions,
    SUM(CASE 
        WHEN (label >= 0.5 AND model_is_hate_speech = 1) OR 
             (label < 0.5 AND model_is_hate_speech = 0) 
        THEN 1 ELSE 0 
    END) as correct_predictions,
    ROUND(
        (SUM(CASE 
            WHEN (label >= 0.5 AND model_is_hate_speech = 1) OR 
                 (label < 0.5 AND model_is_hate_speech = 0) 
            THEN 1 ELSE 0 
        END) * 100.0) / COUNT(*), 2
    ) as accuracy_percentage
FROM hate_speech_results 
WHERE label IS NOT NULL
GROUP BY model
ORDER BY accuracy_percentage DESC;
