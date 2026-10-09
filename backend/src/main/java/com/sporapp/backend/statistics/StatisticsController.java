package com.sporapp.backend.statistics;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.sporapp.backend.config.OpenApiConfig;
import com.sporapp.backend.statistics.dto.AccuracyStatisticsResponse;
import com.sporapp.backend.statistics.dto.MuscleGroupStatisticsResponse;
import com.sporapp.backend.statistics.dto.StatisticsOverviewResponse;
import com.sporapp.backend.statistics.dto.WeeklyStatisticsResponse;

import io.swagger.v3.oas.annotations.Operation;
import io.swagger.v3.oas.annotations.security.SecurityRequirement;
import io.swagger.v3.oas.annotations.tags.Tag;

/**
 * Kullanici kendi istatistiklerini gorur; kullanici id token'dan gelir,
 * istek parametresi olarak disaridan alinmaz.
 * Tum istatistikler SADECE bitmis seanslari sayar (finishedAt != null).
 */
@Tag(name = "İstatistikler", description = "Sadece bitirilmiş seanslar sayılır. Kullanıcı id token'dan gelir.")
@SecurityRequirement(name = OpenApiConfig.GUVENLIK_SEMASI)
@RestController
@RequestMapping("/api/statistics")
public class StatisticsController {

    private final StatisticsService statisticsService;

    public StatisticsController(StatisticsService statisticsService) {
        this.statisticsService = statisticsService;
    }

    @Operation(summary = "Genel toplamlar",
            description = """
                    Hiç bitmiş seans yoksa tüm sayılar 0, firstWorkoutAt/lastWorkoutAt null döner.
                    totalVolumeKg = Σ(tekrar × ağırlık); vücut ağırlığı setleri (weight null) hacme girmez.""")
    @GetMapping("/overview")
    public ResponseEntity<StatisticsOverviewResponse> overview(
            @AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(statisticsService.overview(userId));
    }

    @Operation(summary = "Haftalık kovalar", description = """
            weekStart her zaman pazartesidir. Boş haftalar 0 değerleriyle döner, grafikte boşluk oluşmaz.
            zone: IANA saat dilimi (örn. Europe/Istanbul). Gruplama bu saat dilimine göre yapılır.
            weeks 1-52 arasında olmalı, aksi halde 400.""")
    @GetMapping("/weekly")
    public ResponseEntity<List<WeeklyStatisticsResponse>> weekly(
            @AuthenticationPrincipal Long userId,
            @RequestParam(defaultValue = "8") int weeks,
            @RequestParam(defaultValue = "UTC") String zone) {
        return ResponseEntity.ok(statisticsService.weekly(userId, weeks, zone));
    }

    @Operation(summary = "Kas grubu dağılımı", description = "Set sayısına göre azalan sıralı.")
    @GetMapping("/muscle-groups")
    public ResponseEntity<List<MuscleGroupStatisticsResponse>> muscleGroups(
            @AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(statisticsService.muscleGroups(userId));
    }

    @Operation(summary = "AI hareket doğruluğu", description = """
            trackedReps = sadece AI'in baktigi setlerin tekrari; kamerasiz yapilan setler orana girmez.
            accuracyPercent, hic AI verisi yoksa null doner - "%0 dogru" demek yaniltirdi.""")
    @GetMapping("/accuracy")
    public ResponseEntity<AccuracyStatisticsResponse> accuracy(
            @AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(statisticsService.accuracy(userId));
    }
}
