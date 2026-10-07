package com.sporapp.backend.statistics;

import java.util.List;

import org.springframework.http.ResponseEntity;
import org.springframework.security.core.annotation.AuthenticationPrincipal;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.sporapp.backend.statistics.dto.AccuracyStatisticsResponse;
import com.sporapp.backend.statistics.dto.MuscleGroupStatisticsResponse;
import com.sporapp.backend.statistics.dto.StatisticsOverviewResponse;
import com.sporapp.backend.statistics.dto.WeeklyStatisticsResponse;

/**
 * Kullanici kendi istatistiklerini gorur; kullanici id token'dan gelir,
 * istek parametresi olarak disaridan alinmaz.
 */
@RestController
@RequestMapping("/api/statistics")
public class StatisticsController {

    private final StatisticsService statisticsService;

    public StatisticsController(StatisticsService statisticsService) {
        this.statisticsService = statisticsService;
    }

    @GetMapping("/overview")
    public ResponseEntity<StatisticsOverviewResponse> overview(
            @AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(statisticsService.overview(userId));
    }

    @GetMapping("/weekly")
    public ResponseEntity<List<WeeklyStatisticsResponse>> weekly(
            @AuthenticationPrincipal Long userId,
            @RequestParam(defaultValue = "8") int weeks,
            @RequestParam(defaultValue = "UTC") String zone) {
        return ResponseEntity.ok(statisticsService.weekly(userId, weeks, zone));
    }

    @GetMapping("/muscle-groups")
    public ResponseEntity<List<MuscleGroupStatisticsResponse>> muscleGroups(
            @AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(statisticsService.muscleGroups(userId));
    }

    @GetMapping("/accuracy")
    public ResponseEntity<AccuracyStatisticsResponse> accuracy(
            @AuthenticationPrincipal Long userId) {
        return ResponseEntity.ok(statisticsService.accuracy(userId));
    }
}
