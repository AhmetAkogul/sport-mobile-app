package com.sporapp.backend.statistics;

import java.math.BigDecimal;
import java.time.DayOfWeek;
import java.time.Duration;
import java.time.Instant;
import java.time.LocalDate;
import java.time.ZoneId;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.sporapp.backend.common.exception.InvalidRequestException;
import com.sporapp.backend.statistics.dto.AccuracyStatisticsResponse;
import com.sporapp.backend.statistics.dto.MuscleGroupStatisticsResponse;
import com.sporapp.backend.statistics.dto.StatisticsOverviewResponse;
import com.sporapp.backend.statistics.dto.WeeklyStatisticsResponse;

/**
 * Istatistik hesaplari. Tum sorgular kullaniciya filtrelidir; baska
 * kullanicinin verisi buraya hicbir sekilde sizmaz.
 */
@Service
public class StatisticsService {

    private static final int MIN_WEEKS = 1;
    private static final int MAX_WEEKS = 52;

    private final StatisticsRepository statisticsRepository;

    public StatisticsService(StatisticsRepository statisticsRepository) {
        this.statisticsRepository = statisticsRepository;
    }

    @Transactional(readOnly = true)
    public StatisticsOverviewResponse overview(Long userId) {
        List<SessionTiming> timings = statisticsRepository.findFinishedTimings(userId);

        // Sure ve kalori tek sorgudan gelir; set/tekrar/hacim SQL tarafinda toplanir.
        long totalMinutes = timings.stream().mapToLong(StatisticsService::minutesOf).sum();

        BigDecimal totalCalories = timings.stream()
                .map(timing -> timing.calories())
                .filter(Objects::nonNull)   // kilo girilmemis eski seanslar NULL kalori tasir
                .reduce(BigDecimal.ZERO, (a, b) -> a.add(b));

        Instant firstWorkoutAt = timings.stream()
                .map(timing -> timing.startedAt())
                .min((a, b) -> a.compareTo(b))
                .orElse(null);

        Instant lastWorkoutAt = timings.stream()
                .map(timing -> timing.finishedAt())
                .max((a, b) -> a.compareTo(b))
                .orElse(null);

        return new StatisticsOverviewResponse(
                timings.size(),
                totalMinutes,
                totalCalories,
                statisticsRepository.countSets(userId),
                statisticsRepository.sumReps(userId),
                statisticsRepository.sumVolumeKg(userId),
                firstWorkoutAt,
                lastWorkoutAt);
    }

    /**
     * Haftalik kovalar. Gruplama Java tarafinda yapilir cunku Postgres'in
     * date_trunc('week') fonksiyonu SUNUCU saat dilimini kullanir; kullanici
     * baskа saat dilimindeyse hafta kaymalari yanlis olur. Girilen zone ile
     * cevirip haftanin pazartesisine indiriyoruz.
     */
    @Transactional(readOnly = true)
    public List<WeeklyStatisticsResponse> weekly(Long userId, int weeks, String zone) {
        if (weeks < MIN_WEEKS || weeks > MAX_WEEKS) {
            throw new InvalidRequestException(
                    "Hafta sayisi " + MIN_WEEKS + " ile " + MAX_WEEKS + " arasinda olmali");
        }
        if (zone == null || !ZoneId.getAvailableZoneIds().contains(zone)) {
            throw new InvalidRequestException("Gecersiz saat dilimi: " + zone);
        }

        ZoneId bolge = ZoneId.of(zone);

        // Bu haftanin pazartesisi; oradan geriye dogru kovalari hazirla.
        LocalDate buHafta = LocalDate.now(bolge).with(DayOfWeek.MONDAY);
        LocalDate ilkHafta = buHafta.minusWeeks(weeks - 1L);

        // LinkedHashMap: ekleme sirasi korunur, cikti eskiden yeniye gelir.
        Map<LocalDate, Bucket> kovalar = new LinkedHashMap<>();
        for (int i = 0; i < weeks; i++) {
            kovalar.put(ilkHafta.plusWeeks(i), new Bucket());
        }

        for (SessionTiming timing : statisticsRepository.findFinishedTimings(userId)) {
            LocalDate hafta = timing.startedAt()
                    .atZone(bolge)
                    .toLocalDate()
                    .with(DayOfWeek.MONDAY);

            // Pencerenin disindaki (eski) seanslar haritada karsilik bulamaz, atlanir.
            Bucket kova = kovalar.get(hafta);
            if (kova == null) {
                continue;
            }

            kova.sessions++;
            kova.minutes += minutesOf(timing);
            if (timing.calories() != null) {
                kova.calories = kova.calories.add(timing.calories());
            }
        }

        return kovalar.entrySet().stream()
                .map(entry -> new WeeklyStatisticsResponse(
                        entry.getKey(),
                        entry.getValue().sessions,
                        entry.getValue().minutes,
                        entry.getValue().calories))
                .toList();
    }

    @Transactional(readOnly = true)
    public List<MuscleGroupStatisticsResponse> muscleGroups(Long userId) {
        return statisticsRepository.findMuscleGroupStats(userId);
    }

    @Transactional(readOnly = true)
    public AccuracyStatisticsResponse accuracy(Long userId) {
        return AccuracyStatisticsResponse.of(
                statisticsRepository.sumReps(userId),
                statisticsRepository.sumTrackedReps(userId),
                statisticsRepository.sumIncorrectReps(userId));
    }

    // Duration.toMinutes() tam dakikayi verir (asagi yuvarlar): 90 sn -> 1 dk.
    private static long minutesOf(SessionTiming timing) {
        return Duration.between(timing.startedAt(), timing.finishedAt()).toMinutes();
    }

    // Sadece servis icinde kullanilan degistirilebilir sayac.
    private static final class Bucket {
        private long sessions;
        private long minutes;
        private BigDecimal calories = BigDecimal.ZERO;
    }
}
