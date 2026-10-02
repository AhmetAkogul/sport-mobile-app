package com.sporapp.backend.workoutsession;

import java.time.Instant;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.sporapp.backend.common.exception.ConflictException;
import com.sporapp.backend.common.exception.InvalidRequestException;
import com.sporapp.backend.common.exception.ResourceNotFoundException;
import com.sporapp.backend.exercise.Exercise;
import com.sporapp.backend.exercise.ExerciseRepository;
import com.sporapp.backend.user.User;
import com.sporapp.backend.user.UserRepository;
import com.sporapp.backend.workoutplan.PlanExercise;
import com.sporapp.backend.workoutplan.WorkoutPlan;
import com.sporapp.backend.workoutplan.WorkoutPlanRepository;
import com.sporapp.backend.workoutsession.dto.AddSessionExerciseRequest;
import com.sporapp.backend.workoutsession.dto.ExerciseResultRequest;
import com.sporapp.backend.workoutsession.dto.FinishSessionRequest;
import com.sporapp.backend.workoutsession.dto.SetResultRequest;
import com.sporapp.backend.workoutsession.dto.StartSessionRequest;
import com.sporapp.backend.workoutsession.dto.WorkoutSessionResponse;
import com.sporapp.backend.workoutsession.dto.WorkoutSessionSummaryResponse;

@Service
public class WorkoutSessionService {

    private final WorkoutSessionRepository workoutSessionRepository;
    private final SessionExerciseRepository sessionExerciseRepository;
    private final WorkoutPlanRepository workoutPlanRepository;
    private final ExerciseRepository exerciseRepository;
    private final UserRepository userRepository;

    public WorkoutSessionService(WorkoutSessionRepository workoutSessionRepository,
                                 SessionExerciseRepository sessionExerciseRepository,
                                 WorkoutPlanRepository workoutPlanRepository,
                                 ExerciseRepository exerciseRepository,
                                 UserRepository userRepository) {
        this.workoutSessionRepository = workoutSessionRepository;
        this.sessionExerciseRepository = sessionExerciseRepository;
        this.workoutPlanRepository = workoutPlanRepository;
        this.exerciseRepository = exerciseRepository;
        this.userRepository = userRepository;
    }

    @Transactional
    public WorkoutSessionResponse start(Long userId, StartSessionRequest request) {
        User user = userRepository.findById(userId)
                .orElseThrow(() -> new ResourceNotFoundException("Kullanıcı bulunamadı"));

        WorkoutSession session = new WorkoutSession();
        session.setUser(user);
        session.setStartedAt(Instant.now()); // saat sunucudan — istemci saati yanlış olabilir

        if (request.planId() != null) {
            // detay sorgusu: plan + egzersizler + her egzersizin bilgisi TEK sorguda gelir
            WorkoutPlan plan = workoutPlanRepository
                    .findDetailByIdAndUserId(request.planId(), userId)
                    .orElseThrow(() -> new ResourceNotFoundException("Antrenman planı bulunamadı"));

            session.setPlan(plan);
            copyPlanExercises(plan, session); // planin O GUNKU hali seansa kopyalanir
        }

        return WorkoutSessionResponse.from(workoutSessionRepository.save(session)); // cascade → satirlar da yazilir
    }

    @Transactional(readOnly = true)
    public List<WorkoutSessionSummaryResponse> list(Long userId) {
        return workoutSessionRepository.findSummariesByUserId(userId);
    }

    @Transactional(readOnly = true)
    public WorkoutSessionResponse getDetail(Long userId, Long sessionId) {
        return detail(userId, sessionId);
    }

    @Transactional
    public void delete(Long userId, Long sessionId) {
        workoutSessionRepository.delete(findOwned(userId, sessionId));
    }

    @Transactional
    public WorkoutSessionResponse addExercise(Long userId, Long sessionId,
                                              AddSessionExerciseRequest request) {
        WorkoutSession session = findOwned(userId, sessionId);

        Exercise exercise = exerciseRepository.findById(request.exerciseId())
                .orElseThrow(() -> new ResourceNotFoundException("Egzersiz bulunamadı"));

        SessionExercise sessionExercise = new SessionExercise();
        sessionExercise.setSession(session);
        sessionExercise.setExercise(exercise);
        sessionExercise.setPosition(request.position());
        sessionExercise.setTargetSets(request.targetSets());
        sessionExercise.setTargetReps(request.targetReps());
        sessionExercise.setTargetRestSeconds(request.targetRestSeconds());
        ensureNotFinished(session);

        session.getExercises().add(sessionExercise); // cascade ALL → save() gerekmez

        return detail(userId, sessionId);
    }

    @Transactional
    public void removeExercise(Long userId, Long sessionId, Long sessionExerciseId) {
        WorkoutSession session = findOwned(userId, sessionId);
        SessionExercise sessionExercise = sessionExerciseRepository
                .findByIdAndSessionId(sessionExerciseId, session.getId())
                .orElseThrow(() -> new ResourceNotFoundException("Seans egzersizi bulunamadı"));
                ensureNotFinished(session);

        session.getExercises().remove(sessionExercise); // orphanRemoval → DELETE
    }






     @Transactional
    public WorkoutSessionResponse finish(Long userId, Long sessionId, FinishSessionRequest request) {
        // detay sorgusu: seans + plan + egzersizler + egzersiz bilgileri TEK sorguda
        WorkoutSession session = workoutSessionRepository
                .findDetailByIdAndUserId(sessionId, userId)
                .orElseThrow(() -> new ResourceNotFoundException("Antrenman seansı bulunamadı"));

        // gelen sessionExerciseId'leri bu seansin satirlariyla eslestirmek icin sozluk
        Map<Long, SessionExercise> seansEgzersizleri = new HashMap<>();
        for (SessionExercise se : session.getExercises()) {
            seansEgzersizleri.put(se.getId(), se);
        }

        Set<Long> gorulenler = new HashSet<>();

        for (ExerciseResultRequest sonuc : request.exercises()) {
            // ayni egzersiz iki kez gelirse ikincisi birincinin sonuclarini sessizce silerdi
            if (!gorulenler.add(sonuc.sessionExerciseId())) {
                throw new InvalidRequestException(
                        "Aynı seans egzersizi birden fazla kez gönderildi: " + sonuc.sessionExerciseId());
            }

            SessionExercise se = seansEgzersizleri.get(sonuc.sessionExerciseId());
            if (se == null) {
                // baskasinin seansi veya bu seansa ait olmayan id → 404 (varlik sizdirilmaz)
                throw new ResourceNotFoundException("Seans egzersizi bulunamadı");
            }

            // tekrar cagrilirsa eski sonuclarin yerine yenisi yazilir (orphanRemoval → DELETE)
            se.getSets().clear();

            for (SetResultRequest setResult : sonuc.sets()) {
                if (setResult.incorrectReps() != null && setResult.incorrectReps() > setResult.reps()) {
                    throw new InvalidRequestException(
                            "Yanlış tekrar sayısı toplam tekrardan büyük olamaz: "
                                    + setResult.incorrectReps() + " > " + setResult.reps());
                }

                ExerciseSet set = new ExerciseSet();
                set.setSessionExercise(se);
                set.setSetNumber(setResult.setNumber());
                set.setReps(setResult.reps());
                set.setWeight(setResult.weight());
                set.setIncorrectReps(setResult.incorrectReps());

                se.getSets().add(set); // cascade ALL → save() gerekmez
            }
        }

        if (request.note() != null) { // gonderilmediyse mevcut not korunur
            session.setNote(request.note());
        }

        if (session.getFinishedAt() == null) { // yumusak koruma: ilk bitis saati degismez
            session.setFinishedAt(Instant.now());
        }

        return detail(userId, sessionId);
    }

    private void ensureNotFinished(WorkoutSession session) {
        if (session.getFinishedAt() != null) {
            throw new ConflictException("Bitmiş antrenman seansına egzersiz eklenip çıkarılamaz");
        }
    }







    private void copyPlanExercises(WorkoutPlan plan, WorkoutSession session) {
        for (PlanExercise planExercise : plan.getExercises()) {
            SessionExercise sessionExercise = new SessionExercise();
            sessionExercise.setSession(session);
            sessionExercise.setExercise(planExercise.getExercise());
            sessionExercise.setPosition(planExercise.getPosition());
            sessionExercise.setTargetSets(planExercise.getSets());
            sessionExercise.setTargetReps(planExercise.getReps());
            sessionExercise.setTargetRestSeconds(planExercise.getRestSeconds());

            session.getExercises().add(sessionExercise);
        }
    }

    private WorkoutSession findOwned(Long userId, Long sessionId) {
        return workoutSessionRepository.findByIdAndUserId(sessionId, userId)
                .orElseThrow(() -> new ResourceNotFoundException("Antrenman seansı bulunamadı"));
    }

    private WorkoutSessionResponse detail(Long userId, Long sessionId) {
        return WorkoutSessionResponse.from(workoutSessionRepository
                .findDetailByIdAndUserId(sessionId, userId)
                .orElseThrow(() -> new ResourceNotFoundException("Antrenman seansı bulunamadı")));
    }
}