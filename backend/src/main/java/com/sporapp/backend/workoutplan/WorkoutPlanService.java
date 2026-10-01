package com.sporapp.backend.workoutplan;

import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.sporapp.backend.common.exception.ResourceNotFoundException;
import com.sporapp.backend.exercise.Exercise;
import com.sporapp.backend.exercise.ExerciseRepository;
import com.sporapp.backend.user.User;
import com.sporapp.backend.user.UserRepository;
import com.sporapp.backend.workoutplan.dto.AddPlanExerciseRequest;
import com.sporapp.backend.workoutplan.dto.UpdatePlanExerciseRequest;
import com.sporapp.backend.workoutplan.dto.WorkoutPlanRequest;
import com.sporapp.backend.workoutplan.dto.WorkoutPlanResponse;
import com.sporapp.backend.workoutplan.dto.WorkoutPlanSummaryResponse;

@Service
public class WorkoutPlanService {

    private final WorkoutPlanRepository workoutPlanRepository;
    private final PlanExerciseRepository planExerciseRepository;
    private final ExerciseRepository exerciseRepository;
    private final UserRepository userRepository;  // bunlarla repodan kullanıcı verilerini dbden çekiyor

    public WorkoutPlanService(WorkoutPlanRepository workoutPlanRepository,
                              PlanExerciseRepository planExerciseRepository,
                              ExerciseRepository exerciseRepository,
                              UserRepository userRepository) {
        this.workoutPlanRepository = workoutPlanRepository;
        this.planExerciseRepository = planExerciseRepository;
        this.exerciseRepository = exerciseRepository;
        this.userRepository = userRepository;
    }

    @Transactional
    public WorkoutPlanResponse create(Long userId, WorkoutPlanRequest request) {
        User user = userRepository.findById(userId) // token geçerli olsa da kullanıcı silinmiş olabilir
                .orElseThrow(() -> new ResourceNotFoundException("Kullanıcı bulunamadı"));

        WorkoutPlan plan = new WorkoutPlan();
        plan.setUser(user);
        plan.setName(request.name().trim());
        plan.setDescription(request.description());

        return WorkoutPlanResponse.from(workoutPlanRepository.save(plan)); // egzersiz listesi boş
    }

    @Transactional(readOnly = true)
    public List<WorkoutPlanSummaryResponse> list(Long userId) {
        return workoutPlanRepository.findSummariesByUserId(userId); // tek sorgu, egzersizler yüklenmez
    }

    @Transactional(readOnly = true)
    public WorkoutPlanResponse getDetail(Long userId, Long planId) {
        return detail(userId, planId);
    }

    @Transactional
    public WorkoutPlanResponse update(Long userId, Long planId, WorkoutPlanRequest request) {
        WorkoutPlan plan = findOwnedPlan(userId, planId); // yoksa veya başkasınınsa 404

        plan.setName(request.name().trim());
        plan.setDescription(request.description());
        // save() yok: entity managed, dirty checking UPDATE'i flush'ta atar

        return detail(userId, planId);
    }

    @Transactional
    public void delete(Long userId, Long planId) {
        workoutPlanRepository.delete(findOwnedPlan(userId, planId)); // cascade ALL → satırlar da silinir
    }

    @Transactional
    public WorkoutPlanResponse addExercise(Long userId, Long planId, AddPlanExerciseRequest request) {
        WorkoutPlan plan = findOwnedPlan(userId, planId);

        Exercise exercise = exerciseRepository.findById(request.exerciseId())
                .orElseThrow(() -> new ResourceNotFoundException("Egzersiz bulunamadı"));

        PlanExercise planExercise = new PlanExercise();
        planExercise.setPlan(plan);
        planExercise.setExercise(exercise);
        planExercise.setPosition(request.position());
        planExercise.setSets(request.sets());
        planExercise.setReps(request.reps());
        planExercise.setRestSeconds(request.restSeconds());

        plan.getExercises().add(planExercise); // cascade ALL → save() çağırmaya gerek yok

        return detail(userId, planId);
    }

    @Transactional
    public WorkoutPlanResponse updateExercise(Long userId, Long planId, Long planExerciseId,
                                              UpdatePlanExerciseRequest request) {
        WorkoutPlan plan = findOwnedPlan(userId, planId);
        PlanExercise planExercise = findOwnedPlanExercise(plan, planExerciseId);

        planExercise.setPosition(request.position());
        planExercise.setSets(request.sets());
        planExercise.setReps(request.reps());
        planExercise.setRestSeconds(request.restSeconds());

        return detail(userId, planId);
    }

    @Transactional
    public void removeExercise(Long userId, Long planId, Long planExerciseId) {
        WorkoutPlan plan = findOwnedPlan(userId, planId);
        PlanExercise planExercise = findOwnedPlanExercise(plan, planExerciseId);

        plan.getExercises().remove(planExercise); // orphanRemoval → DELETE
    }

    private WorkoutPlan findOwnedPlan(Long userId, Long planId) {
        return workoutPlanRepository.findByIdAndUserId(planId, userId)
                .orElseThrow(() -> new ResourceNotFoundException("Antrenman planı bulunamadı"));
    }

    private PlanExercise findOwnedPlanExercise(WorkoutPlan plan, Long planExerciseId) {
        return planExerciseRepository.findByIdAndPlanId(planExerciseId, plan.getId())
                .orElseThrow(() -> new ResourceNotFoundException("Plan egzersizi bulunamadı"));
    }

    private WorkoutPlanResponse detail(Long userId, Long planId) {
        return WorkoutPlanResponse.from(workoutPlanRepository
                .findDetailByIdAndUserId(planId, userId)
                .orElseThrow(() -> new ResourceNotFoundException("Antrenman planı bulunamadı")));
    }
}