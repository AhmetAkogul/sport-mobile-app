package com.sporapp.backend.exercise;

import java.util.List;

import org.springframework.stereotype.Service;
import org.springframework.transaction.annotation.Transactional;

import com.sporapp.backend.common.exception.ResourceNotFoundException;
import com.sporapp.backend.exercise.dto.ExerciseResponse;

@Service
public class ExerciseService {

    private final ExerciseRepository exerciseRepository;

    public ExerciseService(ExerciseRepository exerciseRepository) {
        this.exerciseRepository = exerciseRepository;
    }

    @Transactional(readOnly = true)
    public List<ExerciseResponse> list(MuscleGroup muscleGroup, Equipment equipment) {
        return filter(muscleGroup, equipment).stream()
                .map(ExerciseResponse::from) // her entity'yi DTO'ya çevir
                .toList();                   // değişmez liste (Java 16+)
    }

    @Transactional(readOnly = true)
    public ExerciseResponse getById(Long id) {
        return ExerciseResponse.from(exerciseRepository.findById(id)
                .orElseThrow(() -> new ResourceNotFoundException("Egzersiz bulunamadı")));
    }

    private List<Exercise> filter(MuscleGroup muscleGroup, Equipment equipment) { // 4 olasılık → hangi sorgu?
        if (muscleGroup != null && equipment != null) {
            return exerciseRepository.findByMuscleGroupAndEquipmentOrderByNameAsc(muscleGroup, equipment);
        }
        if (muscleGroup != null) {
            return exerciseRepository.findByMuscleGroupOrderByNameAsc(muscleGroup);
        }
        if (equipment != null) {
            return exerciseRepository.findByEquipmentOrderByNameAsc(equipment);
        }
        return exerciseRepository.findAllByOrderByNameAsc();
    }
}