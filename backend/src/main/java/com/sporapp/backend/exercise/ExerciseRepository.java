package com.sporapp.backend.exercise;

import java.util.List;

import org.springframework.data.jpa.repository.JpaRepository;

public interface ExerciseRepository extends JpaRepository<Exercise, Long> { // Bu repository Exercise entity'siyle çalışacak ve Exercise'ın ID tipi Long
                                                                            // extends ise save, findById, findAll, deleteById gibi temel işlemleri
    List<Exercise> findAllByOrderByNameAsc();

    /*
    find = Veri bul.
    All = Hepsini getir.
    OrderBy = Sıralama yap.
    Name = name alanına göre.
    Asc == Ascending → küçükten büyüğe / A'dan Z'ye.
    
    yani şuna benzer bir SQL üretir:

    SELECT *
    FROM exercises
    ORDER BY name ASC;

    */

    List<Exercise> findByMuscleGroupOrderByNameAsc(MuscleGroup muscleGroup);

    /*
    find
    By
    MuscleGroup
    OrderBy
    Name
    Asc

    Spring yaklaşık olarak:

    SELECT *
    FROM exercises
    WHERE muscle_group = 'CHEST'
    ORDER BY name ASC;


    */

    List<Exercise> findByEquipmentOrderByNameAsc(Equipment equipment);

    List<Exercise> findByMuscleGroupAndEquipmentOrderByNameAsc(MuscleGroup muscleGroup,
                                                               Equipment equipment);


    // bunlar ExerciseRepository içindeki metotların amacı database'deki exercises tablosundan veri çekmek.
}