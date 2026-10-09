-- Gercek egzersiz katalogu: 54 egzersiz, 8 kas grubu.
--
-- ON CONFLICT (name) DO UPDATE neden:
--   1) name kolonu UNIQUE (bkz. V4), migration tekrar calistirilirsa cokmez.
--   2) DB'de zaten duran 3 gecici test egzersizinin (Bench Press, Squat, Plank)
--      met_value'su NULL'di; bu sayede onlar da gercek MET degerini alir.
--      Boylece calories/MetValues yedek tablosu devreden cikar.
--
-- MET degerleri Compendium of Physical Activities'tan:
--   izole agirlik hareketleri 3.5 | bilesik agirlik hareketleri 5.0-6.0
--   kalistenik (vucut agirligi) 8.0 | kardiyo 5.0-12.3
--
-- created_at/updated_at NOT NULL oldugu icin (bkz. V4) her satirda elle
-- verilir. Insert'te ikisi de NOW(); UPDATE dalinda sadece updated_at tazelenir
-- (created_at ilk olusturulma anini korur).
--
-- description bilincli olarak NULL: mobil takim kendi metinlerini yazacak.
-- Isimler bilincli olarak Ingilizce: API makine-okunur kalir, mobil cevirir.

INSERT INTO exercises (name, muscle_group, equipment, met_value, created_at, updated_at)
VALUES
    -- CHEST
    ('Bench Press',              'CHEST',     'BARBELL',    6.0, NOW(), NOW()),
    ('Incline Bench Press',      'CHEST',     'BARBELL',    5.0, NOW(), NOW()),
    ('Dumbbell Fly',             'CHEST',     'DUMBBELL',   3.5, NOW(), NOW()),
    ('Push-Up',                  'CHEST',     'BODYWEIGHT', 8.0, NOW(), NOW()),
    ('Cable Crossover',          'CHEST',     'CABLE',      3.5, NOW(), NOW()),
    ('Chest Dip',                'CHEST',     'BODYWEIGHT', 8.0, NOW(), NOW()),

    -- BACK
    ('Deadlift',                 'BACK',      'BARBELL',    6.0, NOW(), NOW()),
    ('Barbell Row',              'BACK',      'BARBELL',    5.0, NOW(), NOW()),
    ('Lat Pulldown',             'BACK',      'CABLE',      5.0, NOW(), NOW()),
    ('Pull-Up',                  'BACK',      'BODYWEIGHT', 8.0, NOW(), NOW()),
    ('Seated Cable Row',         'BACK',      'CABLE',      5.0, NOW(), NOW()),
    ('Dumbbell Row',             'BACK',      'DUMBBELL',   5.0, NOW(), NOW()),
    ('Face Pull',                'BACK',      'CABLE',      3.5, NOW(), NOW()),

    -- LEGS
    ('Squat',                    'LEGS',      'BARBELL',    6.0, NOW(), NOW()),
    ('Leg Press',                'LEGS',      'MACHINE',    5.0, NOW(), NOW()),
    ('Romanian Deadlift',        'LEGS',      'BARBELL',    6.0, NOW(), NOW()),
    ('Lunge',                    'LEGS',      'DUMBBELL',   5.0, NOW(), NOW()),
    ('Leg Extension',            'LEGS',      'MACHINE',    3.5, NOW(), NOW()),
    ('Leg Curl',                 'LEGS',      'MACHINE',    3.5, NOW(), NOW()),
    ('Calf Raise',               'LEGS',      'MACHINE',    3.5, NOW(), NOW()),
    ('Hip Thrust',               'LEGS',      'BARBELL',    5.0, NOW(), NOW()),

    -- SHOULDERS
    ('Overhead Press',           'SHOULDERS', 'BARBELL',    5.0, NOW(), NOW()),
    ('Lateral Raise',            'SHOULDERS', 'DUMBBELL',   3.5, NOW(), NOW()),
    ('Front Raise',              'SHOULDERS', 'DUMBBELL',   3.5, NOW(), NOW()),
    ('Rear Delt Fly',            'SHOULDERS', 'DUMBBELL',   3.5, NOW(), NOW()),
    ('Arnold Press',             'SHOULDERS', 'DUMBBELL',   5.0, NOW(), NOW()),
    ('Upright Row',              'SHOULDERS', 'BARBELL',    3.5, NOW(), NOW()),

    -- ARMS
    ('Barbell Curl',             'ARMS',      'BARBELL',    3.5, NOW(), NOW()),
    ('Dumbbell Curl',            'ARMS',      'DUMBBELL',   3.5, NOW(), NOW()),
    ('Hammer Curl',              'ARMS',      'DUMBBELL',   3.5, NOW(), NOW()),
    ('Preacher Curl',            'ARMS',      'BARBELL',    3.5, NOW(), NOW()),
    ('Triceps Pushdown',         'ARMS',      'CABLE',      3.5, NOW(), NOW()),
    ('Skull Crusher',            'ARMS',      'BARBELL',    3.5, NOW(), NOW()),
    ('Close-Grip Bench Press',   'ARMS',      'BARBELL',    5.0, NOW(), NOW()),
    ('Triceps Dip',              'ARMS',      'BODYWEIGHT', 8.0, NOW(), NOW()),

    -- CORE
    ('Plank',                    'CORE',      'BODYWEIGHT', 3.5, NOW(), NOW()),
    ('Crunch',                   'CORE',      'BODYWEIGHT', 3.8, NOW(), NOW()),
    ('Hanging Leg Raise',        'CORE',      'BODYWEIGHT', 8.0, NOW(), NOW()),
    ('Russian Twist',            'CORE',      'BODYWEIGHT', 3.8, NOW(), NOW()),
    ('Ab Wheel Rollout',         'CORE',      'OTHER',      3.8, NOW(), NOW()),
    ('Mountain Climber',         'CORE',      'BODYWEIGHT', 8.0, NOW(), NOW()),
    ('Bicycle Crunch',           'CORE',      'BODYWEIGHT', 3.8, NOW(), NOW()),

    -- CARDIO
    ('Treadmill Run',            'CARDIO',    'MACHINE',    9.8, NOW(), NOW()),
    ('Cycling',                  'CARDIO',    'MACHINE',    7.5, NOW(), NOW()),
    ('Rowing Machine',           'CARDIO',    'MACHINE',    7.0, NOW(), NOW()),
    ('Jump Rope',                'CARDIO',    'OTHER',     12.3, NOW(), NOW()),
    ('Elliptical',               'CARDIO',    'MACHINE',    5.0, NOW(), NOW()),
    ('Stair Climber',            'CARDIO',    'MACHINE',    9.0, NOW(), NOW()),
    ('Burpee',                   'CARDIO',    'BODYWEIGHT', 8.0, NOW(), NOW()),

    -- FULL_BODY
    ('Kettlebell Swing',         'FULL_BODY', 'KETTLEBELL', 9.8, NOW(), NOW()),
    ('Clean and Press',          'FULL_BODY', 'BARBELL',    6.0, NOW(), NOW()),
    ('Thruster',                 'FULL_BODY', 'BARBELL',    6.0, NOW(), NOW()),
    ('Turkish Get-Up',           'FULL_BODY', 'KETTLEBELL', 6.0, NOW(), NOW()),
    ('Battle Ropes',             'FULL_BODY', 'OTHER',      8.0, NOW(), NOW())
ON CONFLICT (name) DO UPDATE SET
    muscle_group = EXCLUDED.muscle_group,
    equipment    = EXCLUDED.equipment,
    met_value    = EXCLUDED.met_value,
    updated_at   = NOW();
