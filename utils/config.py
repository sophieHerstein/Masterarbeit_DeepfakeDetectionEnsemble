# Array mit Kategorien
CATEGORIES = ["human", "building", "landscape"]

# Anzahl der Varianten je Prompt
VARIANTEN_BEKANNT = 14
VARIANTEN_UNBEKANNT = 6

MODELS = [] #todo: ergänzen

# enthält Pfade usw.
CONFIG = {
    "synthetic_images_log_path": "logs/image_synthesis/image_synthesis_log.csv",
    "images_path": "images",
    "preprocessed_images_path": "images/preprocessed",
}

# Prompts für synthetische Bilder
PROMPTS = {
    "human": [ # → 25 Prompts á 14 Varianten bzw. á 6 Varianten für unbekannten Datensatz
        "portrait photo of a young Asian woman with long black hair, natural daylight, 50mm lens",
        # "portrait of an elderly Black man with gray beard, warm indoor lighting",
        # "studio portrait of a middle-aged Caucasian woman with freckles, soft background",
        # "photo of a young boy with curly hair, smiling outdoors in the park",
        # "professional headshot of a South Asian man with glasses, neutral background",
        # "photo of a teenage girl in casual clothing, standing on a city street",
        # "close-up portrait of a young Arab woman wearing a headscarf, soft light",
        # "photo of an elderly East Asian woman with wrinkles, natural daylight",
        # "portrait of a middle-aged Hispanic man, short hair, warm tone lighting",
        # "photo of a young woman with dyed blue hair, street style fashion",
        # "group photo of three friends, mixed ethnicities, laughing in a café",
        # "photo of a toddler with blond hair, sitting on the grass",
        # "portrait of a young Black woman with braids, golden hour lighting",
        # "photo of a middle-aged man with a beard, wearing a suit outdoors",
        # "portrait of a teenage boy with acne, school hallway background",
        # "photo of a pregnant woman smiling, home interior",
        # "portrait of a man with long hair and tattoos, dramatic lighting",
        # "photo of an elderly couple holding hands in a park",
        # "portrait of a young Indigenous woman in traditional clothing",
        # "photo of a middle-aged man with shaved head, gym background",
        # "portrait of a young woman with glasses, in a library",
        # "photo of a man in his 30s with a beard, winter jacket, snowy background",
        # "close-up of a young child with face paint, festival background",
        # "portrait of a middle-aged woman with short hair, workplace setting",
        # "photo of a diverse group of five people posing outdoors, wide shot"
    ],
    "building": [ # → 25 Prompts á 14 Varianten bzw. á 6 Varianten für unbekannten Datensatz
        "modern glass skyscraper in city center, photographed at dusk",
        # "old European cathedral with gothic details, daylight",
        # "traditional Japanese wooden house, garden with lanterns",
        # "minimalist concrete building, clean lines, overcast sky",
        # "colorful houses along a canal, daytime",
        # "desert adobe houses, warm sunset light",
        # "classic New York brownstone street, autumn leaves",
        # "futuristic office tower with curved glass, blue sky",
        # "ancient stone castle on a hill, cloudy sky",
        # "suburban residential street with family houses",
        # "Greek island whitewashed buildings with blue roofs",
        # "old industrial warehouse with brick walls",
        # "wooden mountain cabin with chimney smoke",
        # "modern museum building with geometric shapes",
        # "abandoned farmhouse in rural field",
        # "colorful colonial houses in Latin America",
        # "mosque with detailed minarets, photographed at sunset",
        # "modern luxury villa with swimming pool, daylight",
        # "snow-covered alpine chalet with warm lights inside",
        # "old European train station with glass roof",
        # "skyscraper construction site with cranes",
        # "traditional Chinese pagoda surrounded by trees",
        # "apartment block with balconies, urban daylight",
        # "ancient Roman amphitheater ruins",
        # "small countryside church with stone walls"
    ],
    "landscape": [ # → 25 Prompts á 14 Varianten bzw. á 6 Varianten für unbekannten Datensatz
        "alpine mountain landscape at sunrise, snow peaks glowing",
        # "dense rainforest with mist, sunlight filtering through trees",
        # "desert dunes under clear blue sky, detailed sand textures",
        # "northern lights over snowy forest, night sky full of stars",
        # "coastal cliffs with crashing waves, overcast weather",
        # "savannah landscape with acacia trees, golden hour",
        # "autumn forest with colorful leaves, path covered in foliage",
        # "volcanic landscape with lava fields, dramatic sky",
        # "panoramic view of a glacier, icy textures",
        # "rolling green hills in spring, wildflowers in bloom",
        # "river winding through canyon, aerial perspective",
        # "tropical beach with palm trees, turquoise water",
        # "foggy moorland with rocky outcrops",
        # "frozen lake with snow-covered pine trees",
        # "desert oasis with palm trees and water pool",
        # "grassland with grazing animals, cloudy sky",
        # "mountain valley with river, summer daylight",
        # "dramatic thunderstorm over open fields",
        # "sunset over rice terraces, Southeast Asia",
        # "snowy mountain village, smoke from chimneys",
        # "desert plateau with cacti, midday sun",
        # "vast tundra under cloudy sky, lonely feeling",
        # "panoramic coastline with fishing boats, sunrise",
        # "lush jungle waterfall, spray in the air",
        # "rolling vineyards in summer, Mediterranean style"
    ],
}