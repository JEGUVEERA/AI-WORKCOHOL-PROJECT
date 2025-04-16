
import re
from collections import defaultdict
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType

load_dotenv()

# Initialize Gemini LLM
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.7)

positive_words = [
    "happy", "joy", "love", "great", "amazing", "wonderful", "brilliant", "cheerful", "delightful", "ecstatic",
    "welcoming", "wise", "witty", "worthy", "youthful", "zealous", "zesty", "good","happy", "joy", "love", "great", "amazing", "wonderful", "brilliant", "cheerful", "delightful", "ecstatic",
    "fantastic", "grateful", "harmonious", "inspiring", "jubilant", "kind", "lively", "marvelous", "optimistic",
    "peaceful", "radiant", "spectacular", "thriving", "uplifting", "victorious", "warmhearted", "zealous",
    "accomplished", "admirable", "affectionate", "authentic", "benevolent", "blessed", "bountiful", "buoyant",
    "calm", "charming", "compassionate", "confident", "courageous", "courteous", "dazzling", "dedicated",
    "determined", "dynamic", "eager", "effervescent", "elevated", "empowered", "enchanting", "energetic",
    "enthusiastic", "exceptional", "exuberant", "faithful", "flourishing", "forgiving", "friendly", "generous",
    "gentle", "genuine", "glorious", "graceful", "grounded", "hardworking", "helpful", "honest", "hopeful",
    "humble", "illustrious", "impressive", "independent", "ingenious", "innovative", "intelligent",
    "invigorating", "joyous", "jovial", "keen", "legendary", "limitless", "lovable", "magnificent",
    "mindful", "motivated", "noble", "nurturing", "open-minded", "outstanding", "passionate", "patient",
    "peace-loving", "persevering", "persistent", "philanthropic", "playful", "positive", "powerful",
    "proactive", "productive", "prosperous", "proud", "radiant", "refreshing", "reliable", "resilient",
    "resourceful", "respectful", "responsible", "rewarding", "satisfied", "selfless", "sensible",
    "sincere", "skillful", "soothing", "spirited", "spontaneous", "strong", "successful", "supportive",
    "sympathetic", "thoughtful", "tolerant", "trustworthy", "unwavering", "valiant", "vibrant", "wise",
    "witty", "youthful", "zesty", "adventurous", "affluent", "amiable", "artistic", "aspiring",
    "authentic", "balanced", "breathtaking", "bright", "captivating", "carefree", "celebratory",
    "chivalrous", "classic", "colorful", "compelling", "congenial", "content", "creative",
    "cultivated", "daring", "decisive", "dedicated", "delicate", "diligent", "distinguished",
    "divine", "effortless", "elegant", "elated", "eloquent", "empathic", "empowered",
    "enlightened", "enterprising", "expressive", "exquisite", "fascinating", "fearless",
    "fertile", "festive", "flawless", "fortunate", "free-spirited", "fun-loving", "generative",
    "genius", "glamorous", "glowing", "graceful", "groundbreaking", "handsome", "healing",
    "heartwarming", "heroic", "high-spirited", "hopeful", "hospitable", "humorous", "idealistic",
    "imaginative", "immaculate", "industrious", "influential", "insightful", "intuitive",
    "inventive", "jolly", "jubilant", "keen", "laudable", "lively", "loving", "loyal",
    "magical", "majestic", "masterful", "meditative", "mesmerizing", "meticulous",
    "mind-blowing", "miraculous", "motivational", "natural", "neat", "nurturing",
    "observant", "omniscient", "opulent", "orderly", "original", "outgoing",
    "outstanding", "passionate", "peaceful", "perceptive", "perseverant",
    "persistent", "philosophical", "playful", "poetic", "polished", "popular",
    "practical", "precious", "priceless", "profound", "progressive", "pure",
    "purposeful", "quick-witted", "radiant", "reassuring", "refined", "refreshing",
    "rejuvenating", "remarkable", "resilient", "resourceful", "respectable",
    "revered", "rewarding", "romantic", "sagacious", "sensational", "sensuous",
    "serene", "sharp", "shining", "skillful", "smart", "sociable", "soulful",
    "sparkling", "spectacular", "spontaneous", "steadfast", "stunning", "suave",
    "sublime", "successful", "sufficient", "superb", "supportive", "sweet",
    "sympathetic", "talented", "tenacious", "tender", "thrilled", "tidy",
     "transformative", "trustworthy", "truthful", "unconditional",
    "unfailing", "unique", "uplifted", "valiant", "versatile", "vibrant",
    "visionary", "vivacious", "warm", "welcoming", "wise", "witty", "wonderful",
    "worthy", "youthful", "zealous", "zesty","good", "happy", "joy", "love", "great", "amazing", "wonderful", "brilliant", "cheerful", "delightful", "ecstatic",
    "fantastic", "grateful", "harmonious", "inspiring", "jubilant", "kind", "lively", "marvelous", "optimistic",
]

negative_words = [
    "sad", "angry", "hate", "bad", "awful", "terrible", "horrible", "miserable", "depressed", "annoyed",
    "frustrated", "disappointed", "upset", "resentful", "unhappy", "gloomy", "hopeless", "anxious", "worried","sad", "angry", "hate", "bad", "awful", "terrible", "horrible", "miserable", "depressed", "annoyed",
    "frustrated", "disappointed", "upset", "resentful", "unhappy", "gloomy", "hopeless", "pessimistic",
    "anxious", "worried", "stressed", "fearful", "nervous", "jealous", "insecure", "guilty", "ashamed",
    "regretful", "lonely", "isolated", "betrayed", "rejected", "hurt", "humiliated", "embarrassed",
    "offended", "defensive", "irritated", "hostile", "vengeful", "rude", "arrogant", "selfish",
    "greedy", "manipulative", "deceitful", "insincere", "dishonest", "corrupt", "cruel", "cold",
    "insensitive", "callous", "apathetic", "neglectful", "inconsiderate", "ungrateful", "lazy",
    "incompetent", "careless", "reckless", "clumsy", "useless", "worthless", "pathetic",
    "pointless", "meaningless", "hopeless", "tragic", "painful", "brutal", "savage", "sinister",
    "evil", "wicked", "malicious", "vindictive", "spiteful", "destructive", "dangerous",
    "toxic", "poisonous", "contaminated", "dirty", "filthy", "polluted", "nasty", "disgusting",
    "repulsive", "rotten", "vile", "horrendous", "shameful", "unforgiving", "harsh", "unfair",
    "displeasing", "dismal", "insulting", "distasteful", "disastrous", "frightening", "dangerous",
    "painful", "grieving", "sorrowful", "unfortunate", "tragic", "mournful", "unpleasant",
    "toxic", "disorienting", "blameful", "condeyymning", "unjust", "mean", "difficult", "untrustworthy",
    "divisive", "angst", "struggling", "bitter", "suspicious", "hostile", "dark", "oppressive", "disturbing",
    "hateful", "alienated", "horrible", "apathetic", "ugly", "irritating", "disappointing", "low", 
    "mean-spirited", "untrustworthy", "horrific", "devastating", "vicious", "ugly", "dreadful", "abominable", 
    "unbearable", "unfortunate", "scary", "undesirable", "unwelcome", "unnecessary", "desolate", 
    "sickening", "appalling", "unreliable", "hateful", "aggressive", "tormenting", "abusive", 
    "discomforting", "dismaying", "untrusting", "paranoid", "disgusted", "haggard", "unworthy", 
    "sour", "suffocating", "discontent", "doubtful", "unmotivated", "neglected", "paralyzing", 
    "harrowing", "unjustified", "unsatisfying", "despondent", "revolting", "pitiful", "unhappy", 
    "disillusioned", "defeatist", "distressing", "hopelessness", "grievous", "apathetic", "dreary", 
    "frustrating", "dreadful", "complicated", "undeserving", "helpless", "downhearted", "suffocating", 
    "mournful", "unfavorable", "aggravating", "sickly", "damning", "reprehensible", "off-putting", 
    "counterproductive", "self-destructive", "unsympathetic", "uncooperative", "toxic", "unpredictable", 
    "unsuccessful", "opposing", "debilitating", "unattainable", "miserably", "hurtful", "demoralizing", 
    "distasteful", "ungracious", "unreceptive", "persecutory", "sabotaging", "irking", "paranoiac", 
    "pathetically", "disillusioned", "uninspiring", "unfitting", "unimpressive", "unhealthy", 
    "negative", "irritating", "broken", "regret", "unfulfilled", "degraded", "contradictory", 
    "depressing", "disconnected", "disheartening", "inferior", "intolerable", "vulgar", "morose", 
    "insufficient", "unfortunate", "oppressive", "hollow", "detrimental", "harsh", "frightening", 
    "grueling", "unwilling", "reprehensible", "unrelenting", "disturbing", "inflexible", "ruinous", 
    "deficient", "failing", "unethical", "unfulfilling", "hostile", "unjust", "destructive", 
    "disruptive", "worthless", "rejected", "downhearted", "resentful", "lacking", "resenting", 
    "oppositional", "obnoxious", "unappealing", "overbearing", "unforgiving", "pointless", 
    "insulting", "tragic", "imperfect", "wretched", "worrisome", "unfit", "discouraging", "dark", 
    "morbid", "regrettable", "rejected", "dismaying", "undesirable", "heartbreaking", "unsavory", 
    "undermining", "dejected", "despairing", "horrifying", "dread", "opposing", "unwanted", "unfocused", 
    "shocking", "grating", "unsuccessful", "compromised", "unworthy", "unpleasant", "terrifying", 
    "scornful", "intolerant", "ugly", "uncompromising", "disturbing", "discouraged", "exasperated", 
    "troublesome", "demotivating", "unapproachable", "unreliable", "distressed", "divisive", 
    "inconsiderate", "unwanted", "unsatisfactory", "destructive", "grievous", "hopeless", "shameful", 
    "pointless", "useless", "intolerable", "upsetting", "resenting", "repulsive", "bitter", "devastating",
    "discouraging", "unforgiving", "downcast", "unsuccessful", "ruining", "toxic", "draining", "stifling", 
    "conflicting", "distasteful", "unproductive", "blaming", "unsuitable", "tragically", "unfriendly",
    "infuriating", "agonizing", "unrecoverable", "unethical", "paralyzing", "unsolicited", "horrendous",
    "unsound", "unhelpful", "unresolvable", "failing", "unresolved", "lousy", "regretful"
]

# Emotion mapping
emotion = {
    "joy": 
    ["happy", "joy", "delighted", "cheerful", "smile", "grateful", "optimistic", "excited","happy", "joy", "delighted", "cheerful", "content", "smile", "grateful", "excited", "lively", "radiant","optimistic", "laugh", "gleeful", "blissful", "ecstatic", "jubilant", "elated", "satisfied", "thrilled","merry", "sunny", "glowing", "lighthearted", "playful", "uplifted", "thankful", "euphoric", "chipper","bubbly", "rejoicing", "exhilarated", "tickled", "overjoyed", "buoyant", "contented", "heartwarming","vivacious", "sparkling", "beaming", "giddy", "cheery", "zestful", "upbeat", "blithe", "perky", "radiating","joyous", "elation", "glee", "mirth", "jovial", "sunshine", "gratified", "serene", "peaceful", "fulfilled","enthusiastic", "eager", "spirited", "animated", "effervescent", "breezy", "carefree", "light-hearted","exultant", "gladsome", "jocund", "jolly", "jubilance", "merriment", "over the moon", "on cloud nine","walking on air", "in high spirits", "tickled pink", "chuffed", "blessed", "smiling", "laughing", "grinning","beaming", "radiating happiness", "cheerfulness", "good spirits", "high-spirited", "sunny disposition","positive vibes", "inner peace", "contentment", "satisfaction", "delight", "pleasure", "enjoyment","amusement", "rejoice", "celebration", "festive", "gladness", "happiness", "ecstasy", "euphoria","rapture", "bliss", "nirvana", "utopia", "heavenly", "paradise", "tranquility", "serenity", "calmness""composure", "relaxation", "ease", "comfort", "well-being", "balance", "harmony", "equanimity","gratitude", "appreciation", "thankfulness", "acknowledgment", "recognition", "praise", "admiration","respect", "esteem", "affection", "love", "fondness", "devotion", "adoration", "passion", "infatuation","romance", "tenderness", "warmth", "caring", "compassion", "empathy", "sympathy", "kindness","generosity", "altruism", "benevolence", "philanthropy", "humanity", "goodness", "virtue", "morality","ethics", "integrity", "honesty", "sincerity", "authenticity", "genuineness", "truthfulness", "openness","transparency", "clarity", "insight", "wisdom", "knowledge", "understanding", "awareness", "consciousness","mindfulness", "presence", "focus", "attention", "concentration", "engagement", "involvement", "participation","interaction", "communication", "connection", "relationship", "bond", "friendship", "companionship","partnership", "collaboration", "cooperation", "teamwork", "unity", "solidarity", "community", "society","culture", "tradition", "heritage", "history", "legacy", "memory", "nostalgia", "sentiment", "emotion","feeling", "sensation", "perception", "experience", "expression", "creativity", "imagination", "innovation","inspiration", "motivation", "aspiration", "ambition", "goal", "dream", "vision", "purpose", "meaning","significance", "value", "belief", "faith", "hope", "trust", "confidence", "courage", "bravery", "strength","resilience", "perseverance", "determination", "willpower", "discipline", "commitment", "dedication","loyalty", "devotion", "service", "duty", "responsibility", "accountability", "reliability", "dependability","consistency", "stability", "security", "safety", "protection", "support", "assistance", "help", "aid","relief", "comfort", "solace", "consolation", "encouragement", "reassurance", "hopefulness", "optimism","positivity", "cheerfulness", "joyfulness", "happiness", "delightfulness", "pleasure", "enjoyment","amusement", "entertainment", "fun", "laughter", "humor", "comedy", "joke", "wit", "banter", "jest","playfulness", "mischief", "prank", "trick", "surprise", "wonder", "awe", "amazement", "astonishment","marvel", "miracle", "magic", "fantasy", "dream", "imagination", "creativity", "art", "music", "dance","theater", "performance", "expression", "emotion", "feeling", "sensation", "perception", "experience","awareness", "consciousness", "mindfulness", "presence", "focus", "attention", "concentration","engagement", "involvement", "participation", "interaction", "communication", "connection", "relationship","bond", "friendship", "companionship", "partnership", "collaboration", "cooperation", "teamwork", "unity","solidarity", "community", "society", "culture", "tradition", "heritage", "history", "legacy", "memory","nostalgia", "sentiment", "emotion", "feeling", "sensation", "perception", "experience", "expression","creativity", "imagination", "innovation", "inspiration", "motivation", "aspiration", "ambition", "goal","dream", "vision", "purpose", "meaning", "significance", "value", "belief", "faith", "hope", "trust","confidence", "courage", "bravery", "strength", "resilience", "perseverance", "determination", "willpower","discipline", "commitment", "dedication", "loyalty", "devotion", "service", "duty", "responsibility","accountability", "reliability", "dependability", "consistency", "stability", "security", "safety",
    ],

    "frustration": 
    ["frustrated", "annoyed", "irritated", "upset", "discouraged", "fed up","frustrated", "annoyed", "irritated", "fed up", "disappointed", "exasperated", "upset", "stuck", "blocked",
    "bothered", "overwhelmed", "displeased", "discouraged", "aggravated", "tired", "burned out", "defeated",
    "powerless", "sigh", "ugh", "exhausted", "helpless", "inconvenienced", "unfulfilled", "restless",
    "miffed", "infuriated", "vexed", "pissed off", "grumpy", "disgruntled", "angsty", "bitter", "irate",
    "resentful", "fuming", "flustered", "short-tempered", "snappy", "moody", "cranky", "sour", "testy",
    "on edge", "tense", "uptight", "edgy", "troubled", "worn out", "mentally drained", "done with it",
    "ticked off", "enraged", "out of patience", "had enough", "about to snap", "on the verge", "choking on it",
    "boiling inside", "can’t take it anymore", "in turmoil", "emotionally drained", "in a bad mood",
    "blowing a fuse", "pulling my hair out", "in a rut", "feeling stuck", "backed into a corner", "clenched",
    "fed to the teeth", "wound up", "losing my cool", "losing patience", "irritated beyond belief",
    "emotionally blocked", "mentally blocked", "brain fog", "clouded mind", "out of sorts", "faintly enraged",
    "boiling over", "reaching my limit", "unsatisfied", "under pressure", "in a bind", "trapped", "stifled",
    "not in control", "feeling useless", "ignored", "not heard", "left behind", "pushed aside", "marginalized",
    "invisible", "torn", "conflicted", "unsure", "dissatisfied", "cynical", "jaded", "apathetic",
    "low motivation", "no energy", "dragging myself", "not again", "numb", "disheartened", "lethargic",
    "strained", "anxious", "twitchy", "itchy", "raw nerves", "high strung", "internally screaming",
    "screaming inside", "rage bubbling", "gritting teeth", "white-knuckling it", "barely holding on",
    "done dealing", "hitting a wall", "slammed", "frayed", "emotionally frayed", "overcapacity",
    "reached threshold", "meltdown mode", "temper rising", "no bandwidth", "saturated", "emotionally overloaded",
    "brain meltdown", "shutdown mode", "storming inside", "tornado in my head", "emotional clutter",
    "wired but tired", "mental gridlock", "jammed", "irritable", "whiny", "complaining", "nagged", "nagging feeling",
    "drained", "crushed under pressure", "blown gasket", "flicked off", "mad", "rattled", "snapped", "rageful",
    "in denial", "can't focus", "lost interest", "checked out", "tuned out", "coiled", "on the brink",
    "emotionally constipated", "banging my head", "run down", "out of gas", "emotionally stuck", "feeling useless",
    "repeating problems", "no resolution", "pointless", "hopeless", "circling thoughts", "looped in anger",
    "emotionally explosive", "silently suffering", "mental fatigue", "on my last nerve", "irritably impatient"
    ],

    "confidence": 
    ["confident", "bold", "capable", "assured", "certain", "secure","confident", "assured", "secure", "positive", "certain", "self-assured", "empowered", "bold", "capable", "unshakable", 
        "determined", "self-reliant", "assertive", "strong", "self-confident", "unwavering", "resolute", "brave", "daring", 
        "undaunted", "fearless", "unfaltering", "steady", "unperturbed", "self-sufficient", "self-belief", "decisive", 
        "composed", "unflappable", "optimistic", "reliable", "independent", "unmovable", "surefooted", "in control", 
        "stoic", "unrattled", "uncompromising", "unshaken", "persistent", "relentless", "intrepid", "assuredly", 
        "unhesitant", "firm", "dominant", "cool-headed", "self-reliant", "unwavering", "steady-handed", "proven", "proficient", 
        "successful", "commanding", "unconstrained", "trustworthy", "focused", "accomplished", "competent", "composed", 
        "unflinching", "audacious", "resilient", "goal-oriented", "fearless leader", "driven", "undeterred", "unshaken", 
        "accomplished", "capable", "empowered", "infallible", "masterful", "reliable", "steady", "positive-thinking", 
        "unrelenting", "decisive", "bold", "dominant", "self-assured", "at ease", "unintimidated", "self-reliant", 
        "unafraid", "in charge", "self-sustained", "unstoppable", "mentally strong", "forceful", "unbroken", "gutsy", 
        "stable", "mastermind", "in control", "determined", "motivated", "unbothered", "determined", "assertive"
        ],

    "sadness": 
    ["sad", "depressed", "unhappy", "heartbroken", "miserable", "gloomy","sad", "sorrow", "grief", "depressed", "unhappy", "heartbroken", "disappointed", "melancholy", "mournful","despairing", "gloomy", "downcast", "dismal", "woeful", "distressed", "anguished", "despondent","disheartened", "regretful", "lonely", "isolated", "betrayed", "hurt", "humiliated", "embarrassed","offended", "defensive", "irritated", "hostile", "vengeful","sad", "unhappy", "gloomy", "depressed", "sorrow", "cry", "tears", "melancholy", "blue", "heartbroken","miserable", "grief", "lonely", "downcast", "despair", "dismal", "forlorn", "woeful", "disheartened","hopeless", "bitter", "troubled", "crushed", "wailing", "mournful", "wistful", "regretful", "choked up","aching", "morose", "dejected", "low-spirited", "tearful", "dreary", "somber","sad", "unhappy", "gloomy", "depressed", "sorrow", "cry", "tears", "melancholy", "blue", "heartbroken","miserable", "grief", "lonely", "downcast", "despair", "dismal", "forlorn", "woeful", "disheartened","hopeless", "bitter", "troubled", "crushed", "wailing", "mournful", "wistful", "regretful", "choked up","aching", "morose", "dejected", "low-spirited", "tearful", "dreary", "somber", "sunken", "weeping","brokenhearted", "desolate", "bereft", "grieving", "lost", "pained", "withdrawn", "misery", "downhearted","hurting", "inconsolable", "lamenting", "distraught", "shattered", "pensive", "downbeat", "resigned","bleak", "dispirited", "heartsick", "meltdown", "afflicted", "isolated", "devastated", "longing","yearning", "tragic", "abandoned", "unloved", "pain-stricken", "anguish", "overwhelmed", "fragile","misunderstood", "neglected", "bruised", "aching heart", "emptiness", "unfulfilled", "hopelessness","cold", "silent tears", "lost hope", "aching soul", "guilt-ridden", "remorseful", "misfortune","shadowed", "suffering", "gone", "dull", "meltdown", "drowning", "numb", "ignored", "weary", "darkness","aching inside", "sighing", "depressing", "haunted", "suffocated", "anhedonia", "mourning", "hurt","downhill", "vanishing", "painful", "grief-stricken", "outcast", "confined", "abandoned", "let down",
     ],

    "anxiety": 
    ["nervous", "anxious", "worried", "uneasy", "restless", "panicked","nervous", "anxious", "worried", "uneasy", "tense", "scared", "panicked", "restless", "afraid", "nervy",
        "jittery", "sweating", "dizzy", "shaky", "apprehensive", "freaked out", "timid", "dread", "paranoid",
        "racing heart", "clammy", "numb", "insecure", "uneasiness", "phobic", "hypervigilant", "tight chest",
        "startled", "fearful", "jumpy", "trembling", "pacing", "choked up", "on edge", "worried sick",
        "panic attack", "suffocating", "shallow breathing", "inner turmoil", "overthinking", "mind racing",
        "rattled", "flighty", "churning stomach", "twitchy", "preoccupied", "sleepless", "doubtful",
        "overstimulated", "fidgety", "paralyzed", "excessive worry", "fear of the worst", "tight throat",
        "avoiding", "withdrawn", "blank mind", "worrywart", "obsessive thoughts", "breathless", "impending doom",
        "can’t focus", "can’t breathe", "inner chaos", "high strung", "jumping at sounds", "checking constantly",
        "edge of breakdown", "mental fog", "tension headache", "nagging fear", "inward collapse",
        "self-doubting", "dry mouth", "racing thoughts", "helpless", "tightness", "self-conscious", "pale", "sinking feeling",
        "mentally frozen", "stiff muscles", "avoidance", "overalert", "tight jaw", "grinding teeth", "ruminating",
        "frozen with fear", "muscle cramps", "internal panic", "drenched in sweat", "no appetite", "butterflies in stomach"
        ],

    "anger": 
    ["angry", "furious", "rage", "mad", "irritated", "hostile","angry", "furious", "rage", "mad", "irritated", "outraged", "annoyed", "fuming", "resentful", "agitated","enraged", "livid", "hostile", "bitter", "wrathful", "indignant", "cross", "infuriated", "seething","provoked", "irate", "disgusted", "boiling", "gritted teeth", "stormed", "snapped", "yelled", "barked","glared", "grumpy", "temper", "short-tempered", "pissed", "blazing", "vengeful", "bristling", "raging","slammed", "exploded", "kicked off", "boiling over", "red-faced", "gritting teeth", "hands clenched","raised voice", "overheated", "bursting", "spat", "hissed", "screamed", "howled", "temper tantrum","banged", "cussed", "cursed", "foul-mouthed", "irascible", "hot-headed", "snarled", "growled", "flared","retaliated", "lash out", "blurted", "snapped back", "stomped", "slammed door", "went off", "on edge","miffed", "peeved", "incensed", "fierce", "boiled", "uncontrollable", "displeased", "tantrum", "punching","threw fit", "sulking", "blow up", "ticked off", "blasted", "burning with anger", "cutting", "caustic","bitter tone", "ice cold", "snippy", "venomous", "sharp-tongued", "acidic", "sarcastic fury", "grumbling","frowned", "accusatory", "belligerent", "hostility", "verbal attack", "temperamental", "losing it","flipping out", "unhinged", "intolerant", "disrespected", "holding a grudge", "retorted", "revengeful","cold rage", "staring daggers", "puffed up", "heated", "prickly", "combative", "simmering", "mean","glowering", "spiteful", "cutting words", "feeling disrespected", "volatile", "boiling point", "snapping","retaliation", "mocking", "fighting words", "flustered", "quarrelsome", "sarcastic", "criticizing harshly","threatening", "punchy", "vengeance", "ranting", "flipped", "cracked", "temper spike", "testy", "grudgeful","pouting", "burning", "trembling with rage", "steaming", "snorting", "backbiting", "bashing", "judgmental","offended", "stiff tone", "sassy", "combustive", "screeching", "nagging", "disdainful", "sneering", "roared","grievance", "lecturing", "berating", "torqued", "scolding", "biting tone", "fury", "tantruming", "venom","explosive", "kicked things", "threw phone", "destroyed", "broke things", "flailing", "ruthless", "pushy","triggered", "militant", "retorted", "smacked", "shoved", "recoiled", "hit back", "defensive", "offensive","argumentative", "stubborn", "inflicted", "slapped", "grunted", "harrumphed", "rolled eyes", "mocked","destructive", "clenched jaw", "yanked", "snubbed", "slurred", "tensed", "pounded", "charged", "bashing","devastated (in fury)", "jeering", "accused", "boiling inside", "stinging words", "insulted", "cuss words","thundered", "grimaced", "soured", "bitter remarks", "violent", "stormed off", "threw hands", "clashed","clenched fists", "fuming silence", "biting sarcasm", "verbal abuse", "menacing", "scowling", "snide","brutal honesty", "retaliating", "punch lines", "criticized", "disrupted", "argued", "battling", "warring","tension", "belligerence", "hypercritical", "hot-tempered", "intense glare", "deep frown", "mouth tight","jaw locked", "vengeful stare", "fire in eyes", "angsty", "mean-spirited", "cold tone", "ice in voice","furious", "rage", "fury", "irritated", "hostile", "aggressive", "annoyed", "enraged", "outraged", "infuriated",
        "wrath", "irate", "livid", "incensed", "fuming", "seething", "heated", "mad", "exasperated", "offended", 
        "displeased", "provoked", "bitter", "vengeful", "spiteful", "disgusted", "scornful", "resentful", "agitated", 
        "unsettled", "volatile", "upset", "sulking", "incensed", "indignant", "unforgiving", "irritating", "disgruntled", 
        "cross", "in a huff", "infuriating", "on edge", "heated", "short-tempered", "ticked off", "mad as hell", "fury", 
        "boiling", "wrathful", "exasperating", "frustrating", "vexed", "blazing", "cranky", "ranting", "huffy", "stormy", 
        "angst", "spitting mad", "teed off", "snappy", "sore", "outrageous", "combative", "irritation", "stewing", "perturbed",
        "angry", "outraged", "tempestuous", "fiery", "displeasure", "perturbed", "wrathful", "abominable", "madder", 
        "unhappy", "impatient", "fiendish", "lacerating", "scalding", "turbulent", "malicious", "scathing", "bitterly", 
        "aggressive", "wrathfully", "agitated", "boiling over", "incensed", "savage", "revolting", "overcome with anger", 
        "hostility", "irritated beyond belief", "foul-mouthed", "vindictive", "caustic", "passionate", "irritable", 
        "vengeful", "irascible", "wrathful", "furiously", "snarling", "rage-fueled", "stirred up", "incendiary", "impeachable",
        "upset beyond words", "outraged", "fuming", "insulted", "tempestuous", "seething", "spiteful", "disrespected", 
        "reviled", "furiously", "full of rage", "burning", "intolerable", "infuriated beyond measure", "reckless", "unforgiving",
        "unrelenting", "unforgiving", "unpleasantly", "unruly", "unsettled", "untrusting", "untrustworthy",
        ],

    "sarcasm": 
    ["yeah right", "totally", "sure", "as if", "good luck with that","yeah, right", "sure, whatever", "oh, really?", "that’s just great", "totally", "as if", "seriously?", "sure thing", 
        "yeah, like that’s going to happen", "great idea", "just wonderful", "you don't say", "wow, amazing", "oh fantastic", 
        "oh joy", "sure, go ahead", "good luck with that", "right, I believe that", "unbelievable", "that's hilarious", 
        "you've got to be kidding", "very clever", "oh, how original", "who would’ve thought?", "good one", "totally believable", 
        "please, tell me more", "that’s cute", "that’s rich", "oh, how exciting", "big surprise", "very impressive", 
        "what a genius", "such a great idea", "right, okay", "yeah, I can see that", "oh, that’s going to be fun", 
        "couldn't agree more", "no, really?", "sure, you’re so right", "oh, how thoughtful", "really, wow", "you must be joking", 
        "please, tell me more", "oh, how thoughtful", "you really think so?", "wow, that's some idea", "what a clever observation", 
        "you're really on top of things", "yeah, because that’s going to work", "brilliant", "good one, Einstein", 
        "no kidding", "how impressive", "so subtle", "just what I needed", "I’m so impressed", "wow, just wow", 
        "don’t make me laugh", "sure, like that’s realistic", "oh, tell me more", "sure, tell me how that works out", 
        "how original", "that’s such a brilliant idea", "that’s what I was thinking", "well, aren’t you just clever?"
        ],

    "positive": positive_words,
    "negative": negative_words,
}

def analyze_sentiment_and_emotion(text: str) -> dict:
    text_lower = text.lower()
    words = set(re.findall(r'\b\w+\b', text_lower))
    emotion_counts = defaultdict(int)

    for emo_label, word_list in emotion.items():
        for word in word_list:
            if ' ' in word and word in text_lower:
                emotion_counts[emo_label] += 1
            elif word in words:
                emotion_counts[emo_label] += 1

    pos = emotion_counts.pop("positive", 0)
    neg = emotion_counts.pop("negative", 0)

    if pos > neg:
        sentiment = "Positive"
    elif neg > pos:
        sentiment = "Negative"
    elif pos == neg and pos > 0:
        sentiment = "Mixed"
    else:
        sentiment = "Neutral"

    emotion_label = max(emotion_counts.items(), key=lambda x: x[1], default=("Neutral", 0))[0]

    return {
        "sentiment": sentiment,
        "emotion": emotion_label,
        "counts": dict(emotion_counts)
    }

def generate_poetic_response(text: str) -> str:
    sentiment = analyze_sentiment_and_emotion(text)["sentiment"]
    prompt = f"The sentiment is {sentiment}. Create a poetic response to:\n{text}"
    return model.invoke([HumanMessage(content=prompt)]).content

# LangChain Agent
tools = [
    Tool(name="SentimentEmotionTool", func=analyze_sentiment_and_emotion, description="Analyze sentiment/emotion."),
    Tool(name="PoeticResponseTool", func=generate_poetic_response, description="Generate poetic response.")
]

agent = initialize_agent(
    tools=tools,
    llm=model,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False
)
