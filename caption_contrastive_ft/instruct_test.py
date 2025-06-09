from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


#human_labels= ["Clymene Dolphin",        "Bottlenose Dolphin",   "Spinner Dolphin",         "Beluga, White Whale",      "Bearded Seal",         "Minke Whale",                  "Humpback Whale",           "Southern Right Whale",     "White-sided Dolphin",    "Narwhal",            "White-beaked Dolphin",         "Northern Right Whale", "Frasers Dolphin",      "Grampus, Rissos Dolphin",  "Harp Seal",                "Atlantic Spotted Dolphin",     "Fin, Finback Whale",       "Ross Seal",         "Rough-Toothed Dolphin", "Killer Whale", "Pantropical Spotted Dolphin", "Short-Finned Pacific Pilot Whale", "Bowhead Whale",       "False Killer Whale",   "Melon Headed Whale", "Long-Finned Pilot Whale",     "Striped Dolphin",         "Leopard Seal",      "Walrus",            "Sperm Whale",            "Common Dolphin"]
#scientific_labels = ['Stenella clymene', 'Tursiops truncatus',   'Stenella longirostris',   'Delphinapterus leucas',    'Erignathus barbatus',  'Balaenoptera acutorostrata',   'Megaptera novaeangliae',   'Eubalaena australis',      'Lagenorhynchus acutus',  'Monodon monoceros',  'Lagenorhynchus albirostris',   'Eubalaena glacialis', 'Lagenodelphis hosei',   'Grampus griseus',          'Pagophilus groenlandicus', 'Stenella frontalis',           'Balaenoptera physalus',    'Ommatophoca rossi', 'Steno bredanensis',     'Orcinus orca', 'Stenella attenuata',          'Globicephala macrorhynchus',       'Balaena mysticetus',  'Pseudorca crassidens', 'Peponocephala electra', 'Globicephala melas',       'Stenella coeruleoalba',   'Hydrurga leptonyx', 'Odobenus rosmarus', 'Physeter macrocephalus', 'Delphinus delphis']

human_labels =  ["Alder Flycatcher", "American Avocet", "American Bittern", "American Crow", "American Goldfinch", "American Kestrel", "Buff-bellied Pipit", "American Redstart", "American Robin", "American Wigeon", "American Woodcock", "American Tree Sparrow", "Anna's Hummingbird", "Ash-throated Flycatcher", "Baird's Sandpiper", "Bald Eagle", "Baltimore Oriole", "Sand Martin", "Barn Swallow", "Black-and-white Warbler", "Belted Kingfisher", "Bell's Sparrow", "Bewick's Wren", "Black-billed Cuckoo", "Black-billed Magpie", "Blackburnian Warbler", "Black-capped Chickadee", "Black-chinned Hummingbird", "Black-headed Grosbeak", "Blackpoll Warbler", "Black-throated Sparrow", "Black Phoebe", "Blue Grosbeak", "Blue Jay", "Brown-headed Cowbird", "Bobolink", "Bonaparte's Gull", "Barred Owl", "Brewer's Blackbird", "Brewer's Sparrow", "Brown Creeper", "Brown Thrasher", "Broad-tailed Hummingbird", "Broad-winged Hawk", "Black-throated Blue Warbler", "Black-throated Green Warbler", "Black-throated Grey Warbler", "Bufflehead", "Blue-grey Gnatcatcher", "Blue-headed Vireo", "Bullock's Oriole", "American Bushtit", "Blue-winged Teal", "Blue-winged Warbler", "Cactus Wren", "California Gull", "California Quail", "Cape May Warbler", "Canada Goose", "Canada Warbler", "Canyon Wren", "Carolina Wren", "Cassin's Finch", "Caspian Tern", "Cassin's Vireo", "Cedar Waxwing", "Chipping Sparrow", "Chimney Swift", "Chestnut-sided Warbler", "Chukar Partridge", "Clark's Nutcracker", "American Cliff Swallow", "Common Goldeneye", "Common Grackle", "Common Loon", "Common Merganser", "Common Nighthawk", "Northern Raven", "Common Redpoll", "Common Tern", "Common Yellowthroat", "Cooper's Hawk", "Costa's Hummingbird", "California Scrub Jay", "Dark-eyed Junco", "Double-crested Cormorant", "Downy Woodpecker", "American Dusky Flycatcher", "Black-necked Grebe", "Eastern Bluebird", "Eastern Kingbird", "Eastern Meadowlark", "Eastern Phoebe", "Eastern Towhee", "Eastern Wood Pewee", "Eurasian Collared Dove", "Common Starling", "Evening Grosbeak", "Field Sparrow", "Fish Crow", "Red Fox Sparrow", "Gadwall", "Grey-crowned Rosy Finch", "Green-tailed Towhee", "Eurasian Teal", "Golden-crowned Kinglet", "Golden-crowned Sparrow", "Golden Eagle", "Great Blue Heron", "Great Crested Flycatcher", "Great Egret", "Greater Roadrunner", "Greater Yellowlegs", "Great Horned Owl", "Green Heron", "Great-tailed Grackle", "Grey Catbird", "American Grey Flycatcher", "Hairy Woodpecker", "Hammond's Flycatcher", "European Herring Gull", "Hermit Thrush", "Hooded Merganser", "Hooded Warbler", "Horned Grebe", "Horned Lark", "House Finch", "House Sparrow", "House Wren", "Indigo Bunting", "Juniper Titmouse", "Killdeer", "Ladder-backed Woodpecker", "Lark Sparrow", "Lazuli Bunting", "Least Bittern", "Least Flycatcher", "Least Sandpiper", "LeConte's Thrasher", "Lesser Goldfinch", "Lesser Nighthawk", "Lesser Yellowlegs", "Lewis's Woodpecker", "Lincoln's Sparrow", "Long-billed Curlew", "Long-billed Dowitcher", "Loggerhead Shrike", "Long-tailed Duck", "Louisiana Waterthrush", "MacGillivray's Warbler", "Magnolia Warbler", "Mallard", "Marsh Wren", "Merlin", "Mountain Bluebird", "Mountain Chickadee", "Mourning Dove", "Northern Cardinal", "Northern Flicker", "Northern Harrier", "Northern Mockingbird", "Northern Parula", "Northern Pintail", "Northern Shoveler", "Northern Waterthrush", "Northern Rough-winged Swallow", "Nuttall's Woodpecker", "Olive-sided Flycatcher", "Orange-crowned Warbler", "Western Osprey", "Ovenbird", "Palm Warbler", "Pacific-slope Flycatcher", "Pectoral Sandpiper", "Peregrine Falcon", "Phainopepla", "Pied-billed Grebe", "Pileated Woodpecker", "Pine Grosbeak", "Pinyon Jay", "Pine Siskin", "Pine Warbler", "Plumbeous Vireo", "Prairie Warbler", "Purple Finch", "Pygmy Nuthatch", "Red-breasted Merganser", "Red-breasted Nuthatch", "Red-breasted Sapsucker", "Red-bellied Woodpecker", "Red Crossbill", "Redhead", "Red-eyed Vireo", "Red-necked Phalarope", "Red-shouldered Hawk", "Red-tailed Hawk", "Red-winged Blackbird", "Ring-billed Gull", "Ring-necked Duck", "Rose-breasted Grosbeak", "Rock Dove", "Rock Wren", "Ruby-throated Hummingbird", "Ruby-crowned Kinglet", "Ruddy Duck", "Ruffed Grouse", "Rufous Hummingbird", "Rusty Blackbird", "Sagebrush Sparrow", "Sage Thrasher", "Savannah Sparrow", "Say's Phoebe", "Scarlet Tanager", "Scott's Oriole", "Semipalmated Plover", "Semipalmated Sandpiper", "Short-eared Owl", "Sharp-shinned Hawk", "Snow Bunting", "Snow Goose", "Solitary Sandpiper", "Song Sparrow", "Sora", "Spotted Sandpiper", "Spotted Towhee", "Steller's Jay", "Swainson's Hawk", "Swamp Sparrow", "Swainson's Thrush", "Tree Swallow", "Trumpeter Swan", "Tufted Titmouse", "Tundra Swan", "Veery", "Vesper Sparrow", "Violet-green Swallow", "Warbling Vireo", "Western Bluebird", "Western Grebe", "Western Kingbird", "Western Meadowlark", "Western Sandpiper", "Western Tanager", "Western Wood Pewee", "White-breasted Nuthatch", "White-crowned Sparrow", "White-faced Ibis", "White-throated Sparrow", "White-throated Swift", "Willow Flycatcher", "Wilson's Snipe", "Wild Turkey", "Winter Wren", "Wilson's Warbler", "Wood Duck", "Woodhouse's Scrub Jay", "Wood Thrush", "American Coot", "Yellow-bellied Flycatcher", "Yellow-bellied Sapsucker", "Yellow-headed Blackbird", "Mangrove Warbler", "Myrtle Warbler", "Yellow-throated Vireo"]
scientific_labels = ['Empidonax alnorum', 'Recurvirostra americana', 'Botaurus lentiginosus', 'Corvus brachyrhynchos', 'Spinus tristis', 'Falco sparverius', 'Anthus rubescens', 'Setophaga ruticilla', 'Turdus migratorius', 'Mareca americana', 'Scolopax minor', 'Spizelloides arborea', 'Calypte', 'Myiarchus cinerascens', 'Calidris bairdii', 'Haliaeetus leucocephalus', 'Icterus galbula', 'Riparia', 'Hirundo rustica', 'Mniotilta varia', 'Megaceryle alcyon', 'Artemisiospiza belli', 'Thryomanes bewickii', 'Coccyzus erythropthalmus', 'Pica hudsonia', 'Setophaga fusca', 'Poecile atricapillus', 'Archilochus', 'Pheucticus melanocephalus', 'Setophaga striata', 'Amphispiza bilineata', 'Sayornis nigricans', 'Passerina caerulea', 'Cyanocitta', 'Molothrus ater', 'Dolichonyx', 'Chroicocephalus philadelphia', 'Strix varia', 'Euphagus', 'Spizella breweri', 'Certhia americana', 'Toxostoma rufum', 'Selasphorus platycercus', 'Buteo platypterus', 'Setophaga caerulescens', 'Setophaga virens', 'Setophaga nigrescens', 'Bucephala', 'Polioptila caerulea', 'Vireo solitarius', 'Icterus bullockii', 'Psaltriparus minimus', 'Spatula', 'Vermivora', 'Campylorhynchus brunneicapillus', 'Larus californicus', 'Callipepla californica', 'Setophaga tigrina', 'Branta canadensis', 'Cardellina canadensis', 'Catherpes mexicanus', 'Thryothorus ludovicianus', 'Haemorhous', 'Hydroprogne', 'Vireo cassinii', 'Bombycilla cedrorum', 'Spizella passerina', 'Chaetura pelagica', 'Setophaga pensylvanica', 'Alectoris chukar', 'Nucifraga columbiana', 'Petrochelidon pyrrhonota', 'Bucephala clangula', 'Quiscalus quiscula', 'Gavia immer', 'Mergus merganser', 'Chordeiles minor', 'Corvus corax', 'Acanthis flammea', 'Sterna hirundo', 'Geothlypis trichas', 'Astur cooperii', 'Calypte', 'Aphelocoma californica', 'Junco hyemalis', 'Nannopterum auritum', 'Dryobates pubescens', 'Empidonax oberholseri', 'Podiceps nigricollis', 'Sialia sialis', 'Tyrannus tyrannus', 'Sturnella', 'Sayornis phoebe', 'Pipilo erythrophthalmus', 'Contopus virens', 'Streptopelia decaocto', 'Sturnus', 'Coccothraustes vespertinus', 'Spizella pusilla', 'Corvus ossifragus', 'Passerella iliaca', 'Mareca strepera', 'Leucosticte tephrocotis', 'Pipilo chlorurus', 'Anas crecca', 'Regulus satrapa', 'Zonotrichia atricapilla', 'Aquila chrysaetos', 'Ardea herodias', 'Myiarchus crinitus', 'Ardea', 'Geococcyx californianus', 'Tringa melanoleuca', 'Bubo virginianus', 'Butorides', 'Quiscalus mexicanus', 'Dumetella carolinensis', 'Empidonax wrightii', 'Dryobates villosus', 'Empidonax hammondii', 'Larus argentatus', 'Catharus guttatus', 'Lophodytes cucullatus', 'Setophaga citrina', 'Podiceps auritus', 'Eremophila', 'Haemorhous', 'Passer domesticus', 'Troglodytes', 'Passerina cyanea', 'Baeolophus ridgwayi', 'Charadrius vociferus', 'Dryobates scalaris', 'Chondestes grammacus', 'Passerina amoena', 'Botaurus exilis', 'Empidonax minimus', 'Calidris minutilla', 'Toxostoma lecontei', 'Spinus psaltria', 'Chordeiles acutipennis', 'Tringa flavipes', 'Melanerpes lewis', 'Melospiza lincolnii', 'Numenius americanus', 'Limnodromus scolopaceus', 'Lanius ludovicianus', 'Clangula hyemalis', 'Parkesia motacilla', 'Geothlypis tolmiei', 'Setophaga magnolia', 'Anas', 'Cistothorus', 'Daucus carota', 'Sialia currucoides', 'Poecile gambeli', 'Zenaida macroura', 'Cardinalis cardinalis', 'Colaptes auratus', 'Circus hudsonius', 'Mimus polyglottos', 'Setophaga americana', 'Anas acuta', 'Spatula clypeata', 'Parkesia noveboracensis', 'Stelgidopteryx serripennis', 'Dryobates nuttallii', 'Contopus cooperi', 'Leiothlypis celata', 'Pandion haliaetus', 'Furnariidae', 'Setophaga palmarum', 'Empidonax difficilis', 'Calidris melanotos', 'Falco peregrinus', 'Phainopepla', 'Podilymbus', 'Dryocopus pileatus', 'Pinicola enucleator', 'Gymnorhinus', 'Spinus pinus', 'Setophaga pinus', 'Vireo plumbeus', 'Setophaga discolor', 'Haemorhous', 'Sitta pygmaea', 'Mergus serrator', 'Sitta canadensis', 'Sphyrapicus ruber', 'Melanerpes carolinus', 'Loxia curvirostra', 'Scolopendra', 'Vireo olivaceus', 'Phalaropus lobatus', 'Buteo lineatus', 'Buteo jamaicensis', 'Agelaius phoeniceus', 'Larus delawarensis', 'Aythya collaris', 'Pheucticus ludovicianus', 'Columba livia', 'Salpinctes obsoletus', 'Archilochus', 'Corthylio', 'Oxyura jamaicensis', 'Bonasa umbellus', 'Selasphorus', 'Euphagus', 'Artemisiospiza nevadensis', 'Oreoscoptes montanus', 'Passerculus sandwichensis', 'Sayornis saya', 'Piranga olivacea', 'Icterus parisorum', 'Charadrius semipalmatus', 'Calidris pusilla', 'Asio flammeus', 'Accipiter striatus', 'Calcariidae', 'Anser caerulescens', 'Tringa solitaria', 'Melospiza melodia', 'Artiodactyla', 'Actitis', 'Pipilo maculatus', 'Cyanocitta', 'Buteo swainsoni', 'Melospiza georgiana', 'Catharus ustulatus', 'Tachycineta bicolor', 'Cygnus buccinator', 'Baeolophus bicolor', 'Cygnus columbianus', 'Ardea ibis', 'Pooecetes gramineus', 'Tachycineta thalassina', 'Vireo gilvus', 'Sialia mexicana', 'Aechmophorus', 'Tyrannus verticalis', 'Sturnella', 'Calidris mauri', 'Piranga ludoviciana', 'Contopus sordidulus', 'Sitta carolinensis', 'Zonotrichia leucophrys', 'Plegadis chihi', 'Zonotrichia albicollis', 'Aeronautes saxatalis', 'Empidonax traillii', 'Gallinago delicata', 'Meleagris gallopavo', 'Troglodytes', 'Cardellina pusilla', 'Aix', 'Aphelocoma woodhouseii', 'Hylocichla mustelina', 'Fulica americana', 'Empidonax flaviventris', 'Sphyrapicus varius', 'Xanthocephalus', 'Setophaga petechia', 'Setophaga coronata coronata', 'Vireo flavifrons']


# scientific_labels =  ["Eleutherodactylus gryllus", "Eleutherodactylus brittoni", "leptodactylus albilabris", "Eleutherodactylus coqui", "Eleutherodactylus hedricki", "Dendroica angelae", "Melanerpes portoricensis", "Coereba flaveola", "Eleutherodactylus locustus", "Margarops fuscatus", "Loxigilla portoricensis", "Vireo altiloquus", "Eleutherodactylus portoricensis", "Megascops nudipes", "Eleutherodactylus richmondi", "Patagioenas squamosa", "Eleutherodactylus antillensis", "Turdus plumbeus", "Eleutherodactylus unicolor", "Coccyzus vieilloti", "Todus mexicanus", "Eleutherodactylus wightmanae", "Nesospingus speculiferus", "Spindalis portoricensis"]
# human_labels = ["Cricket coqui", "Grass coqui", "White-lipped Frog", "Common coqui", "Hedrick's coqui", "Elfin woods warbler", "Puerto Rican woodpecker", "Bananaquit", "Locust coqui", "Pearly-eyed thrasher", "Puerto Rican bullfinch", "Black-whiskered vireo", "Mountain Coqui", "Puerto Rican owl", "Bronze coqui", "Scaly-naped pigeon", "Red-eyed coqui", "Red-legged thrush", "Dwarf coqui", "Puerto Rican lizard cuckoo", "Puerto Rican tody", "Melodius coqui", "Puerto Rican tanager", "Puerto Rican spindalis"]
  
model = SentenceTransformer("Alibaba-NLP/gte-Qwen2-7B-instruct", trust_remote_code=True)
# In case you want to reduce the maximum length:
model.max_seq_length = 8192


# Get embeddings
#prompt = "Instruct: Given either the scientific or common name of an animal, retrive the corresponding name in the scientific format. \nQuery: "


#prompt = "Instruct: Given the name of an animal (either scientific or common), retrieve the corresponding alternative name (common or scientific).\nQuery: "


prompt = "Instruct: Retrieve the corresponding name (common or scientific) for the given animal name.\nQuery: "

human_embeddings = model.encode(human_labels, normalize_embeddings=True, prompt = prompt)
scientific_embeddings = model.encode(scientific_labels, normalize_embeddings=True, prompt = prompt)

# Cosine similarity matrix
similarity_matrix = cosine_similarity(human_embeddings, scientific_embeddings)


# Source to Target (common → scientific)
src2trg_preds = np.argmax(similarity_matrix, axis=1)
src2trg_acc = np.mean(src2trg_preds == np.arange(len(human_labels)))

# Target to Source (scientific → common)
trg2src_preds = np.argmax(similarity_matrix.T, axis=1)
trg2src_acc = np.mean(trg2src_preds == np.arange(len(scientific_labels)))

print(f"SRC2TRG Accuracy: {src2trg_acc:.2f}")
print(f"TRG2SRC Accuracy: {trg2src_acc:.2f}")


# print("\nMismatches (Common → Scientific):")
# for i, pred in enumerate(src2trg_preds):
#     if pred != i:
#         print(f"  Common: {human_labels[i]:35} → Predicted: {scientific_labels[pred]:25} (Correct: {scientific_labels[i]})")
        
# print("\nMismatches (Scientific → Common):")
# for i, pred in enumerate(trg2src_preds):
#     if pred != i:
#         print(f"  Scientific: {scientific_labels[i]:30} → Predicted: {human_labels[pred]:30} (Correct: {human_labels[i]})")


üü
import pandas as pd

df = pd.read_csv("Species.csv")
# Convert names to lists
common_names = df["Common name"].tolist()
scientific_names = df["Scientific name"].tolist()


from sentence_transformers import SentenceTransformer, util
import torch

# Encode common names (human labels) and scientific names (scientific labels)
human_embeddings = model.encode(common_names, normalize_embeddings=True, batch_size=64, prompt=prompt)
scientific_embeddings = model.encode(scientific_names, normalize_embeddings=True, batch_size=64, prompt=prompt)

# Compute cosine similarities
src2trg_scores = util.cos_sim(human_embeddings, scientific_embeddings)  # common -> scientific
trg2src_scores = util.cos_sim(scientific_embeddings, human_embeddings)  # scientific -> common

# Get top-1 matches for both directions
top_matches_src2trg = torch.argmax(src2trg_scores, dim=1)
top_matches_trg2src = torch.argmax(trg2src_scores, dim=1)

# Compute accuracies
correct_src2trg = sum(i == match.item() for i, match in enumerate(top_matches_src2trg))
correct_trg2src = sum(i == match.item() for i, match in enumerate(top_matches_trg2src))

accuracy_src2trg = correct_src2trg / len(common_names)
accuracy_trg2src = correct_trg2src / len(scientific_names)

print(f"Common to Scientific accuracy (src2trg): {accuracy_src2trg:.4f}")
print(f"Scientific to Common accuracy (trg2src): {accuracy_trg2src:.4f}")


from datasets import load_dataset, Dataset

dataset = load_dataset("davidrrobinson/AnimalSpeak", split="train")
dataset = dataset.filter(lambda x: x["species_common"] is not None and x["species_scientific"] is not None)
# Convert to DataFrame
df = dataset.to_pandas()

# Drop duplicates first by common name
df = df.drop_duplicates(subset="species_common")

# Then drop duplicates by scientific name (now from the reduced df)
df = df.drop_duplicates(subset="species_scientific")

# Back to HF Dataset
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.025, seed=42)
# Convert the split dataset back to pandas DataFrame

df = dataset['test'].to_pandas()


common_names = df["species_common"].tolist()
scientific_names = df["species_scientific"].tolist()


from sentence_transformers import SentenceTransformer, util
import torch

# Encode common names (human labels) and scientific names (scientific labels)
human_embeddings = model.encode(common_names, normalize_embeddings=True, batch_size=64, prompt=prompt)
scientific_embeddings = model.encode(scientific_names, normalize_embeddings=True, batch_size=64, prompt=prompt)

# Compute cosine similarities
src2trg_scores = util.cos_sim(human_embeddings, scientific_embeddings)  # common -> scientific
trg2src_scores = util.cos_sim(scientific_embeddings, human_embeddings)  # scientific -> common

# Get top-1 matches for both directions
top_matches_src2trg = torch.argmax(src2trg_scores, dim=1)
top_matches_trg2src = torch.argmax(trg2src_scores, dim=1)

# Compute accuracies
correct_src2trg = sum(i == match.item() for i, match in enumerate(top_matches_src2trg))
correct_trg2src = sum(i == match.item() for i, match in enumerate(top_matches_trg2src))

accuracy_src2trg = correct_src2trg / len(common_names)
accuracy_trg2src = correct_trg2src / len(scientific_names)

print(f"Common to Scientific accuracy (src2trg): {accuracy_src2trg:.4f}")
print(f"Scientific to Common accuracy (trg2src): {accuracy_trg2src:.4f}")
