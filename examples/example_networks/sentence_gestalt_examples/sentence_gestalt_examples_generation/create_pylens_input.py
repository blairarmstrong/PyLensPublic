import numpy as np
import json
import pandas as pd

SGWords = [
	"consumed",
	"ate",
	"drank",
	"stirred",
	"spread",
	"kissed",
	"gave",
	"hit",
	"threw",
	"drove",
	"shed",
	"rose",
	"someone",
	"adult",
	"child",
	"dog",
	"busdriver",
	"teacher",
	"schoolgirl",
	"pitcher",
	"spot",
	"something",
	"food",
	"steak",
	"soup",
	"icecream",
	"crackers",
	"jelly",
	"icedtea",
	"koolaid",
	"utensil",
	"spoon",
	"knife",
	"finger",
	"kitchen",
	"livingroom",
	"park",
	"bat",
	"ball",
	"bus",
	"fur",
	"gusto",
	"pleasure",
	"daintiness",
	"with",
	"in",
	"to",
	"by",
	"was",
	"start",
]


SGRoles = [
	"Agent",
	"Action",
	"Patient",
	"Instrument",
	"CoAgent",
	"CoPatient",
	"Location",
	"Adverb",
	"Recipient",
]


SGFillers = [
	"Consumed",
	"Ate",
	"Drank",
	"Stirred",
	"Spread",
	"Kissed",
	"Gave",
	"Hit",
	"ThrewTossed",
	"ThrewHosted",
	"DroveTrans",
	"DroveMotiv",
	"ShedV",
	"ShedN",
	"RoseV",
	"RoseN",
	"Someone",
	"Adult",
	"Child",
	"Dog",
	"Busdriver",
	"Teacher",
	"Schoolgirl",
	"PitcherPers",
	"PitcherCont",
	"Spot",
	"Something",
	"Food",
	"Steak",
	"Soup",
	"Icecream",
	"Crackers",
	"Jelly",
	"Icedtea",
	"Koolaid",
	"Utensil",
	"Spoon",
	"Knife",
	"Finger",
	"Kitchen",
	"Livingroom",
	"Park",
	"BatBall",
	"BatAnim",
	"BallSphere",
	"BallParty",
	"Bus",
	"Fur",
	"Gusto",
	"Pleasure",
	"Daintiness",
	"None",
	"Ex1", # adding extras so it has 55 units to match layer size
	"Ex2",
	"Ex3",
]

word2idx   = {w:i for i, w in enumerate(SGWords)}
role2idx   = {r:i for i, r in enumerate(SGRoles)}
filler2idx = {f:i for i, f in enumerate(SGFillers)}

def lookup(i, j, k, l): 
    return SGWords[i], SGRoles[j], SGFillers[k], SGFillers[l]

def encode_query(word, role, filler):
    word_vec   = np.zeros(len(SGWords), dtype=np.float32)
    role_vec   = np.zeros(len(SGRoles), dtype=np.float32)
    filler_vec = np.zeros(len(SGFillers), dtype=np.float32)

    word_vec[word2idx[word]]     = 1.0
    role_vec[role2idx[role]]     = 1.0
    filler_vec[filler2idx[filler]] = 1.0
    return word_vec, role_vec, filler_vec

def lens_encode(word, role, filler):
    word_vector, role_vector, filler_vector = encode_query(word, role, filler)

    def format_vector(vector, group=None):
        body = " ".join(map(str, vector.astype(int)))
        return f"{{{group}}} {body}" if group else body

    input_line = "I: " + format_vector(word_vector, "word") + " " + format_vector(role_vector, "role")
    target_line = "T: " + format_vector(filler_vector)

    return input_line, target_line


# with open("sg_train_data.jsonl") as f:
with open("sg_test_data.jsonl") as f:
    data = [json.loads(line) for line in f]

df = pd.DataFrame(data)

# Each row has query_sequence = list of [word, role, filler, status]
# Expand so that each element becomes its own row
df_exp = df.explode("query_sequence").reset_index(drop=True)
df_exp[["word", "role", "filler", "status"]] = pd.DataFrame(
    df_exp["query_sequence"].tolist(), index=df_exp.index
)
df_exp = df_exp.drop(columns=["query_sequence"])

## training_data
with open("./sentence_gestalt_10000.ex", "w") as f:
## test_data
# with open("./sentence_gestalt_test.ex", "w") as f:
    for sentence_id, group in df_exp.groupby("idx"):
        # header = number of rows in this sentence
        f.write(f"{len(group)}\n")
        for j, row in enumerate(group.itertuples(index=False)):
            input_line, target_line = lens_encode(row.word, row.role, row.filler)
            f.write(f"[{j}]\n")
            f.write(input_line + "\n")

            # add semicolon for the last word in this sentence
            if j == len(group) - 1:
                f.write(target_line + ";\n")
            else:
                f.write(target_line + "\n")
        f.write("\n")  # blank line between sentences




