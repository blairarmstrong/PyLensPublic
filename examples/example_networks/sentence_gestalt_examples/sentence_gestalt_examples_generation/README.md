This directory is adapted from the repository for the simulations accompanying Chapter 10 of the *Computational Cognitive Neuroscience* textbook:  
[https://sims.compcogneuro.org/ch10/sg/](https://sims.compcogneuro.org/ch10/sg/)

O’Reilly, R. C., Munakata, Y., Frank, M. J., Hazy, T. E., & Contributors. (2024). *Computational Cognitive Neuroscience* (5th ed.). [https://compcogneuro.org/book](https://compcogneuro.org/book)

To generate the examples, run sg.go with `go run .`

* line 83 `const N = 10000` indicate the number of examples to be generated.
* To generate training data
    * uncomment line 68 `f, err := os.Create("sg_train_data.jsonl")` 
    * uncomment line 78 `ev := sim.Envs.ByMode(etime.Train).(*SentGenEnv)`
    * comment corresponding lines for testing data
* To generate testing data
    * uncomment line 70 `f, err := os.Create("sg_test_data.jsonl")`
    * uncomment line 80 `ev := sim.Envs.ByMode(etime.Test).(*SentGenEnv)`
    * comment corresponding lines for testing data

To convert the jsonl data into PyLens Format, run `create_pylens_input.py`

* To convert training data
    * uncomment line 160 `with open("sg_train_data.jsonl") as f:`
    * uncomment line 175 `with open("./sentence_gestalt_10000.ex", "w") as f:`
    * comment corresponding lines for testing data
* To convert testing data
    * uncomment line 161 `with open("sg_test_data.jsonl") as f:`
    * uncomment line 177 `with open("./sentence_gestalt_test.ex", "w") as f:`
    * comment corresponding lines for training data
