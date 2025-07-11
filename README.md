# Dataset
Our datasets are available in the dataset/ folder.   
📦dataset   
 ┣ 📂cc_news   
 ┃ ┣ 📂graph   
 ┃ ┣ 📂text   
 ┃ ┗ 📂verbalized   
 ┣ 📂wikitext   
   ┣ 📂graph   
   ┣ 📂text   
   ┗ 📂verbalized   

The folder named `text/` contains the text data (original text documents and modified text documents)
The folder named `graph/` contains the knowledge graph (Each knowledge graph constructed from each text document in `text/` folder)
The folder named `verbalized/` contains the verbalized knowledge graph (Each verbalized knowledge graph constructed from each text knowledge graph in `graph/` folder)

In each folder, `text/`, `graph/`, and `verbalized/` folder, documents (text or KG) are classified by modification options as follows   
   
📂synonym_replacement   
┣ 📂0.3   
┗ 📂0.6   
📂context_replacement   
┣ 📂0.3   
┗ 📂0.6   
📂dipper_paraphraser   
┣ 📂60_0   
┗ 📂60_20   

Each text document in `text/`, and `verbalized/` folder are stored in each `.txt` files and each knowledge graph in `graph/` are stored in each `.json` files.


# Dataset generation
You can generate the dataset with your own text documents with the codes in the dataset_generation folder.
The folder structure is as follows:   
📦dataset_generation   
 ┣ 📂kg-construction   
 ┃ ┣ 📜kg-construction.py   
 ┃ ┣ 📜kg-tidy.py   
 ┃ ┗ 📜kg-verification.py   
 ┣ 📂load_text   
 ┃ ┣ 📜cc-news.py   
 ┃ ┗ 📜wikitext.py   
 ┣ 📜text-modification.py   
 ┗ 📜verbalize-kg.py   

You can generate your own data with the following process:
1. **Load your text data**  
   Sample code is available in the `load_text/` folder.

2. **Make modification of your text documents**  
   Run `text-modification.py`.

3. **Build the knowledge graphs** with the scripts in `kg-construction/`  

   2.1&nbsp;&nbsp;Run `kg-construction.py` to create a knowledge-graph file for every document in `text_documents/`.

   2.2&nbsp;&nbsp;Verify each graph by running `kg-verification.py`.

   2.3&nbsp;&nbsp;If any graph fails verification, run `kg-tidy.py` to repair it, then execute `kg-verification.py` again.

   2.4&nbsp;&nbsp;If a graph **still** fails verification, isolate only the affected documents and rerun `kg-construction.py` on that subset.

4. **Make verbalized documents of each knowledge graph** 
   Run `verbalize-kg.py`

# Similarity score calculation
📦scoring   
 ┣ 📂_utils   
 ┃ ┗ 📜ingram_utils.py   
 ┣ 📜SBERT-scoring.py   
 ┣ 📜WE-pretrained-scoring.py   
 ┣ 📜WE-scoring.py   
 ┣ 📜base-kernel-scoring.py   
 ┣ 📜ingram-scoring-ent.py   
 ┣ 📜ingram-scoring.py   
 ┣ 📜kge-scoring-ent.py   
 ┣ 📜kge-scoring.py   
 ┗ 📜wl-kernel-scoring.py   

All the similarity score calculation methods used in our experiments are implemented in `scoring/` folder. 
Run one of the python code in `scoring/` folder. The similarity scores are then calculated and will be recorded as json file in the folder named `Result/`.

# Evaluation of similarity score methods
📦evaluation   
 ┣ 📜metric.py   
 ┣ 📜ranking.py   
 ┗ 📜result_display.py   

After you run each scoring methods, then you can evaluate the performance of each scoring methods based in the results recorded in `Result/` folder by following process:
1. Run `ranking.py` in the `evaluation/` folder
2. Run `metric.py` in the `evaluation/` folder
After this process, the performance results (metric values) of each methods on each setting with each dataset are recorded in the folder named `Metric_Results/`
If you want to display those results in a single text files, run `python3 evaluation\result_display.py > ~~.txt`

