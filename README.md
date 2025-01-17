# MDAGCN

MDAGCN: predicting mutation-drug associations through signed graph convolutional networks via graph sampling



![](/MDAGCN-main/workflow.png)

 An overview of our proposed MDAGCN model

1) We generated an in-house benchmark data set of anti-cancer drug sensitivity/resistance mutations, promoting the research and development of computational methodology.
2) We proposed a GCN model based on signed graphs, named MDAGCN, which utilizes a graph sampler to extract subgraphs to predict mutation-drug associations with specific types.
3) We adopted the normalization techniques and a light-weight sampling algorithm, improving the model training accuracy.
4) We designed extensive experiments to show the proposed method has achieved better performance compared with state-of-the-art methods in predicting mutation-drug associations with types.



## Datasets
· D_SM.txt：drug-similarity matrix, which is calculated based on drug features.  
· M_SM.txt：mutation-similarity matrix, which is calculated based on mutation features.  
· drug_mutation_pairs.csv: the drug-mutation association network.  
· all mutation feature 661x248.csv: mutation feature matrix.

· all_drug_feature_184x231.csv: drug feature matrix.



## Run the MDAGCN
1. Dependency Installation

Download Required Dependencies
Use the following command to create a virtual environment and install dependencies via Conda:
conda env create -f environment.yml
Alternatively, you can manually install the required libraries based on the code.

2. Data Pre-Processing

Run the Data Pre-Processing Script
Execute Modified_Construct_MDP_graph.ipynb, located in the Data pre-processing folder.
For unbalanced data, modify the following line in the code:

Change: 
for isbalance in [True]: 
To:
for isbalance in [False]:

The output data will be saved in the balance folder. Rename the generated node_feature_label.csv file based on the type of data:

Balanced data: Rename to node_feature_label__balance.csv.
Unbalanced data: Rename to node_feature_label__nobalance.csv.

3. Finalizing Preprocessed Data

Run the Data Pre-Processing Scripts
Execute Modified_Construct_MDP_graph.ipynb located in the Data pre-processing folder.
If the output data generated by Modified_Construct_MDP_graph.ipynb matches the expected file names (e.g., node_feature_label__balance.csv for balanced data or node_feature_label__nobalance.csv for unbalanced data), you can directly execute Modified_2_data_preprocessing_to_model_type1.ipynb.
   
4. Train and test, for example:

python -m graphsaint.pytorch_version.train_saveys --data_prefix drug_mutation_data/task_Tp__testlabel0_3knn_edge_fold0  --train_config graphsaint/parameters_epoch_1.yml

***notes: Run the data using the terminal, ensuring that both the code and file paths are correct. Use the input data from the drug_mutation_data file, or, if necessary, generate your own data by running the 2_data_preprocessing_to_model_type1.ipynb script.


## Comparative experiments
machine learning file: includes code for decision trees (DTs), random forests (RF), and extremely random trees (ERTs).

state-of-the-art methods file: includes code for gcmc, SGCN, SNEA, TDRC, SGNNMD and NMCMDA.

In addition, the code for these comparison experiments is executable, ensuring the transparency of the experiments.


## Contact

Please feel free to contact us if you need any help: xiaosongwang@stu.ahau.edu.cn


