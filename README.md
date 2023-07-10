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



## Run the MDAGCN
1. pip install -r MDAGCN_requirements.txt  
2. 1_Construct MDP graph.ipynb
3. 2_data preprocessing to model.ipynb
4. Train and test, for example:
   python -m graphsaint.pytorch_version.train_saveys --data_prefix drug_mutation_data/task_Tp__testlabel0_7knn_edge_fold0  --train_config graphsaint/parameters_epoch_1.yml

## Experiment
machine learning file: includes code for decision trees (DTs), random forests (RF), and extremely random trees (ERTs).
state-of-the-art methods file: includes code for gcmc, SGCN, SNEA, TDRC, SGNNMD and NMCMDA.

In addition, the code for these comparison experiments is executable, ensuring the transparency of the experiments.


## Requirements

· MDAGCN is implemented to work under Python 3.7.0  
· torch==1.7.1  
· numpy==1.19.2  
· scipy==1.5.4  
· cython==0.29.2  
· pandas==1.1.5  

## Contact

Please feel free to contact us if you need any help: xiaosongwang@ahau.edu.cn


