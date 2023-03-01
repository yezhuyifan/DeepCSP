# Organic Crystal Structure Prediction via Coupled Generative Adversarial Networks and Graph Convolutional Networks #  

## MolGAT for predicting crystal structure density. ##  
After installing the dependent packages, users can follow the usage instructions to validate the MolGAT performance and predict the crystal structure density for both marketed drugs and case study compounds.  

### Dependent packages: ###  
python == 3.6  
autograd == 1.2  
numpy == 1.19.2  
pandas == 1.1.2  
rdkit == 2018.03.4.0  
scikit-learn == 0.23.2  
six == 1.15.0  
tensorflow == 1.2.0  

### Usage instruction: ###  
•	Enter the current directory: cd ./deepcsp/  
•	Usage: python predict_test.py [Option]  
  
[Option]  
        -h, --help  --Usage instruction  
        -t, --test   --Model performance on test data  
        -d, --drug  --Predicting density for marketed drugs  
        -c, --case  --Predicting density for case study drugs  
  
Note: If the option is NULL, default value "--test" will be used.  
  
## DeepCSP for generating crystal structure structure. ##  
By implementing DeepCSP, users can generate crystal structures for marketed drugs, rank the drug crystal structures by the density-based method, and evaluate the accuracy of the generated crystal structures against their real counterparts.  
  
### Dependent packages: ###  
python == 3.6  
ctgan == 0.5.0  
numpy == 1.19.5  
pandas == 1.1.4  
rdt == 0.6.2  
scikit-learn == 0.24.2  
sdv == 0.13.1  
threadpoolctl == 3.0.0  
torch == 1.10.1  
tqdm == 4.62.3  
  
### Usage instruction: ###  
•	Enter the current directory:   
cd ./deepcsp/monomorph/ or cd ./deepcsp/polymorph/  
•	Usage:  
Generate crystal structures for monomorph or polymorph drugs: python generate.py.  
Evaluate the generated crystal structures: python evaluate.py.  
We have provided the real structures and generated crystal structures of the marketed drugs.  
 
