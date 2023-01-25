# landslidesusceptibility-haderetal2022
Hader et al (2022) - Landslide risk assessment considering socionatural factors: methodology and application to Cubatão municipality, São Paulo, Brazil
Published by Natural Hazards
https://doi.org/10.1007/s11069-021-04991-4
Programming language: R

######## Introduction ########
The script LS_RF_Haderetal2022.R produce the Landslide Susceptibility Map using machine learning of the article resulted from my Masters Degree. The article explains all the details about the method, analysis and specifications such as acronyms and symbols used that you need to know for reproducibility

######## Script LS_RF_Haderetal2022.R ########
The script is organised into sections and subsections as a maximum level (e.g: 1, 1.1), as well as in all the steps there are comments made for the compression of what was done.

######## Files ########
You need to download the LS_RF folder and put it in your C: directory for the code to run smoothly.

# Data folder --
There are three .csv files, the training and the testing data (already ramdomly splitted), and the colour palette for the final map. The training and testing data were ramdomly splitted in the QGIS software with the LaGriSU v 0.2 (Landslide Grid and Slope Units) tool (https://github.com/Althuwaynee/LaGriSU_Landslide-Grid-and-Slope-Units-QGIS_ToolPack)

# Raster folder --
In this folder you can find all the thematic maps (or independent variables) for the deployment of the model in .tif format

# Resampled folder --
In this folder you can find all the resampled raster by the smallest 
