# Use the official R base image
FROM r-base:latest

# Install system dependencies if needed
RUN apt-get update && apt-get install -y \
    libcurl4-openssl-dev \
    libxml2-dev \
    libssl-dev \
    libgdal-dev

# Install necessary R packages
RUN R -e "install.packages(c('rgdal', 'raster', 'plyr', 'dplyr', 'RStoolbox', 'RColorBrewer', \
    'ggplot2', 'sp', 'caret', 'doParallel', 'randomForest', 'Information'), repos='https://cloud.r-project.org/')"

# Set the working directory inside the container
WORKDIR /usr/src/app

# Copy the R script into the container
COPY /Users/paulohader/Documents/data-science-projects/landslidesusceptibility-haderetal2022/ls_model_haderetal2022.R /usr/src/app/ls_model_haderetal2022.R

# Command to run your R script
CMD ["Rscript", "ls_model_haderetal2022.R"]
