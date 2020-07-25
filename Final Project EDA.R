library(tidyverse)
library(ggcorrplot)
library(knitr)

wine <- read_delim('winequality-red.csv', delim = ';')

colnames(wine)

#Creating data format suited for multiple histograms
hist_data <- wine %>% gather()

#Creating histograms for each variable
ggplot(hist_data, aes(x = value)) + 
  geom_histogram(fill = 'darkred', bins = 20) +
  facet_wrap(~key, scales = c('free_x'))

#Creating data format suited for multiple scatterplots
scatter_data <- wine %>% gather(key = 'variable', value = 'value', -quality)

#Creating scatterplots of each variable
ggplot(scatter_data, aes(x = value, y = quality)) +
  geom_jitter(color = 'darkred', alpha = .15) +
  facet_wrap(~variable, scales = 'free_x')

#Creating correlation matrix of variables
ggcorrplot(cor(wine), type= 'upper', title = 'Correlation Matrix of Wine Data', legend.title = 'Correlation', 
           lab = TRUE, lab_size = 3, outline.color = 'black')
