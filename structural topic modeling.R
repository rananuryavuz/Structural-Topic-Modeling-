library(tm)
library(quanteda)
library(topicmodels)
library(ldatuning)
library(tidytext)
library(plyr)
library(data.table)
library(jsonlite)
library(rpart)
library(rpart.plot)
library(RColorBrewer)
library(tidyverse)
library(randomForest)
library(caret)
library(slam)
library(stm)
library(mgcv)
library(stringr)

root            <- 'your path'
dataDir         <- paste0(root,"data", collapse = "")
outDir          <- paste0(root,"outputs", collapse = "")
utDir           <- paste0(root,"utilities",	collapse = "")
modelDir	<- paste0(root,"models", collapse = "")

rm("root")

setwd(dataDir); dset 	        <- fread("dataset.csv")
setwd(dataDir); pic_topic_scores<- fread("picture_topic_scores.csv")
setwd(dataDir); labels	        <- fread("topic_labels.csv")
setwd(dataDir); it_post_labels	<- fread("it_post_labels.csv")
setwd(dataDir); messages        <- fread("messages.csv")
setwd(dataDir); it_sw		<- fread("it_stopwords.txt")

# separate the hashtags and the messages in a dedicated table, remove unprintable or unreadable characters
# if done previously, the messages file should already be available in the data directory



messages[hashtags == "" | hashtags == "NA" | is.na(hashtags), hashtags := "none"]
messages[clean_message == "" | is.na(clean_message), clean_message := "none"]
messages[,has_tags := ifelse(hashtags != "none", 1, 0)]

#       create a corpus with a single document per brand, by:
# 	grouping by brand
# 	summarising each group with a single entry per brand and a string of all general concepts


#       creating model 
brand_corpus <- dset[,.(brand,general_concepts)] %>%
  group_by(brand) %>%
  summarise(concepts = paste0(general_concepts,collapse = " ")) %>%
  corpus(text_field = "concepts")

# textProcessor converts the corpus into an stm object
# prepDocuments filters less frequent terms

meta <- docvars(brand_corpus)

brand_corpus <-	textProcessor(
		brand_corpus,
		metadata = meta,
		removestopwords = F,
		removenumbers = F,
		removepunctuation = F,
		stem = F
	)

brand_corpus <- prepDocuments(brand_corpus$documents,brand_corpus$vocab)

docs <- brand_corpus$documents
vocb <- brand_corpus$vocab
meta <- brand_corpus$meta

# search optimal number of topics - use only the first time, then store the number
# of topics in the follow up modeling function

# topic_search <- searchK(docs, vocb, K = c(2,5,10,15,20,25,30,50,75,100))

# model topics, input previously selected number of topics as K

topic_model <- stm(
										docs,
										vocb,
										K = 30,
										verbose = T,
										data = meta
									 )

# label the topics with the first 3 terms for each according to the FREX metric
# create a vector which holds the names for later labeling of variables

labels <- labelTopics(topic_model, n = 3)$frex %>%
	data.frame()

labels$label <- NA

for(row in 1:length(labels$label)){
	labels$label[row] <- paste0(labels$X1[row], "_", labels$X2[row], "_", labels$X3[row])
}

labels[1:3] <- NULL
labels$label <- gsub("-","",labels$label)


setwd(dataDir); fwrite(labels, "topic_labels.csv", row.names = F)

term_topics <- topic_model$beta %>%
	data.frame() %>% t() %>% data.table()
colnames(term_topics) <- labels$label

# create a corpus with the whole dataset using general_concepts as text field
# then prep it, align the vocabulary to the model vocabulary, and assign
# topic probabilities

pic_corpus <- corpus(dset,text_field = "general_concepts")

pic_corpus <- textProcessor(
	pic_corpus,
	metadata = docvars(pic_corpus),
	lowercase = T,
	removestopwords = F,
	removenumbers = F,
	removepunctuation = F,
	stem = F
)
pic_corpus <- alignCorpus(pic_corpus,topic_model$vocab)

pic_topics <- fitNewDocuments(topic_model,pic_corpus$documents, newData = pic_corpus$meta)

# extract the document - topic matrix, rename the columns, add document id's as
# column, then save the result

pic_topics_scores <- as.data.table(pic_topics$theta)
colnames(pic_topics_scores) <- labels$label
pic_topics_scores[,id := pic_corpus$meta$id]

fwrite(pic_topics_scores, "picture_topic_scores.csv")

