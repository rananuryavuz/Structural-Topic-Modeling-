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

root						<- "C:/Users/The Soviet Unit/Documents/R/Predicting Likes/"
dataDir         <- paste0(root,"data", collapse = "")
outDir          <- paste0(root,"outputs", collapse = "")
utDir           <- paste0(root,"utilities",	collapse = "")
modelDir				<- paste0(root,"models", collapse = "")

rm("root")

setwd(dataDir); dset 							<- fread("dataset.csv")
setwd(dataDir); pic_topic_scores	<- fread("picture_topic_scores.csv")
setwd(dataDir); labels						<- fread("topic_labels.csv")
setwd(dataDir); it_post_labels		<- fread("it_post_labels.csv")
setwd(dataDir); messages					<- fread("messages.csv")
setwd(dataDir); it_sw							<- fread("it_stopwords.txt")

# separate the hashtags and the messages in a dedicated table, remove unprintable or unreadable characters
# if done previously, the messages file should already be available in the data directory

hashtag		<- "#[[:alnum:]]+"
badcoded	<- "<.+>"
url_http	<- "http[[:alnum:][:punct:]]+"
url_www		<- "www[[:alnum:][:punct:]]+"
messages	<- dset[,.(id,	language, brand, year_month, clean_message = str_replace_all(message, badcoded, ""))]
messages	<- messages[,hashtags := str_extract_all(clean_message, hashtag)]
messages	<- messages[,hashtags := lapply(hashtags, paste, collapse = " ")]
messages	<- messages[,clean_message := str_replace_all(clean_message, hashtag, "")]
messages  <- messages[,clean_message := str_replace_all(clean_message, url_http, "")]
messages  <- messages[,clean_message := str_replace_all(clean_message, url_www, "")]
messages	<- messages[,.(id, language, brand, year_month, clean_message = as.character(clean_message), hashtags = as.character(hashtags))]

messages[hashtags == "" | hashtags == "NA" | is.na(hashtags), hashtags := "none"]
messages[clean_message == "" | is.na(clean_message), clean_message := "none"]
messages[,has_tags := ifelse(hashtags != "none", 1, 0)]

setwd(dataDir); fwrite(messages, "messages.csv")
rm("hashtag","badcoded","url_http","url_www")

# create a corpus with a single document per brand, by:
# 	extracting a data.table from the dataset with columns 'brand' and 'general_concepts'
# 	grouping by brand
# 	summarising each group with a single entry per brand and a string of all general concepts
# 	transforming the resulting table into a corpus with 'general_concepts' as text field

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

# create a corpus of Italian stopwords and extract topics (then repeat for English)
# messages are grouped by month and brand, under the assumption that message themes
# are likely to vary on a 'per campaign basis', of which the month is a broad proxy

it_corpus <- messages[language == "it" & !is.na(clean_message)] %>%
	group_by(brand,year_month) %>%
	summarise(text = paste(clean_message, collapse = " ")) %>%
	ungroup()

en_corpus <- messages[language == "en" & !is.na(clean_message)] %>%
	group_by(brand,year_month) %>%
	summarise(text = paste(clean_message, collapse = " ")) %>%
	ungroup()

de_corpus <- messages[language == "de" & !is.na(clean_message)] %>%
	group_by(brand,year_month) %>%
	summarise(text = paste(clean_message, collapse = " ")) %>%
	ungroup()

it_corpus <- corpus(it_corpus, text_field = "text")
en_corpus <- corpus(en_corpus, text_field = "text")
de_corpus <- corpus(de_corpus, text_field = "text")

it_corpus <- textProcessor(
	it_corpus,
	metadata = docvars(it_corpus),
	stem = F,
	wordLengths = c(4,Inf),
	language = "it",
	customstopwords = it_sw$a
)

en_corpus <- textProcessor(
	en_corpus,
	metadata = docvars(en_corpus),
	stem = F,
	wordLengths = c(4,Inf),
	language = "en"
)

de_corpus <- textProcessor(
	de_corpus,
	metadata = docvars(de_corpus),
	stem = F,
	wordLengths = c(4,Inf),
	language = "de"
)

it_corpus <- prepDocuments(it_corpus$documents, it_corpus$vocab, meta = it_corpus$meta)
en_corpus <- prepDocuments(en_corpus$documents, en_corpus$vocab, meta = en_corpus$meta)
de_corpus <- prepDocuments(de_corpus$documents, de_corpus$vocab, meta = de_corpus$meta)

# search for optimal number of topics

model_search <- searchK(
	en_corpus$documents,
	en_corpus$vocab,
	K = seq(3,20),
	init.type = "Spectral"
)

plot.searchK(model_search)

# create topic model for Italian posts

it_posts_model <- stm(it_corpus$documents, it_corpus$vocab,	K = 50,	verbose = T, data = it_corpus$meta)
en_posts_model <- stm(en_corpus$documents, en_corpus$vocab,	K = 10,	verbose = T, data = en_corpus$meta)
de_posts_model <- stm(de_corpus$documents, de_corpus$vocab,	K = 7,	verbose = T, data = de_corpus$meta)

it_post_labels <- labelTopics(it_posts_model, n = 3)$frex %>%	data.frame()
en_post_labels <- labelTopics(en_posts_model, n = 3)$frex %>%	data.frame()
de_post_labels <- labelTopics(de_posts_model, n = 3)$frex %>%	data.frame()

it_post_labels$label <- NA
en_post_labels$label <- NA
de_post_labels$label <- NA

for(row in 1:length(it_post_labels$label)){
	it_post_labels$label[row] <- paste0(it_post_labels$X1[row], "_", it_post_labels$X2[row], "_", it_post_labels$X3[row])
}

for(row in 1:length(en_post_labels$label)){
	en_post_labels$label[row] <- paste0(en_post_labels$X1[row], "_", en_post_labels$X2[row], "_", en_post_labels$X3[row])
}

for(row in 1:length(de_post_labels$label)){
	de_post_labels$label[row] <- paste0(de_post_labels$X1[row], "_", de_post_labels$X2[row], "_", de_post_labels$X3[row])
}

it_post_labels[1:3] <- NULL
en_post_labels[1:3] <- NULL
de_post_labels[1:3] <- NULL

it_post_labels$label <- gsub("-","",it_post_labels$label)
en_post_labels$label <- gsub("-","",en_post_labels$label)
de_post_labels$label <- gsub("-","",de_post_labels$label)

setwd(dataDir); fwrite(it_post_labels, "it_post_labels.csv", row.names = F)
setwd(dataDir); fwrite(en_post_labels, "en_post_labels.csv", row.names = F)
setwd(dataDir); fwrite(de_post_labels, "de_post_labels.csv", row.names = F)

# create a corpus with the italian subset of the posts from the messages dataset,
# and 'clean_message' as text field then prep it, align the vocabulary to the model
# vocabulary, and assign topic probabilities

it_post_corpus <- corpus(messages[language == "it" & !is.na(clean_message)],text_field = "clean_message")
en_post_corpus <- corpus(messages[language == "en" & !is.na(clean_message)],text_field = "clean_message")
de_post_corpus <- corpus(messages[language == "de" & !is.na(clean_message)],text_field = "clean_message")

it_post_corpus <- textProcessor(it_post_corpus,	metadata = docvars(it_post_corpus),	stem = F,	wordLengths = c(4,Inf),	customstopwords = it_sw$a)
en_post_corpus <- textProcessor(en_post_corpus,	metadata = docvars(en_post_corpus),	stem = F,	wordLengths = c(4,Inf))
de_post_corpus <- textProcessor(de_post_corpus,	metadata = docvars(de_post_corpus),	stem = F,	wordLengths = c(4,Inf))

it_post_corpus <- alignCorpus(it_post_corpus,it_posts_model$vocab)
en_post_corpus <- alignCorpus(en_post_corpus,en_posts_model$vocab)
de_post_corpus <- alignCorpus(de_post_corpus,de_posts_model$vocab)

it_post_topics <- fitNewDocuments(it_posts_model,it_post_corpus$documents, newData = it_post_corpus$meta)
en_post_topics <- fitNewDocuments(en_posts_model,en_post_corpus$documents, newData = en_post_corpus$meta)
de_post_topics <- fitNewDocuments(de_posts_model,de_post_corpus$documents, newData = de_post_corpus$meta)

# extract the document - topic matrix, rename the columns, add document id's as
# column, then save the result

it_post_scores <- as.data.table(it_post_topics$theta)
en_post_scores <- as.data.table(en_post_topics$theta)
de_post_scores <- as.data.table(de_post_topics$theta)

colnames(it_post_scores) <- it_post_labels$label
colnames(en_post_scores) <- en_post_labels$label
colnames(de_post_scores) <- de_post_labels$label

it_post_scores[,id := it_post_corpus$meta$id]
en_post_scores[,id := en_post_corpus$meta$id]
de_post_scores[,id := de_post_corpus$meta$id]

fwrite(it_post_scores, "it_post_scores.csv")
fwrite(en_post_scores, "en_post_scores.csv")
fwrite(de_post_scores, "de_post_scores.csv")