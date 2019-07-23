library(knitr)
opts_chunk$set(message = FALSE)

library(dplyr)
library(stringr)

url <- "https://www.amazon.com/Harry-Potter-Chamber-Secrets-Rowling/dp/0439064872/"

library(rvest)

h <- read_html(url)


get_page_info <- function(url) {
  description <- h %>%
    html_nodes("#productTitle") %>% #still looking for the right css selector
    html_text()
  
  title <- h %>%
    html_nodes("#productTitle") %>%
    html_text()
  
  official_reviews <- h %>%
    html_nodes(".a-expander-partial-collapse-content") %>%
    html_text()

  official_reviews
}

result <- get_page_info(url)
result