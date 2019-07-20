library(knitr)
opts_chunk$set(message = FALSE)

library(dplyr)
library(stringr)

url <- "http://www.amazon.com/ggplot2-Elegant-Graphics-Data-Analysis/product-reviews/0387981403/ref=cm_cr_dp_qt_see_all_top?ie=UTF8&showViewpoints=1&sortBy=helpful"

library(rvest)

h <- read_html(url)

url_base <- "http://www.amazon.com/ggplot2-Elegant-Graphics-Data-Analysis/product-reviews/0387981403/ref=undefined_2?ie=UTF8&showViewpoints=1&sortBy=helpful&pageNumber="
urls <- paste0(url_base, 1:5)
urls

library(stringr)

read_page_reviews <- function(url) {
  title <- h %>%
    html_nodes(".a-color-base") %>%
    html_text()
  
  format <- h %>%
    html_nodes(".review-text") %>%
    html_text()
  
  helpful <- h %>%
    html_nodes("#cm_cr-review_list .review-votes") %>%
    html_text() %>%
    str_extract("\\d+") %>%
    as.numeric()
  
  stars <- h %>%
    html_nodes("#cm_cr-review_list .review-rating") %>%
    html_text() %>%
    str_extract("\\d") %>%
    as.numeric()
  
  data_frame(title, format, stars, helpful)
}

ggplot2_reviews <- bind_rows(lapply(urls, read_page_reviews))

knitr::kable(ggplot2_reviews)