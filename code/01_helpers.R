show_top <- function(df, var, n=100) {
  df %>% count({{ var }}) %>% arrange(desc(n)) %>% print(n=n)
}
