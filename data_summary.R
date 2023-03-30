pacman::p_load(dplyr,
               ggplot2,
               ggthemes,
               skimr,
               gridExtra,
               Rmisc)

# import data
df <- read.csv(file = file.path('data', 'heart.csv'))
X <- select(df, -target)
y <- select(df, target)

# density plots
col_names <- names(X)
n_cols <- length(col_names)
plots <- list()

for (i in 1:n_cols) {
  col_ <- col_names[i]
  column <- sym(col_)

  plots[[col_]] <- ggplot(df, aes(x = !!column, fill = as.factor(target))) +
    geom_histogram(alpha = 0.5) +
    scale_fill_fivethirtyeight() +
    theme(legend.position = 'none')
}

multiplot(plotlist = plots, cols = 3)

