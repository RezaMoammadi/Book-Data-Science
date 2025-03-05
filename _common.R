set.seed(42)

# R options set globally
# options(width = 60)

# Any package that is required by the script below is given here:
inst_pkgs = load_pkgs =  c( "kableExtra", 
                            "DMwR2", 
                           "liver", 
                           "ggplot2", 
                           "plyr",    # for mutate function
                           "dplyr",   # for filter and between functions
                           "forcats", # for fct_collapse function
                           "Hmisc",   # for handling missing values
                           "naniar",  # for visualizing missing values
                           
                           "pROC", 
                           "neuralnet",
                           "psych", 
                           "rpart", 
                           "rpart.plot", 
                           "C50", 
                           "randomForest",
                           "naivebayes",
                           "factoextra")
inst_pkgs = inst_pkgs[!(inst_pkgs %in% installed.packages()[,"Package"])]
if (length(inst_pkgs)) install.packages(inst_pkgs)

# Dynamically load packages
pkgs_loaded = lapply(load_pkgs, require, character.only = TRUE)


# Activate crayon output
options(
  #crayon.enabled = TRUE,
  pillar.bold = TRUE,
  stringr.html = FALSE
)

# example chunk options set globally
knitr::opts_chunk$set(
    comment    = "  ",
    collapse   = TRUE,
    echo       = TRUE, 
    message    = FALSE, 
    warning    = FALSE, 
    error      = FALSE,
    fig.show   = 'hold',
    fig.align  = 'center',
    fig.retina = 2,
    out.width  = '70%', #fig.width  = 6,
    fig.asp    = 2/3
  )

options(dplyr.print_min = 6, dplyr.print_max = 6)

# for pdf output
options(knitr.graphics.auto_pdf = TRUE)

ggplot2::theme_set(ggplot2::theme( 
                            panel.background = ggplot2::element_rect(fill = "white", colour = "white", linewidth = 0.5, linetype = "solid"),
                            panel.grid.major = ggplot2::element_line(linewidth = 0.2, linetype = 'solid', colour = "gray77"), 
                            panel.grid.minor = ggplot2::element_line(linewidth = 0.1, linetype = 'solid', colour = "gray90"),
                            axis.text  = ggplot2::element_text(size = 11), 
                            axis.title = ggplot2::element_text(size = 12, face = "bold"),
                            title = ggplot2::element_text(size = 14, face = "bold"), 
                            plot.title = ggplot2::element_text(hjust = 0.5)
                  ))

# For LaTeX output
options(tinytex.verbose = TRUE)
