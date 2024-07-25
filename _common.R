set.seed( 42 )

# R options set globally
# options( width = 60 )

# Activate crayon output
options(
  #crayon.enabled = TRUE,
  pillar.bold = TRUE,
  stringr.html = FALSE
)

# example chunk options set globally
knitr::opts_chunk$set(
    comment = "  ",
    collapse = TRUE,
    echo = TRUE, 
    message = FALSE, 
    warning = FALSE, 
    error = FALSE, 
    fig.align = 'center',
    fig.retina = 2,
    fig.width = 6,
    fig.asp = 2/3,
    fig.show = "hold"
  )

options( dplyr.print_min = 6, dplyr.print_max = 6 )

ggplot2::theme_set( ggplot2::theme( 
                            panel.background = ggplot2::element_rect( fill = "white", colour = "white", size = 0.5, linetype = "solid" ),
                            panel.grid.major = ggplot2::element_line( size = 0.2, linetype = 'solid', colour = "gray77" ), 
                            panel.grid.minor = ggplot2::element_line( size = 0.1, linetype = 'solid', colour = "gray90" ),
                            axis.text  = ggplot2::element_text( size = 11 ), 
                            axis.title = ggplot2::element_text( size = 12, face = "bold" ),
                            title = ggplot2::element_text( size = 14, face = "bold" )
                  ) )
