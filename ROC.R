
ROCMetrics <- R6::R6Class (
  'ROCMetrics',
  lock_objects = FALSE,
  
  private = list(
    
  ),
  
  public = list(
    initialize = function(y_true, p_pred) {
      self$y_true <- y_true
      self$p_pred <- p_pred
    },
    
    conf.matrix = function(y_pred) {
      conf <- caret::confusionMatrix(data = factor(y_pred), reference = factor(self$y_true))
      return(conf)
    },
    
    threshold.matrix = function(step_size) {
      if (step_size < 0 || step_size > 1) {
        stop('step_size must be a valid probability')
      }
      
      cols <- seq(0, 1, step_size)
      matr <- data.frame()
      
      for (i in cols) {
        y_pred <- ifelse(self$p_pred >= i, 1, 0)
        conf_matrix <- self$conf.matrix(y_pred = y_pred)
        temp_df <- t(data.frame(conf_matrix$byClass))
        row.names(temp_df) <- i
        matr <- rbind(matr, temp_df)
      }
      
      return(matr)
    },
    
    roc_plot = function(threshold_matrix, file_name = NULL) {
      fpr <- 1 - threshold_matrix[, 'Specificity']
      tpr <- threshold_matrix[, 'Sensitivity']
      auc <- pracma::trapz(fpr, tpr)
      
      plt <- ggplot2::ggplot(NULL, aes(x = fpr, y = tpr)) + 
        ggplot2::geom_line() +
        ggplot2::geom_area(fill = 'red', alpha = 0.5) +
        ggplot2::ggtitle(paste0('ROC Curve (AUC = ', round(auc, 4), ')')) +
        ggplot2::ylab('TPR') +
        ggplot2::xlab('FPR')
      
      if (!is.null(file_name)) {
        ggplot2::ggsave(file_name, dpi = 200)
      }
      
      return(plt)
    }
  )
)
