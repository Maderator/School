read_data <- function(pth) {
  
  df_tmp <- read_csv(file.path(local_data_pth, pth))
  
  # set participant_id, trial id and stimulus id
  parsed_name <- pth %>% str_split("_") %>% unlist()
  df_tmp <- df_tmp %>% 
    mutate(participant_id = parsed_name[1],
           task_type = parsed_name[2])
  df_tmp %>% select(participant_id, task_type, everything())
}