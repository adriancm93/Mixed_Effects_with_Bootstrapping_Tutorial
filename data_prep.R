library(purrr)
library(dplyr)

seasons <- 2000:2019
pbp <- purrr::map_df(seasons, function(x) {
  readRDS(
    url(
      glue::glue("https://raw.githubusercontent.com/guga31bb/nflfastR-data/master/legacy-data/play_by_play_{x}.rds")
    )
  )
})

pbp_mut<-pbp%>%
  dplyr::mutate(
    def_coach = if_else(defteam==home_team,home_coach,away_coach)
    ,
    pos_coach = if_else(posteam==home_team,home_coach,away_coach)
    ,
    home = if_else(posteam==home_team,1,0)
    ,
    wind = ifelse(is.na(wind)==T,min(wind,na.rm=T),wind)
    ,
    temp = ifelse(is.na(temp)==T,75,temp)
    ,
    opponent = paste(season.x,defteam,sep='_')
    ,
    team = paste(season.x,posteam,sep='_')
    ,
    player_id = passer_id
    ,
    season = as.factor(season.x)
  )  %>% 
  dplyr::filter(
    play_type == 'pass',
    game_type == 'REG',
    wp >= .20,
    wp <= .80,
    game_half != 'Overtime',
    season.x > 2005
  ) %>% select(epa,temp,wind,home,season,player_id,pos_coach,def_coach,team,opponent,wp)


