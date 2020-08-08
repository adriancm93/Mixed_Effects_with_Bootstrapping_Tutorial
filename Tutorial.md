# Mixed Effects Modeling with Bootstrapping Tutorial
 Exercise to leverage re-sampling (bootstrapping) and Mixed Effects modeling for NFL analysis.

*The purpose of this is to show how to leverage re-sampling (bootstrapping) and Mixed Effects modeling for NFL analysis. The goal is to determine which factor is the most important in passing offense: QB ability, HC competence, Opposing HC competence or Supporting Cast*. 

**Goal statement, and variables**

The goal is to determine the most valuable aspect in passing offense by finding the distribution of theta (θ) for the random effect variables in the sample when trying to predict EPA. 

Theta is a vector of the random-effects parameter estimates: these are parameterized as the relative Cholesky factors of each random effect term (from R getME documentation).

My buddy **Parker** @statsowar gave me the following description in "super lame terms" as I asked him to do: "Cholesky just breaks down the R^2 and attributes it to different sources".

In other words, we will use these coefficients to measure the absolute impact each variable has on passing EPA (how much variance it explains).

*Parker gave me very good feedback and suggestions for future research and to improve my model, which I'll discuss at the end. Yes, **Parker** is a genius* 

**Model specs:**

We will be controlling for:
- Temperature
- Wind speed
- Win Probability at the time of the play
- Home/away game for the possession team
- And adjusting for inflation (Season)

Random effects:
- Player id (QB)
- Possession HC
- Opposing HC
- Opposing defensive team*Season (a proxy for opposing defensive roster)
- Possession team*Season (a proxy for offensive supporting cast)

**Limitations**

A couple of limitations with computer power and time will be discussed further.

The main limitation is the lack of play-caller data (is it the Offensive Coordinator callimg the plays or ia it the Head Coach?. As of now, most probably the OC is being captured in our proxy for supporting cast, which is problematic.

There is probably a similar problem with lack of Defensive play-caller (Defensive Coordinator).

**Packages**

```
library(lme4)
library(dplyr)
library(plyr)
library(ggplot2)
library(ggthemes)
library(RColorBrewer)
library(extrafont)
loadfonts(device = "win")
library(parallel)
set1<-brewer.pal(n = 9, name = "Set1") #This is to see color codes
```

**Pre-processing**

Look at data_prep.R file to see which variables I created and how I filtered data. The goal here was to make the sample as small as possible since bootstrapping is extremely computationally demanding and I didn't have two days to spare.

**Load data**

```
pbp_mut <- readRDS(url("https://raw.githubusercontent.com/adriancm93/Mixed_Effects_with_Bootstrapping_Tutorial/master/data/pbp_mut.RDS"))
```

**Mixed-Effects Model with full data**

Here we will be using multiple random effects ```(1|variable)```. It is possible to make these variables interact (one of Parker's recommendations which I'll be doing in the near future) this way: ```(1|player_id::pos_coach)```.
We will be using ```getME("theta")``` to get theta (θ) of the random effects. Remember, each simulation will have a different theta (due re-sampling) and then we will plot the distribution of these. ```fixef()``` is used to retrieve the estimates of the fixed effects (beta: β).

```
#Mixed model 
mixed_model<-pbp_mut %>% 
  lmer(formula=
         epa ~
         temp +
         wind + 
         home +
         wp +
         as.factor(season) +
         (1|player_id)+
         (1|pos_coach)+
         (1|team)+
         (1|def_coach)+
         (1|opponent)
       ,
       control=lmerControl(optimizer="bobyqa",
                           optCtrl=list(maxfun=2e5)))
#Summary
mixed_model %>% summary()

getME(mixed_model, "theta") #Research Cholesky factors | try ?getME
fixef(mixed_model)
```
For the next section, I followed this case study: https://stats.idre.ucla.edu/r/dae/mixed-effects-logistic-regression/ 
This amazing tutorial will explain everything about logistic mixed-effects models, and it has a section to explain how bootstrapping works. I highly recommend reading it. It is a tutorial for logistic mixed models, but it works well for our linear model as well. 

**Create re-sampling function and re-sample***

To bootstrap multi-level models, "We start by resampling from the highest level, and then stepping down one level at a time". as UCLA tutorial explains.

For that, we are using code from the amazing Biostatistics Department at Vanderbilt: http://biostat.mc.vanderbilt.edu/wiki/Main/HowToBootstrapCorrelatedData

```
resampler <- function(dat, clustervar, replace = TRUE, reps = 1) {
  cid <- unique(dat[, clustervar[1]])
  ncid <- length(cid)
  recid <- sample(cid, size = ncid * reps, replace = TRUE)
  if (replace) {
    rid <- lapply(seq_along(recid), function(i) {
      cbind(NewID = i, RowID = sample(which(dat[, clustervar] == recid[i]),
                                      size = length(which(dat[, clustervar] == recid[i])), replace = TRUE))
    })
  } else {
    rid <- lapply(seq_along(recid), function(i) {
      cbind(NewID = i, RowID = which(dat[, clustervar] == recid[i]))
    })
  }
  dat <- as.data.frame(do.call(rbind, rid))
  dat$Replicate <- factor(cut(dat$NewID, breaks = c(1, ncid * 1:reps), include.lowest = TRUE,
                              labels = FALSE))
  dat$NewID <- factor(dat$NewID)
  return(dat)
}
```
We will need to increase memory here since our sample is quite large. We will also set a seed to replicate the exercise (it has a random component when taking samples)

```
memory.limit()
memory.limit(size=30000)

#Set seed
set.seed(20)
```
As explained before, we will start by the highest level, which is player_id (highest number of individuals). I am measuring the time it takes because the next steps will be very time consuming and computationally expensive. Each one demanding more than the previous one. I decided to do 200 simulations. Feel free to increase this number if you have the computer power (and time) to do so. Beware: this might take hours.

```
start_time <- Sys.time();indx <- resampler(pbp_mut, "player_id", reps = 200);end_time <- Sys.time()
end_time - start_time

start_time <- Sys.time(); resampled_data <- cbind(indx, pbp_mut[indx$RowID, ]);end_time <- Sys.time()
end_time - start_time
```
I recommend saving resampled_data as a local RDS file and delete it since it is very heavy. But this is up to you.

**re-Fitting the model using simulations**

The next steps will consist of re-running the model multiple times, one per re-sample (simulation), remember we did 200. To do that, we will use the estimates from our original models (θ and β) as a starting point. Let's retrieve them (again) and store them

```
f <- fixef(mixed_model)
r <- getME(mixed_model, "theta")
```
As I mentioned before, this can be very expensive and time-consuming. So we will use parallel processing to speed-up things. I will be using 4 clusters because my laptop has 4 processors. (check how many your computer has and change makeClusters() if needed.
```
clus <- makeCluster(4)
clusterExport(clus, c("resampled_data", "f", "r"))
clusterEvalQ(clus, require(lme4))
```
Now we create our bootstrapping function:

```
boot_function <- function(i) {
  simu <- try(lmer(formula=
                     epa ~
                     temp +
                     wind + 
                     home +
                     wp +
                     as.factor(season) +
                     (1|player_id)+
                     (1|pos_coach)+
                     (1|team)+
                     (1|def_coach)+
                     (1|opponent)
                   ,
                   data = resampled_data,
                   subset = Replicate == i,
                   start = list(fixef = f, theta = r),
                   control=lmerControl(optimizer="bobyqa",
                                       optCtrl=list(maxfun=2e5))), silent = TRUE)
  if (class(simu) == "try-error")
    return(simu)
  c(fixef(simu), getME(simu, "theta"))
}
```
Here we do a loop to fit our model using the boot_function we created. It will run one time per simulation (200). This is the most time-consuming part of the analysis, so turn on Netflix and relax. Also, make sure to not use your computer since it will be running at max capacity. 

After we are done, we stop the cluster

```
start <- proc.time(); output <- parLapplyLB(clus, X = levels(resampled_data$Replicate), fun = boot_function); end <- proc.time()
end-start

stopCluster(clus)
```
**Create data frame**

For this dataset, I'm almost sure 100% of the models will work, but with other cases might not be the case. So we want to know which models work to retrieve theta only for those. Then we create the data frame using the output of successful models.

```
#Success rate
success <- sapply(output, is.numeric)
mean(success)

#Create dataframe
final <- do.call(cbind, output[success])
final_transposed<-t(final) %>% data.frame()
```
**Future Analysis**

As Parker explained to me, it would be a good idea to allow QB to interact with the different random-effects. That way we would be able to measure:
1- QB true value
2- Which factors impact QB performance (not passing offense) the most. The way interaction variables work, is that they allow us to see the marginal effect of one variable over the other. For example: by how much the effect of a QB ability change, on average, when the supporting cast improves.

Another pending thing to do is measure opposing defending team effect. I will try to increase the number of simulations to get a better estimate, not only for this variable but for all my random-effects. Ideally, it would be 1000 simulations.

**Prepare plot**

*Before continuing, I would like to shout-out my research-partner, twitter-and-zoom-friend, and future podcast Co-host: **Sam Struthers** @Sam_S35. I learned a lot about data viz from him while working on a project together.*

*The last shout-out (I promise) goes to Analytics Twitter extraordinaire and great buddy **Daniel Houston** @CowboysStats for his help with clarity and presentation. It doesn't matter how good the analysis if you can't tell the story effectively!*

Data viz is very important and now we will prepare the data for plotting. 
I decided not to include Opposing Defensive Team in the plot. I feel like a lot more simulations are needed to capture its true effect (around 1000 maybe). Maybe one day.

```
coach <- final_transposed %>% select(pos_coach..Intercept.) %>% mutate(Variable='Head Coach') %>% select(intercept = pos_coach..Intercept.,Variable)
qb <- final_transposed %>% select(player_id..Intercept.) %>% mutate(Variable='Quarterback') %>% select(intercept = player_id..Intercept.,Variable)
def_coach <- final_transposed %>% select(def_coach..Intercept.) %>% mutate(Variable='Opposing Head Coach')%>% select(intercept = def_coach..Intercept.,Variable)
team <- final_transposed %>% select(team..Intercept.) %>% mutate(Variable='Team (Proxy for Offensive Supporting Cast)')%>% select(intercept = team..Intercept.,Variable)

plot <- rbind(coach,qb,def_coach,team)
plot$Variable <- factor(plot$Variable, levels = c('Opposing Head Coach','Head Coach','Team (Proxy for Offensive Supporting Cast)','Quarterback'))
```
This is a straight forward plot:
```
ggplot(plot,aes(x=intercept, fill=Variable))+
  geom_density(alpha=.4)+
  theme_fivethirtyeight()+ 
  scale_fill_brewer(palette = 'Set1') +
  labs(x='Absolute Impact on Passing Offense Efficiency (EPA) | Further to the right means greater impact (positive or negative)', y='Density',
       title = "Quarterback-Ability Matters More Than Supporting Offensive Talent and Head Coach Competence",
       subtitle = 'When predicting passing efficiency, quarterback ability matters the most, followed by offensive personnel',
       caption = "Data and EPA models: nflfastR | Seasons 2006 - 2019 | Win Probability 20%-80% | Quarters 1-4 | Passing Plays Only
Mixed Effects Model to Find Intercept | Using 200 bootstrapped (re-sampled) datasets to find the distribution of Cholesky Factorized Coefficients"
  )
```
This is the fancy plot I used for twitter. It will look weird on Rstudio, but the saved file will look great if saved using my specs.

```
ggplot(plot,aes(x=intercept, fill=Variable))+
  geom_density(alpha=.4,size=.15)+
  theme_fivethirtyeight()+ 
  scale_fill_manual(values=c("#377EB8","#4DAF4A","#FF7F00","#984EA3"))+
  labs(x='Absolute Impact on Passing Offense Efficiency (EPA) | Further to the right means greater impact (positive or negative)', y='Density',
       title = "Quarterback-Ability Matters More Than Supporting Offensive Talent and Head Coach Competence",
       subtitle = 'When predicting passing efficiency, quarterback ability matters the most, followed by offensive personnel',
       caption = "Data and EPA models: nflfastR | Seasons 2006 - 2019 | Win Probability 20%-80% | Quarters 1-4 | Passing Plays Only
Mixed Effects Model to Find Intercept | Using 200 bootstrapped (re-sampled) datasets to find the distribution of Cholesky Factorized Coefficients"
  )+
  theme(text = element_text(),
        plot.title = element_text(size = 7, family = "Trebuchet MS",color = "grey20",hjust = .5),
        plot.subtitle = element_text(size = 6, family = "Trebuchet MS",color = "grey20",hjust = .5),
        axis.title = element_text(size = 5, family = "Trebuchet MS",color = "grey20"),
        axis.text = element_text(size = 3, family = "Trebuchet MS",color = "grey20"),
        legend.text = element_text(size = 5, family = "Trebuchet MS",color = "grey20"),
        legend.title = element_blank(),
        legend.position="top",
        legend.key.size =unit(.3,"line"),
        panel.grid = element_line(size=.11),
        plot.caption = element_text(size=4, family = "Trebuchet MS",color = "grey20",hjust = 0),
        plot.caption.position = "panel",
  )  +
  annotate(geom = "label", x = .001, y = 135, hjust = "left",fill="#F0F0F0",vjust = 1,
           label = "It is extremely hard for a HC to have passing game success 
without an elite QB or a good QB + supporting cast combination",
           size=1.6, 
           family = "Trebuchet MS",
           color = "grey20")  + 
  annotate(
    geom = "curve", x = .11, y = 50,xend = .103, yend = 16, 
    curvature = -.2, size= unit(.12, "mm"),arrow = arrow(length = unit(1.2, "mm")),
  )+
  annotate(geom = "label", x = .095, y = 65, hjust = "left",size=1.45,family = "Trebuchet MS",color = "grey20",fill="#F0F0F0",vjust = 1,
           label = 'Quarterback ability has 
the biggest impact, variance, 
and upside') + 
  annotate(
    geom = "curve", x = .102, y = 101,xend = .078, yend = 67, 
    curvature = -.2, size= unit(.15, "mm"),arrow = arrow(length = unit(1.2, "mm")),
  )+
  annotate(geom = "text", x = .101, y = 115, hjust = "left",size=1.45,family = "Trebuchet MS",color = "grey20",
           label = 'A good supporting cast 
greatly improves passing 
efficiency')+ 
  annotate(
    geom = "curve", x = .0505, y = 90,xend = .043, yend = 44, 
    curvature = .1, size= unit(.15, "mm"),arrow = arrow(length = unit(1.2, "mm")),
  )+
  annotate(geom = "text", x = .051, y = 92, hjust = "left",size=1.45,family = "Trebuchet MS",color = "grey20",
           label = 'Very rarely a HC matters 
more than the QB')+ 
  annotate(
    geom = "curve", x = .014, y = 67,xend = .021, yend = 49, 
    curvature = .1, size= unit(.15, "mm"),arrow = arrow(length = unit(1.2, "mm")),
  )+
  annotate(geom = "text", x = .001, y = 77, hjust = "left",size=1.45,family = "Trebuchet MS",color = "grey20",
           label = 'Defending HC is the least 
impactful factor') 

ggsave('passing_factors.png', dpi=1000, width = 12.5, height = 8, units = "cm")
```

Thanks! I hope this helps. For any questions, please find me @adrian_cadem on twitter or to my email: adriancadenam93@gmail.com <- but try to send me a DM since I might miss it
