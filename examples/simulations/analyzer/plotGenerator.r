# install.packages("stringi", dep = TRUE, repos = 'http://cran.rstudio.com/')
# install.packages("ggplot2", dep=TRUE, repos = "http://cran.us.r-project.org")
# install.packages("dplyr", dep=TRUE, repos = "http://cran.us.r-project.org")
# install.packages("gridExtra", dep=TRUE, repos = "http://cran.us.r-project.org")
# install.packages("envDocument", dep=TRUE, repos = "http://cran.us.r-project.org")

suppressMessages(library(gridExtra))
suppressMessages(library(ggplot2))
suppressMessages(library(stringr))
suppressMessages(library(dplyr))
suppressMessages(library(envDocument))



numTrainingIterations = 10

options(warn=-1)


# functions for calculating confidence intervals using tables for t or z distributions
# (source: https://www.r-bloggers.com/2021/04/calculating-confidence-interval-in-r/)
Margin_Errors <- function(x, ci = 0.95)
{
	df_out <- CI_t(x,ci)
	return(df_out$values[df_out$Measurements=="Margin_Error"])
}

CI_t <- function(x, ci = 0.95)
{
`%>%` <- magrittr::`%>%`
Margin_Error <- qt(ci + (1 - ci)/2, df = length(x) - 1) * sd(x)/sqrt(length(x))
df_out <- data.frame( sample_size=length(x), Mean=mean(x), sd=sd(x),
Margin_Error=Margin_Error,
'CI_LowerLimit'=(mean(x) - Margin_Error),
'CI_UpperLimit'=(mean(x) + Margin_Error)) %>%
tidyr::pivot_longer(names_to = "Measurements", values_to ="values", 1:6)
return(df_out)
}

CI_z <- function(x, ci = 0.95)
{
`%>%` <- magrittr::`%>%`
standard_deviation <- sd(x)
sample_size <- length(x)
Margin_Error <- abs(qnorm((1-ci)/2))* standard_deviation/sqrt(sample_size)
df_out <- data.frame( sample_size=length(x), Mean=mean(x), sd=sd(x),
Margin_Error=Margin_Error,
'CI_LowerLimit'=(mean(x) - Margin_Error),
'CI_UpperLimit'=(mean(x) + Margin_Error)) %>%
tidyr::pivot_longer(names_to = "Measurements", values_to ="values", 1:6)
return(df_out)
}



print("GeneratingPlots...")

# get current script path (source:'https://stackoverflow.com/questions/1815606/determine-path-of-the-executing-script')
initial.options <- commandArgs(trailingOnly = FALSE)
file.arg.name <- "--file="
scriptName <- sub(file.arg.name, "", initial.options[grep(file.arg.name, initial.options)])
scriptPath <- paste(dirname(scriptName), "/results/", sep='')

filenames <- list.files(scriptPath, pattern = ".csv", recursive = TRUE) #recursive=FALSE to only search on scriptPath
filenames <- paste(scriptPath, filenames, sep="")

resultsLog <- do.call(rbind, lapply(filenames, read.csv, header=TRUE, sep=","))
resultsLog <- resultsLog[resultsLog$iteration >= numTrainingIterations,]
resultsLog <- resultsLog[complete.cases(resultsLog),]

# print(sprintf("nRuns: %f", (nrow(unique(resultsLog[c("simsID", "run", "algorithm")])) / nrow(unique(resultsLog[c("algorithm")]))) ))
df <- unique(resultsLog[c("simsID", "run", "algorithm")]) 
df %>% count(algorithm)
# print(unique(resultsLog[c("algorithm")]))
# q()

resultsLog$algorithm[resultsLog$algorithm == "GIMME_CLink"] <- "GIMME-CLink"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_ODPIP"] <- "GIMME-ODPIP"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_PRS"] <- "GIMME-PRS"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_GA"] <- "GIMME-GA"

resultsLog$algorithm[resultsLog$algorithm == "GIMME_ODPIP_Bootstrap"] <- "GIMME-ODPIP-Bootstrap"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_CLink_Bootstrap"] <- "GIMME-CLink-Bootstrap"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_PRS_Bootstrap"] <- "GIMME-PRS-Bootstrap"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_GA_Bootstrap"] <- "GIMME-GA-Bootstrap"

resultsLog$algorithm[resultsLog$algorithm == "GIMME_ODPIP_Tabular"] <- "GIMME-ODPIP (Tabular Qlt. Eval.)"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_CLink_Tabular"] <- "GIMME-CLink (Tabular Qlt. Eval.)"


resultsLog$algorithm[resultsLog$algorithm == "GIMME_PRS_Bootstrap_LowAcc"] <- "GIMME-PRS-Bootstrap\n (\u03B3 = 0.2)"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_PRS_Bootstrap_HighAcc"] <- "GIMME-PRS-Bootstrap\n (\u03B3 = 0.05)"

resultsLog$algorithm[resultsLog$algorithm == "GIMME_GA_Bootstrap_LowAcc"] <- "GIMME-GA-Bootstrap\n (\u03B3 = 0.2)"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_GA_Bootstrap_HighAcc"] <- "GIMME-GA-Bootstrap\n (\u03B3 = 0.05)"

resultsLog$algorithm[resultsLog$algorithm == "GIMME_ODPIP_Bootstrap_LowAcc"] <- "GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.2)"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_ODPIP_Bootstrap_HighAcc"] <- "GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.05)"

resultsLog$algorithm[resultsLog$algorithm == "GIMME_CLink_Bootstrap_LowAcc"] <- "GIMME-CLink-Bootstrap\n (\u03B3 = 0.2)"
resultsLog$algorithm[resultsLog$algorithm == "GIMME_CLink_Bootstrap_HighAcc"] <- "GIMME-CLink-Bootstrap\n (\u03B3 = 0.05)"


# plot strategies
avgPerRun <- aggregate(abilityInc ~ iteration*algorithm*run*simsID, resultsLog, mean)

avg <- aggregate(abilityInc ~ iteration*algorithm, avgPerRun, mean)
deviation <- aggregate(abilityInc ~ iteration*algorithm, avgPerRun, Margin_Errors)
# print(deviation$abilityInc)

# deviation <- aggregate(abilityInc ~ iteration*algorithm , resultsLog , sd)
# does not make sense, because it would be influenced by the learning rate of the students. 
# Instead the standard deviation should be of the average of the class, per run.

# upBound <- max(avgPerRun$abilityInc[avgPerRun$algorithm == "accurate"]) #for maximum of all runs
# # upBound <- max(avg$abilityInc[avg$algorithm == "accurate"]) #for the maximum average value of all runs (more fair)
# avg$abilityInc[avg$algorithm == "accurate"] <- upBound
# deviation$abilityInc[deviation$algorithm == "accurate"] <- 0


buildAbIncPlots <- function(avg, deviation, colors = NULL, yLimInf = 0.30, yLimSup = 0.45){

# 	if(!"linetype" %in% colnames(avg)){
# 		avg$linetype <- "solid"
# 	}

	plot <- ggplot(avg, aes(x = iteration, y=abilityInc, group=algorithm, color=algorithm, shape=algorithm, alpha = 0.8))

	plot <- plot + geom_errorbar(width=.1, aes(ymin=avg$abilityInc-deviation$abilityInc,
		ymax=avg$abilityInc+deviation$abilityInc), size = 0.8)

	plot <- plot + geom_line(aes(linetype=factor("solid")), size = 1.5) + geom_point(size = 4)
	plot <- plot + scale_linetype_manual(values=c("solid" = 1, "dashed" = 2), name = "linetype") + guides(linetype = FALSE)
	
	plot <- plot + labs(x = "Iteration", y = "Avg. Ability Increase") + 
					theme(axis.text = element_text(size = 30), 
					axis.title = element_text(size = 35, face = "bold"), 
					legend.title = element_blank(), 
					legend.text = element_text(size=25), 
					legend.position = 'bottom',
					legend.key = element_blank(),
					panel.background = element_blank(),
					panel.grid.major = element_blank(), 
					panel.grid.minor = element_blank(),
					panel.border = element_rect(colour = "black", fill=NA, size=2.0))
	plot <- plot + scale_x_continuous(labels = 1:20, breaks = 20:39) + scale_alpha(guide=FALSE) 
	
	
	if(!is.na(yLimInf) & !is.na(yLimSup)){
		plot <- plot + ylim(yLimInf,yLimSup)
	}
	
	if(!is.null(colors)){
		plot <- plot + scale_color_manual(values = colors)
	}

	return(plot)
}

# ----------------------------------------------------------------------------------
# cmp avg execution times
print("-----------[Avg execution times]----------")
m_rand <- mean(resultsLog[resultsLog$algorithm=="Random",]$iterationElapsedTime)
m_clink <- mean(resultsLog[resultsLog$algorithm=="GIMME-CLink",]$iterationElapsedTime)
m_prs <- mean(resultsLog[resultsLog$algorithm=="GIMME-PRS",]$iterationElapsedTime)
m_ga <- mean(resultsLog[resultsLog$algorithm=="GIMME-GA",]$iterationElapsedTime)
m_odpip <- mean(resultsLog[resultsLog$algorithm=="GIMME-ODPIP",]$iterationElapsedTime)

print("PRS to Random diff")
print((m_prs/m_rand - 1)*100)

print("GA to PRS diff")
print((m_ga/m_prs - 1)*100)

print("C-Link to GA diff")
print((m_clink/m_ga - 1)*100)

print("ODP-IP to CLink diff")
print((m_odpip/m_clink - 1)*100)


# ----------------------------------------------------------------------------------
# cmp average ability increase
print("-----------[Average ability increase]----------")
m_rand <- mean(resultsLog[resultsLog$algorithm=="Random",]$abilityInc)
m_clink <- mean(resultsLog[resultsLog$algorithm=="GIMME-CLink",]$abilityInc)
m_prs <- mean(resultsLog[resultsLog$algorithm=="GIMME-PRS",]$abilityInc)
m_ga <- mean(resultsLog[resultsLog$algorithm=="GIMME-GA",]$abilityInc)
m_odpip <- mean(resultsLog[resultsLog$algorithm=="GIMME-ODPIP",]$abilityInc)

print("PRS to Random diff")
print((m_prs/m_rand - 1)*100)

print("C-Link to Random diff")
print((m_clink/m_rand - 1)*100)

print("PRS to CLink diff")
print((m_prs/m_clink - 1)*100)

print("GA to PRS diff")
print((m_ga/m_prs - 1)*100)

print("ODP-IP to GA diff")
print((m_odpip/m_ga - 1)*100)


# ----------------------------------------------------------------------------------
# cmp EP average ability increase
print("-----------[Average ability increase]----------")
m_prs_ep <- mean(resultsLog[resultsLog$algorithm=="GIMME_PRS_EP",]$abilityInc)
m_ga_ep <- mean(resultsLog[resultsLog$algorithm=="GIMME_GA_EP",]$abilityInc)
m_clink_ep <- mean(resultsLog[resultsLog$algorithm=="GIMME_CLink_EP",]$abilityInc)
m_odpip_ep <- mean(resultsLog[resultsLog$algorithm=="GIMME_ODPIP_EP",]$abilityInc)

print("PRS to PRS_EP diff")
print((m_prs/m_prs_ep - 1)*100)

print("GA to GA_EP diff")
print((m_ga/m_ga_ep - 1)*100)

print("ODP-IP to ODP-IP_EP diff")
print((m_odpip/m_odpip_ep - 1)*100)

print("C-Link to C-Link_EP diff")
print((m_clink/m_clink_ep - 1)*100)




# ----------------------------------------------------------------------------------
# cmp average ability increase 
currAvg = 	avg[
				avg$algorithm=="GIMME-CLink" |
				avg$algorithm=="GIMME-ODPIP" |
				avg$algorithm=="GIMME-PRS" | 
				avg$algorithm=="GIMME-GA" |
				avg$algorithm=="Random"
				,]

currDeviation = deviation[
				deviation$algorithm=="GIMME-CLink" |
				deviation$algorithm=="GIMME-ODPIP" |
				deviation$algorithm=="GIMME-PRS" |
				deviation$algorithm=="GIMME-GA" |
				deviation$algorithm=="Random"
				,]

# currAvg$linetype[currAvg$algorithm == "Perf. Info."] <- "dashed"

buildAbIncPlots(currAvg, currDeviation, c("#5e3c99", "dodgerblue","#75a352","#75a3e2", "#d7191c", "#d79b19", "#ff29ed"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityInc"), height=7, width=15, units="in", dpi=500))

# ----------------------------------------------------------------------------------
# cmp Bootstrap with non-Bootstrap of the different algorithms
currAvg = avg[
				avg$algorithm=="GIMME-ODPIP" |
				avg$algorithm=="GIMME-ODPIP-Bootstrap"
				,]
currDeviation = deviation[
				deviation$algorithm=="GIMME-ODPIP" |
				deviation$algorithm=="GIMME-ODPIP-Bootstrap"
				,]

buildAbIncPlots(currAvg, currDeviation, c("#5e3c99", "dodgerblue","#75a352","#75a3e2", "#d7191c", "#d79b19", "#ff29ed"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityInc_bootstrapComp_ODPIP"), height=7, width=15, units="in", dpi=500))

currAvg = avg[
				avg$algorithm=="GIMME-CLink" |
				avg$algorithm=="GIMME-CLink-Bootstrap" 
				,]
currDeviation = deviation[
				deviation$algorithm=="GIMME-CLink" |
				deviation$algorithm=="GIMME-CLink-Bootstrap" 
				,]
				
buildAbIncPlots(currAvg, currDeviation, c("#5e3c99", "dodgerblue","#75a352","#75a3e2", "#d7191c", "#d79b19", "#ff29ed"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityInc_bootstrapComp_CLink"), height=7, width=15, units="in", dpi=500))

				
currAvg = avg[
				avg$algorithm=="GIMME-PRS" | 
				avg$algorithm=="GIMME-PRS-Bootstrap"
				,]
currDeviation = deviation[
				deviation$algorithm=="GIMME-PRS" | 
				deviation$algorithm=="GIMME-PRS-Bootstrap"
				,]
				
buildAbIncPlots(currAvg, currDeviation, c("#5e3c99", "dodgerblue","#75a352","#75a3e2", "#d7191c", "#d79b19", "#ff29ed"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityInc_bootstrapComp_PRS"), height=7, width=15, units="in", dpi=500))


currAvg = avg[
				avg$algorithm=="GIMME-GA" |
				avg$algorithm=="GIMME-GA-Bootstrap"
				,]
currDeviation = deviation[
				deviation$algorithm=="GIMME-GA" |
				deviation$algorithm=="GIMME-GA-Bootstrap"
				,]

buildAbIncPlots(currAvg, currDeviation, c("#5e3c99", "dodgerblue","#75a352","#75a3e2", "#d7191c", "#d79b19", "#ff29ed"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityInc_bootstrapComp_GA"), height=7, width=15, units="in", dpi=500))


# ----------------------------------------------------------------------------------
# cmp quality estimation algorithms (KNN vs Tabular)
currAvg = 	avg[
				avg$algorithm=="GIMME-ODPIP" |
				avg$algorithm=="GIMME-ODPIP (Tabular Qlt. Eval.)"
				,]

currDeviation = deviation[
				deviation$algorithm=="GIMME-ODPIP" |
				deviation$algorithm=="GIMME-ODPIP (Tabular Qlt. Eval.)"
				,]

buildAbIncPlots(currAvg, currDeviation, c("#5e3c99", "dodgerblue","#75a352","#75a3e2", "#d7191c", "#d79b19", "#ff29ed"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityInc_qualityEstAlg_ODPIP"), height=7, width=15, units="in", dpi=500))

currAvg = 	avg[
				avg$algorithm=="GIMME-CLink" |
				avg$algorithm=="GIMME-CLink (Tabular Qlt. Eval.)"
				,]

currDeviation = deviation[
				deviation$algorithm=="GIMME-CLink" |
				deviation$algorithm=="GIMME-CLink (Tabular Qlt. Eval.)"
				,]

buildAbIncPlots(currAvg, currDeviation, c("#5e3c99", "dodgerblue","#75a352","#75a3e2", "#d7191c", "#d79b19", "#ff29ed"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityInc_qualityEstAlg_CLink"), height=7, width=15, units="in", dpi=500))


# ----------------------------------------------------------------------------------
# cmp average ability increase of GIMME with different accuracy est

avg$algorithm[avg$algorithm == "GIMME-PRS-Bootstrap"] <- "GIMME-PRS-Bootstrap\n (\u03B3 = 0.1)"
avg$algorithm[avg$algorithm == "GIMME-GA-Bootstrap"] <- "GIMME-GA-Bootstrap\n (\u03B3 = 0.1)"
avg$algorithm[avg$algorithm == "GIMME-ODPIP-Bootstrap"] <- "GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.1)"
avg$algorithm[avg$algorithm == "GIMME-CLink-Bootstrap"] <- "GIMME-CLink-Bootstrap\n (\u03B3 = 0.1)"

deviation$algorithm[deviation$algorithm == "GIMME-PRS-Bootstrap"] <- "GIMME-PRS-Bootstrap\n (\u03B3 = 0.1)"
deviation$algorithm[deviation$algorithm == "GIMME-GA-Bootstrap"] <- "GIMME-GA-Bootstrap\n (\u03B3 = 0.1)"
deviation$algorithm[deviation$algorithm == "GIMME-ODPIP-Bootstrap"] <- "GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.1)"
deviation$algorithm[deviation$algorithm == "GIMME-CLink-Bootstrap"] <- "GIMME-CLink-Bootstrap\n (\u03B3 = 0.1)"

currAvg = avg[
			  avg$algorithm=="GIMME-PRS-Bootstrap\n (\u03B3 = 0.1)" | 
			  avg$algorithm=="GIMME-PRS-Bootstrap\n (\u03B3 = 0.2)" | 
			  avg$algorithm=="GIMME-PRS-Bootstrap\n (\u03B3 = 0.05)"
			  ,]

currDeviation = deviation[
			  deviation$algorithm=="GIMME-PRS-Bootstrap\n (\u03B3 = 0.1)" | 
			  deviation$algorithm=="GIMME-PRS-Bootstrap\n (\u03B3 = 0.2)" | 
			  deviation$algorithm=="GIMME-PRS-Bootstrap\n (\u03B3 = 0.05)"
			  ,]

currAvg$algorithm <- factor(currAvg$algorithm, levels=sort(unique(currAvg[,"algorithm"]), decreasing=TRUE))
buildAbIncPlots(currAvg, currDeviation, c("skyblue", "dodgerblue", "navy"), yLimInf=NA, yLimSup=NA)
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAccuracyComp_PRS"), height=7, width=15, units="in", dpi=500))


currAvg = avg[
			  avg$algorithm=="GIMME-GA-Bootstrap\n (\u03B3 = 0.1)" | 
			  avg$algorithm=="GIMME-GA-Bootstrap\n (\u03B3 = 0.2)" | 
			  avg$algorithm=="GIMME-GA-Bootstrap\n (\u03B3 = 0.05)"
			  ,]

currDeviation = deviation[
			  deviation$algorithm=="GIMME-GA-Bootstrap\n (\u03B3 = 0.1)" | 
			  deviation$algorithm=="GIMME-GA-Bootstrap\n (\u03B3 = 0.2)" | 
			  deviation$algorithm=="GIMME-GA-Bootstrap\n (\u03B3 = 0.05)"
			  ,]

currAvg$algorithm <- factor(currAvg$algorithm, levels=sort(unique(currAvg[,"algorithm"]), decreasing=TRUE))
buildAbIncPlots(currAvg, currDeviation, c("skyblue", "dodgerblue", "navy"), yLimInf=NA, yLimSup=NA)
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAccuracyComp_GA"), height=7, width=15, units="in", dpi=500))


currAvg = avg[
			  avg$algorithm=="GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.1)" | 
			  avg$algorithm=="GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.2)" | 
			  avg$algorithm=="GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.05)"
			  ,]

currDeviation = deviation[
			  deviation$algorithm=="GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.1)" | 
			  deviation$algorithm=="GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.2)" | 
			  deviation$algorithm=="GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.05)"
			  ,]

currAvg$algorithm <- factor(currAvg$algorithm, levels=sort(unique(currAvg[,"algorithm"]), decreasing=TRUE))
buildAbIncPlots(currAvg, currDeviation, c("skyblue", "dodgerblue", "navy"), yLimInf=NA, yLimSup=NA)
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAccuracyComp_ODPIP"), height=7, width=15, units="in", dpi=500))


currAvg = avg[
			  avg$algorithm=="GIMME-CLink-Bootstrap\n (\u03B3 = 0.1)" | 
			  avg$algorithm=="GIMME-CLink-Bootstrap\n (\u03B3 = 0.2)" | 
			  avg$algorithm=="GIMME-CLink-Bootstrap\n (\u03B3 = 0.05)"
			  ,]

currDeviation = deviation[
			  deviation$algorithm=="GIMME-CLink-Bootstrap\n (\u03B3 = 0.1)" | 
			  deviation$algorithm=="GIMME-CLink-Bootstrap\n (\u03B3 = 0.2)" | 
			  deviation$algorithm=="GIMME-CLink-Bootstrap\n (\u03B3 = 0.05)"
			  ,]

currAvg$algorithm <- factor(currAvg$algorithm, levels=sort(unique(currAvg[,"algorithm"]), decreasing=TRUE))
buildAbIncPlots(currAvg, currDeviation, c("skyblue", "dodgerblue", "navy"), yLimInf=NA, yLimSup=NA)
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAccuracyComp_CLink"), height=7, width=15, units="in", dpi=500))





# ----------------------------------------------------------------------------------
# cmp average ability increase of GIMME and GIMME EP
currAvg = avg[avg$algorithm=="GIMME-PRS" | 
			  avg$algorithm=="GIMME_PRS_EP",]

currDeviation = deviation[deviation$algorithm=="GIMME-PRS" |
			    deviation$algorithm=="GIMME_PRS_EP",]

currAvg$algorithm[currAvg$algorithm == "GIMME_PRS_EP"] <- "GIMME-PRS (extr. prfs)" 			  
buildAbIncPlots(currAvg, currDeviation, c("dodgerblue", "#d7191c"), yLimInf=NA, yLimSup=NA)
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityIncEP_PRS"), height=7, width=15, units="in", dpi=500))


currAvg = avg[avg$algorithm=="GIMME-GA" | 
			  avg$algorithm=="GIMME_GA_EP",]

currDeviation = deviation[deviation$algorithm=="GIMME-GA" |
			    deviation$algorithm=="GIMME_GA_EP",]
 
currAvg$algorithm[currAvg$algorithm == "GIMME_GA_EP"] <- "GIMME-GA (extr. prfs)" 			  
buildAbIncPlots(currAvg, currDeviation, c("dodgerblue", "#d7191c"), yLimInf=NA, yLimSup=NA)
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityIncEP_GA"), height=7, width=15, units="in", dpi=500))


currAvg = avg[avg$algorithm=="GIMME-ODPIP" | 
			  avg$algorithm=="GIMME_ODPIP_EP",]

currDeviation = deviation[deviation$algorithm=="GIMME-ODPIP" |
			    deviation$algorithm=="GIMME_ODPIP_EP",]
 
currAvg$algorithm[currAvg$algorithm == "GIMME_ODPIP_EP"] <- "GIMME-ODPIP (extr. prfs)" 			  
buildAbIncPlots(currAvg, currDeviation, c("dodgerblue", "#d7191c"), yLimInf=NA, yLimSup=NA)
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityIncEP_ODPIP"), height=7, width=15, units="in", dpi=500))



currAvg = avg[avg$algorithm=="GIMME-CLink" | 
			  avg$algorithm=="GIMME_CLink_EP",]

currDeviation = deviation[deviation$algorithm=="GIMME-CLink" |
			    deviation$algorithm=="GIMME_CLink_EP",]
 
currAvg$algorithm[currAvg$algorithm == "GIMME_CLink_EP"] <- "GIMME-CLink (extr. prfs)" 			  
buildAbIncPlots(currAvg, currDeviation, c("dodgerblue", "#d7191c"), yLimInf=NA, yLimSup=NA)
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityIncEP_CLink"), height=7, width=15, units="in", dpi=500))




# ----------------------------------------------------------------------------------
# cmp average ability increase of GIMME n-D

avg$algorithm[avg$algorithm == "GIMME_GA1D"] <- "GIMME-GA (1D)" 
avg$algorithm[avg$algorithm == "GIMME-GA"] <-   "GIMME-GA (2D)" 
avg$algorithm[avg$algorithm == "GIMME_GA3D"] <- "GIMME-GA (3D)" 
avg$algorithm[avg$algorithm == "GIMME_GA4D"] <- "GIMME-GA (4D)" 
avg$algorithm[avg$algorithm == "GIMME_GA5D"] <- "GIMME-GA (5D)" 
avg$algorithm[avg$algorithm == "GIMME_GA6D"] <- "GIMME-GA (6D)" 

deviation$algorithm[deviation$algorithm == "GIMME_GA1D"] <- "GIMME-GA (1D)" 
deviation$algorithm[deviation$algorithm == "GIMME-GA"] <-   "GIMME-GA (2D)" 
deviation$algorithm[deviation$algorithm == "GIMME_GA3D"] <- "GIMME-GA (3D)" 
deviation$algorithm[deviation$algorithm == "GIMME_GA4D"] <- "GIMME-GA (4D)" 
deviation$algorithm[deviation$algorithm == "GIMME_GA5D"] <- "GIMME-GA (5D)" 
deviation$algorithm[deviation$algorithm == "GIMME_GA6D"] <- "GIMME-GA (6D)" 



currAvg = avg[
				avg$algorithm=="GIMME-GA (1D)" | 
				avg$algorithm=="GIMME-GA (2D)" | 
				avg$algorithm=="GIMME-GA (5D)" | 
				avg$algorithm=="GIMME-GA (6D)" | 
				avg$algorithm=="GIMME-GA (3D)" | 
				avg$algorithm=="GIMME-GA (4D)"
			,]
		
currDeviation = deviation[
				deviation$algorithm=="GIMME-GA (1D)" |
				deviation$algorithm=="GIMME-GA (2D)" |
				deviation$algorithm=="GIMME-GA (5D)" |
				deviation$algorithm=="GIMME-GA (6D)" |
				deviation$algorithm=="GIMME-GA (3D)" |
				deviation$algorithm=="GIMME-GA (4D)"
			,]


currAvg$algorithm <- factor(currAvg$algorithm, levels=c(sort(unique(currAvg[,"algorithm"]))))
buildAbIncPlots(currAvg, currDeviation, yLimInf=NA, yLimSup=NA)
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityGIPDims"), height=7, width=15, units="in", dpi=500))
