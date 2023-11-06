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


options(warn=-1)


# functions for calculating confidence intervals using tables for t or z distributions
# (source: https://www.r-bloggers.com/2021/04/calculating-confidence-interval-in-r/)
Margin_Errors <- function (x, ci = 0.95)
{
	df_out <- CI_t(x,ci)
	return(df_out$values[df_out$Measurements=="Margin_Error"])
}

CI_t <- function (x, ci = 0.95)
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

CI_z <- function (x, ci = 0.95)
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

filenames <- list.files(scriptPath, pattern = ".csv", recursive = TRUE)
filenames <- paste(scriptPath, filenames, sep="")

print(filenames)
q()

resultsLog <- do.call(rbind, lapply(filenames, read.csv, header=TRUE, sep=","))
resultsLog <- resultsLog[resultsLog$iteration > 19,]
resultsLog <- resultsLog[complete.cases(resultsLog),]

print(sprintf("nRuns: %f", (nrow(unique(resultsLog[c("simsID", "run", "algorithm")])) / nrow(unique(resultsLog[c("algorithm")]))) ))


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



buildAbIncPlots <- function(avg, deviation, colors = NULL){

# 	if(!"linetype" %in% colnames(avg)){
# 		avg$linetype <- "solid"
# 	}

	plot <- ggplot(avg, aes(x = iteration, y=abilityInc, group=algorithm, color=algorithm, alpha = 0.8))

	plot <- plot + geom_errorbar(width=.1, aes(ymin=avg$abilityInc-deviation$abilityInc,
		ymax=avg$abilityInc+deviation$abilityInc), size = 0.8)

	plot <- plot + geom_line(aes(linetype=factor("solid")), size = 1.5) + geom_point(size = 4)
	plot <- plot + scale_linetype_manual(values=c("solid" = 1, "dashed" = 2), name = "linetype") + guides(linetype = FALSE)
	
	plot <- plot + labs(x = "Iteration", y = "Avg. Ability Increase", color="Algorithm") + 
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
	if(!is.null(colors)){
		plot <- plot + scale_color_manual(values = colors)
	}

	return(plot)
}

# ----------------------------------------------------------------------------------
# cmp avg execution times
print("avg execution times")
m_ga <- mean(resultsLog[resultsLog$algorithm=="GIMME_GA",]$iterationElapsedTime)
m_prs <- mean(resultsLog[resultsLog$algorithm=="GIMME_PRS",]$iterationElapsedTime)
m_rand <- mean(resultsLog[resultsLog$algorithm=="Random",]$iterationElapsedTime)
print("GA to PRS diff")
print((m_ga/m_prs - 1)*100)

print("GA to Random diff")
print((m_ga/m_rand - 1)*100)

print("PRS to Random diff")
print((m_prs/m_rand - 1)*100)


# ----------------------------------------------------------------------------------
# cmp average ability increase
print("average ability increase")
m_ga <- mean(resultsLog[resultsLog$algorithm=="GIMME_GA",]$abilityInc)
m_prs <- mean(resultsLog[resultsLog$algorithm=="GIMME_PRS",]$abilityInc)
m_rand <- mean(resultsLog[resultsLog$algorithm=="Random",]$abilityInc)
print("GA to PRS diff")
print((m_ga/m_prs - 1)*100)

print("GA to Random diff")
print((m_ga/m_rand - 1)*100)

print("PRS to Random diff")
print((m_prs/m_rand - 1)*100)



# q()


# ----------------------------------------------------------------------------------
# cmp Bootstrap with non-Bootstrap
currAvg = 	avg[
				avg$algorithm=="GIMME_ODPIP_Bootstrap" |
				avg$algorithm=="GIMME_CLink_Bootstrap" |
				avg$algorithm=="GIMME_PRS_Bootstrap" |
				avg$algorithm=="GIMME_GA_Bootstrap"
				,]

currDeviation = deviation[
				deviation$algorithm=="GIMME_ODPIP_Bootstrap" |
				deviation$algorithm=="GIMME_CLink_Bootstrap" |
				deviation$algorithm=="GIMME_PRS_Bootstrap" |
				deviation$algorithm=="GIMME_GA_Bootstrap"
				,]

# currAvg$algorithm[currAvg$algorithm == "GIMME_GA_Bootstrap"] <- "GIMME-GA-Bootstrap"

buildAbIncPlots(currAvg, currDeviation, c("#5e3c99", "dodgerblue","#75a352","#75a3e2", "#d7191c", "#d79b19", "#ff29ed"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityInc_bootstrapComp"), height=7, width=15, units="in", dpi=500))

# q()
# ----------------------------------------------------------------------------------
# cmp average ability increase 
currAvg = 	avg[
				avg$algorithm=="GIMME_CLink" |
				avg$algorithm=="GIMME_ODPIP" |
				avg$algorithm=="GIMME_PRS" | 
				avg$algorithm=="GIMME_GA" |
				avg$algorithm=="Random"
				,]

currDeviation = deviation[
				deviation$algorithm=="GIMME_CLink" |
				deviation$algorithm=="GIMME_ODPIP" |
				deviation$algorithm=="GIMME_PRS" |
				deviation$algorithm=="GIMME_GA" |
				deviation$algorithm=="Random"
				,]


currAvg$algorithm[currAvg$algorithm == "GIMME_CLink"] <- "GIMME-CLink"
currAvg$algorithm[currAvg$algorithm == "GIMME_ODPIP"] <- "GIMME-ODPIP"
currAvg$algorithm[currAvg$algorithm == "GIMME_PRS"] <- "GIMME-PRS"
currAvg$algorithm[currAvg$algorithm == "GIMME_GA"] <- "GIMME-GA"
currAvg$algorithm[currAvg$algorithm == "Random"] <- "Random"

# currAvg$linetype[currAvg$algorithm == "Perf. Info."] <- "dashed"

buildAbIncPlots(currAvg, currDeviation, c("#5e3c99", "dodgerblue","#75a352","#75a3e2", "#d7191c", "#d79b19", "#ff29ed"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityInc"), height=7, width=15, units="in", dpi=500))

# ----------------------------------------------------------------------------------
# cmp quality estimation algorithms (KNN vs Tabular)
currAvg = 	avg[
				avg$algorithm=="GIMME_ODPIP" |
				avg$algorithm=="GIMME_CLink" |
				avg$algorithm=="GIMME_ODPIP_Tabular" |
				avg$algorithm=="GIMME_CLink_Tabular"
				,]

currDeviation = deviation[
				deviation$algorithm=="GIMME_ODPIP" |
				deviation$algorithm=="GIMME_CLink" |
				deviation$algorithm=="GIMME_ODPIP_Tabular" |
				deviation$algorithm=="GIMME_CLink_Tabular"
				,]

buildAbIncPlots(currAvg, currDeviation, c("#5e3c99", "dodgerblue","#75a352","#75a3e2", "#d7191c", "#d79b19", "#ff29ed"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityInc_per_qualityEstAlg"), height=7, width=15, units="in", dpi=500))

q()

# ----------------------------------------------------------------------------------
# cmp average ability increase of GIMME with different accuracy est
currAvg = avg[
			  avg$algorithm=="GIMME_GA_Bootstrap" | 
			  avg$algorithm=="GIMME_GA_Bootstrap_LowAcc" | 
			  avg$algorithm=="GIMME_GA_Bootstrap_HighAcc"
			  ,]

currDeviation = deviation[
			  deviation$algorithm=="GIMME_GA_Bootstrap" |
			  deviation$algorithm=="GIMME_GA_Bootstrap_LowAcc" |
			  deviation$algorithm=="GIMME_GA_Bootstrap_HighAcc"
			  ,]

currAvg$linetype <- "solid"
currAvg$algorithm[currAvg$algorithm == "GIMME_GA_Bootstrap"] <- "GIMME-GA-Bootstrapp\n (\u03B3 = 0.1)"
currAvg$algorithm[currAvg$algorithm == "GIMME_GA_Bootstrap_LowAcc"] <- "GIMME-GA-Bootstrap\n (\u03B3 = 0.2)"
currAvg$algorithm[currAvg$algorithm == "GIMME_GA_Bootstrap_HighAcc"] <- "GIMME-GA-Bootstrap\n (\u03B3 = 0.05)"
currAvg$algorithm <- factor(currAvg$algorithm, levels=sort(unique(currAvg[,"algorithm"]), decreasing=TRUE))
ggp1 <- buildAbIncPlots(currAvg, currDeviation, c("skyblue", "dodgerblue", "navy"))


# ----------------------------------------------------------------------------------
# cmp average ability increase of GIMME with different accuracy est
currAvg = avg[
			  avg$algorithm=="GIMME_ODPIP_Bootstrap" |
			  avg$algorithm=="GIMME_ODPIP_Bootstrap_LowAcc" |
			  avg$algorithm=="GIMME_ODPIP_Bootstrap_HighAcc"
			  ,]

currDeviation = deviation[
			  deviation$algorithm=="GIMME_ODPIP_Bootstrap" |
			  deviation$algorithm=="GIMME_ODPIP_Bootstrap_LowAcc" |
			  deviation$algorithm=="GIMME_ODPIP_Bootstrap_HighAcc"
			  ,]

currAvg$linetype <- "solid"
currAvg$algorithm[currAvg$algorithm == "GIMME_ODPIP_Bootstrap"] <- "GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.1)"
currAvg$algorithm[currAvg$algorithm == "GIMME_ODPIP_Bootstrap_LowAcc"] <- "GIMME-ODPIP-Bootstrapp\n (\u03B3 = 0.2)"
currAvg$algorithm[currAvg$algorithm == "GIMME_ODPIP_Bootstrap_HighAcc"] <- "GIMME-ODPIP-Bootstrap\n (\u03B3 = 0.05)"
currAvg$algorithm <- factor(currAvg$algorithm, levels=sort(unique(currAvg[,"algorithm"]), decreasing=TRUE))
ggp2 <- buildAbIncPlots(currAvg, currDeviation, c("skyblue", "dodgerblue", "navy"))

# # ----------------------------------------------------------------------------------
# # cmp average ability increase of GIMME with different accuracy est
# currAvg = avg[
# 			  avg$algorithm=="GIMME_PRS_Bootstrap" |
# 			  avg$algorithm=="GIMME_PRS_Bootstrap_LowAcc" |
# 			  avg$algorithm=="GIMME_PRS_Bootstrap_HighAcc"
# 			  ,]
#
# currDeviation = deviation[
# 			  deviation$algorithm=="GIMME_PRS_Bootstrap" |
# 			  deviation$algorithm=="GIMME_PRS_Bootstrap_LowAcc" |
# 			  deviation$algorithm=="GIMME_PRS_Bootstrap_HighAcc"
# 			  ,]
#
# currAvg$linetype <- "solid"
# currAvg$algorithm[currAvg$algorithm == "GIMME_PRS_Bootstrap"] <- "GIMME-PRS-Bootstrap\n (\u03B3 = 0.1)"
# currAvg$algorithm[currAvg$algorithm == "GIMME_PRS_Bootstrap_LowAcc"] <- "GIMME-PRS-Bootstrapp\n (\u03B3 = 0.2)"
# currAvg$algorithm[currAvg$algorithm == "GIMME_PRS_Bootstrap_HighAcc"] <- "GIMME-PRS-Bootstrap\n (\u03B3 = 0.05)"
# currAvg$algorithm <- factor(currAvg$algorithm, levels=sort(unique(currAvg[,"algorithm"]), decreasing=TRUE))
# ggp2 <- buildAbIncPlots(currAvg, currDeviation, c("skyblue", "dodgerblue", "navy"))

ggp1 <- ggp1 + theme(plot.margin = margin(2,2,2,2, "cm"))
ggp2 <- ggp2 + theme(plot.margin = margin(2,2,2,2, "cm"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAccuracyComp"), height=7, width=25, units="in", dpi=500, arrangeGrob(ggp1, ggp2, ncol=2)))


# ----------------------------------------------------------------------------------
# cmp average ability increase of GIMME n-D
currAvg = avg[
				avg$algorithm=="GIMME_GA1D" | 
				avg$algorithm=="GIMME_GA" | 
				avg$algorithm=="GIMME_GA3D" | 
				avg$algorithm=="GIMME_GA4D" | 
				avg$algorithm=="GIMME_GA5D" | 
				avg$algorithm=="GIMME_GA6D" 
			,]
		
currDeviation = deviation[
				deviation$algorithm=="GIMME_GA1D" |
				deviation$algorithm=="GIMME_GA" |
				deviation$algorithm=="GIMME_GA3D" |
				deviation$algorithm=="GIMME_GA4D" |
				deviation$algorithm=="GIMME_GA5D" |
				deviation$algorithm=="GIMME_GA6D"
			,]


currAvg$algorithm[currAvg$algorithm == "GIMME_GA1D"] <- "GIMME-GA (1D)" 
currAvg$algorithm[currAvg$algorithm == "GIMME_GA"] <- "GIMME-GA (2D)" 
currAvg$algorithm[currAvg$algorithm == "GIMME_GA5D"] <- "GIMME-GA (5D)" 
currAvg$algorithm[currAvg$algorithm == "GIMME_GA6D"] <- "GIMME-GA (6D)" 
currAvg$algorithm[currAvg$algorithm == "GIMME_GA3D"] <- "GIMME-GA (3D)" 
currAvg$algorithm[currAvg$algorithm == "GIMME_GA4D"] <- "GIMME-GA (4D)"
# currAvg$algorithm[currAvg$algorithm == "Random"] <- "Random" 

currAvg$linetype <- "solid"
# currAvg$linetype[currAvg$algorithm == "Random"] <- "dashed" 

currAvg$algorithm <- factor(currAvg$algorithm, levels=c(sort(unique(currAvg[,"algorithm"]))))
buildAbIncPlots(currAvg, currDeviation)
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityGIPDims"), height=7, width=15, units="in", dpi=500))



# ----------------------------------------------------------------------------------
# cmp average ability increase of GIMME and GIMME EP
currAvg = avg[avg$algorithm=="GIMME_GA" | 
			  avg$algorithm=="GIMME_GA_EP",]

currDeviation = deviation[deviation$algorithm=="GIMME_GA" |
			    deviation$algorithm=="GIMME_GA_EP",]

currAvg$linetype <- "solid" 
currAvg$algorithm[currAvg$algorithm == "GIMME_GA"] <- "GIMME-GA" 
currAvg$algorithm[currAvg$algorithm == "GIMME_GA_EP"] <- "GIMME-GA (extr. prfs)" 			  
ggp1 <- buildAbIncPlots(currAvg, currDeviation, c("dodgerblue", "#d7191c"))


# ----------------------------------------------------------------------------------
# cmp average ability increase of GIMME and GIMME EP
currAvg = avg[avg$algorithm=="GIMME_PRS" | 
			  avg$algorithm=="GIMME_PRS_EP",]

currDeviation = deviation[deviation$algorithm=="GIMME_PRS" |
			    deviation$algorithm=="GIMME_PRS_EP",]

currAvg$linetype <- "solid" 
currAvg$algorithm[currAvg$algorithm == "GIMME_PRS"] <- "GIMME-PRS" 
currAvg$algorithm[currAvg$algorithm == "GIMME_PRS_EP"] <- "GIMME-PRS (extr. prfs)" 			  
ggp2 <- buildAbIncPlots(currAvg, currDeviation, c("dodgerblue", "#d7191c"))


ggp1 <- ggp1 + theme(plot.margin = margin(2,2,2,2, "cm"))
ggp2 <- ggp2 + theme(plot.margin = margin(2,2,2,2, "cm"))
suppressMessages(ggsave(sprintf(paste(scriptPath, "/plots/%s.png", sep=""), "simulationsResultsAbilityEP"), height=7, width=25, units="in", dpi=500, arrangeGrob(ggp1, ggp2, ncol=2)))

