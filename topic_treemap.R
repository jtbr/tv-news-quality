#library("ggplot2")
#install.packages("showtext")
#install.packages("igraph")
#install.packages("treemap")
#install.packages("Cairo")

library("showtext")
font_add("Charis SIL", "/usr/share/fonts/opentype/charis/CharisSIL-R.ttf")

#font.add("perpetua", "PER_____.ttf")
#windowsFonts(Perpetua=windowsFont("TT Perpetua"))

library("Cairo") # for embedding fonts in pdf
#Cairo()

#library(scales) #defines comma, scientific in order to format numbers in legend labels

library(treemap) # defines treemap visualization

showtext_auto()

group=c("government", "business & economics", "current events", "ads", "anecdotes", "transitions", "elections: soft", "elections: hard", "cultural", "entertainment", "science & tech", "sports", "products", "weather", "none")
supergroup=c("2","1", "3","5","3","5", "2","2", "3", "4", "1", "4", "1", "3", "5")
value=c(19.48, 13.52, 12.55, 8.61, 6.93, 6.78, 6.01, 2.02, 5.82, 5.29, 3.66, 2.69, 2.50, 2.16, 1.97)

# add values to the group names so they show up in the labels
valuepct <- paste(value, "%", sep="")
group <- paste(group, valuepct, sep = "\n")

data=data.frame(supergroup,group,value)


# start cairo device
Cairo(file="../../dissertation/classification/figures/topic_random_labels-treemap.pdf",
      type="pdf",
      units="in", 
      width=6.5, 
      height=4.5, 
      pointsize=12, 
      dpi=72)

# treemap
treemap(data,
        index=c("supergroup","group"),
        vSize="value",
        type="index",
        title = "",
        fontsize.labels=c(13,11),  
        fontfamily.labels = "Charis SIL",
        fontcolor.labels = "white",
        align.labels=list(                 # Where to place labels in the rectangle?
          c("left", "top"),           # top-level upper left
          c("center", "center")       # group-level centered
        )
)        

#end cairo device, saving file
dev.off()
