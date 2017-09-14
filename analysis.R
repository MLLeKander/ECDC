rm(list=ls())
my.require = function(package) {
  print(package)
  if (!require(package, character.only = TRUE)) {
    install.packages(package)
    library(package, character.only = TRUE)
  }
}
my.require('ggplot2')
my.require('data.table')
my.require('cowplot')


dat = data.table(read.csv('run.csv'))

ggplot(data = dat, aes(x = totalFrameCount, y = return)) +
  background_grid(major = 'y', minor = "none") + geom_smooth()# + geom_point(alpha=0.1, size=0.1)

