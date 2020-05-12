
# VGLMADJCAT: ADJACENT CATEGORIES PROBABILITY MODEL FOR ORDINAL DATA: https://cran.r-project.org/web/packages/VGAM/
# rpartScore: CART OR ORDINAL RESPONSES: https://cran.r-project.org/web/packages/rpartScore/
rpartScore.Grid <-  expand.grid(cp = seq(100,500,100), split = c('abs', 'quad'), prune = c('mr', 'mc'))

# vglmContRatio: CONTINUATION RATIO MODEL FOR ORDINAL DATA: https://cran.r-project.org/web/packages/VGAM/
model.Grid <-  expand.grid(parallel = c(TRUE, FALSE), link = c('probit'))

# vglmCumulative: CUMULATIVE PROBABILITY MODEL FOR ORDINAL DATA: https://cran.r-project.org/web/packages/VGAM/

# polr: ORDERED LOGISTIC OR PROBIT REGRESSION: https://cran.r-project.org/web/packages/MASS/

# ordinalNet: PENALIZED ORDINAL REGRESSION: https://cran.r-project.org/web/packages/ordinalNet/

# ordinalRF: RANDOM FOREST: https://cran.r-project.org/web/packages/ordinalForest/ : https://cran.r-project.org/web/packages/ranger/