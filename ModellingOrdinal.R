
# VGLMADJCAT: ADJACENT CATEGORIES PROBABILITY MODEL FOR ORDINAL DATA: https://cran.r-project.org/web/packages/VGAM/
# rpartScore: CART OR ORDINAL RESPONSES: https://cran.r-project.org/web/packages/rpartScore/
rpartScore.Grid <-  expand.grid(cp = seq(100,500,100), split = c('abs', 'quad'), prune = c('mr', 'mc'))

# 