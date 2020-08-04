# This script specifies some test cases (test_cases variable below) and uses the GetNewDataB function from
# MultiPLIER to project the data into the latent space. Then it saves these results into .RDS files, which will be
# used by unit tests to check the Python implementation of the GetNewDataB function.

suppressMessages(require(PLIER))

source(file.path("scripts", "plier_util.R"))

args <- commandArgs(trailingOnly=TRUE)
if (length(args) == 0) {
  stop('At least one argument is required: the path to the MultiPLIER model file (.rds).')
}

message('Reading MultiPLIER model')
model <- readRDS(args[1])

output_dir <- args[2]

# specify some small use cases for unit tests
test_cases <- list(
  # Two genes and two traits
  list(
    genes = c('GAS6', 'DSP'),
    trait1 = c(-0.25, 0.50),
    trait2 = c(1.25, -1.50)
  ),
  # Three genes and two traits
  list(
    genes = c('GAS6', 'DSP', 'MMP14'),
    trait1 = c(-0.25, 0.50, 0.90),
    trait2 = c(1.25, -1.50, -1.56)
  ),
  # Three genes (one does not exist in the model) and three traits
  list(
    genes = c('GAS6', 'DOESNOTEXIST', 'MMP14'),
    trait1 = c(-0.25, 0.50, 0.90),
    trait2 = c(1.25, -1.50, -1.56),
    trait3 = c(-11.2, 0.17, 1.99)
  ),
  # Missing data pattern #1
  list(
    genes = c('GAS6', 'DSP', 'MMP14'),
    trait1 = c(-0.25, 0.50, 0.90),
    trait2 = c(1.25, -1.50, -1.56),
    trait3 = c(-11.2, 0.17, NA)
  )
)

# this test case contains a sample from PhenomeXcan data
new_data <- readRDS(file.path(output_dir, 'test_case5', 'phenomexcan_sample.rds'))
test_cases[[length(test_cases) + 1]] <- list(
  genes = rownames(new_data),
  trait1 = new_data[[1]],
  trait2 = new_data[[2]],
  trait3 = new_data[[3]]
)


# go through all test cases, project them into the latent space, and save the results in an rds file.
for (tc.idx in 1:length(test_cases)) {
  message(paste0('Test case ', tc.idx))
  tc <- test_cases[[tc.idx]]

  tc.df <- data.frame(tc)
  tc.mat <- as.matrix(tc.df)
  tc.mat <- matrix(as.numeric(tc.mat[, 2:ncol(tc.mat)]), nrow=nrow(tc.mat))
  rownames(tc.mat) <- tc.df$genes
  colnames(tc.mat) <- colnames(tc.df)[2:ncol(tc.df)]

  tc_output_dir <- file.path(output_dir, paste0('test_case', tc.idx))
  if (!dir.exists(tc_output_dir)) {
    dir.create(tc_output_dir)
  }

  # save input data
  print(head(tc.mat))
  saveRDS(tc.mat, file.path(tc_output_dir, 'input_data.rds'))

  # save output data
  b.matrix <- GetNewDataB(exprs.mat = tc.mat, plier.model = model)
  saveRDS(b.matrix, file.path(tc_output_dir, 'output_data.rds'))

  message('\n')
}

message('Finished')
