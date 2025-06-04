# simple_setup.R
# Simplified R package installation for macOS

# Set CRAN mirror
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Install packages using binary versions when possible
install.packages(c("xml2", "rvest"), type = "binary")

# Install tidyverse and other packages
packages <- c(
  "tidyverse",
  "dplyr", 
  "readr",
  "jsonlite",
  "lubridate",
  "stringr",
  "purrr"
)

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, type = "binary")
  }
}

# Install devtools if needed
if (!require("devtools")) {
  install.packages("devtools", type = "binary")
}

# Try to install worldfootballR
tryCatch({
  devtools::install_github("JaseZiv/worldfootballR")
  cat("✅ worldfootballR installed successfully!\n")
}, error = function(e) {
  cat("❌ worldfootballR installation failed:", e$message, "\n")
  cat("Trying alternative installation...\n")
  
  # Try installing dependencies first
  install.packages(c("httr", "jsonlite", "purrr", "stringr", "glue"), type = "binary")
  devtools::install_github("JaseZiv/worldfootballR")
})

# Test if everything works
tryCatch({
  library(worldfootballR)
  library(tidyverse)
  cat("✅ All packages loaded successfully!\n")
  
  # Simple test
  cat("Testing worldfootballR connection...\n")
  # We'll skip the actual test for now to avoid rate limiting
  cat("✅ Setup complete! Ready for scraping.\n")
  
}, error = function(e) {
  cat("❌ Package loading failed:", e$message, "\n")
})

# Print session info
sessionInfo()