# scrape_injuries.R
# FIXED VERSION: Scrape injury data from Transfermarkt for 2022-2023 season (matching player data)

library(worldfootballR)
library(tidyverse)
library(lubridate)
library(jsonlite)

# Set up logging
log_message <- function(msg) {
  cat(paste0("[", Sys.time(), "] ", msg, "\n"))
}

# CRITICAL FIX: Target 2022-2023 season to match your player data
TARGET_SEASON_START <- 2022
TARGET_SEASON_END <- 2023
TARGET_SEASON_STRING <- "22/23"

# Function to get all team URLs for top 5 leagues (FIXED SEASON)
get_top5_league_teams <- function() {
  log_message(paste("Getting team URLs for top 5 leagues -", TARGET_SEASON_STRING, "season..."))
  
  leagues <- list(
    England = "England",
    Spain = "Spain", 
    Germany = "Germany",
    Italy = "Italy",
    France = "France"
  )
  
  all_teams <- list()
  
  for (league_name in names(leagues)) {
    log_message(paste("Getting teams for", league_name, TARGET_SEASON_STRING))
    tryCatch({
      # FIXED: Use correct season year
      teams <- tm_league_team_urls(
        country_name = leagues[[league_name]], 
        start_year = TARGET_SEASON_START  # 2022 for 2022-2023 season
      )
      all_teams[[league_name]] <- teams
      log_message(paste("Found", length(teams), "teams in", league_name))
      Sys.sleep(5)  # Increased rate limiting
    }, error = function(e) {
      log_message(paste("Error getting teams for", league_name, ":", e$message))
    })
  }
  
  return(all_teams)
}

# Function to get current injuries for all leagues
get_current_injuries <- function() {
  log_message("Getting current injuries for all leagues...")
  
  leagues <- c("England", "Spain", "Germany", "Italy", "France")
  all_injuries <- list()
  
  for (league in leagues) {
    log_message(paste("Getting injuries for", league))
    tryCatch({
      injuries <- tm_league_injuries(country_name = league)
      injuries$league <- league
      injuries$scrape_date <- Sys.Date()
      all_injuries[[league]] <- injuries
      log_message(paste("Found", nrow(injuries), "current injuries in", league))
      Sys.sleep(5)  # Increased rate limiting
    }, error = function(e) {
      log_message(paste("Error getting injuries for", league, ":", e$message))
    })
  }
  
  # Combine all injuries
  if (length(all_injuries) > 0) {
    combined_injuries <- bind_rows(all_injuries)
    return(combined_injuries)
  } else {
    return(data.frame())
  }
}

# Function to get player URLs for all teams (with better error handling)
get_all_player_urls <- function(team_urls_by_league, max_teams_per_league = 20) {
  log_message("Getting player URLs for all teams...")
  
  all_players <- list()
  
  for (league in names(team_urls_by_league)) {
    teams <- team_urls_by_league[[league]]
    # Limit teams for testing
    teams <- head(teams, max_teams_per_league)
    log_message(paste("Processing", length(teams), "teams in", league))
    
    for (i in seq_along(teams)) {
      team_url <- teams[i]
      log_message(paste("Getting players for team", i, "of", length(teams), "in", league))
      
      tryCatch({
        players <- tm_team_player_urls(team_url = team_url)
        if (length(players) > 0) {
          all_players[[paste0(league, "_", i)]] <- data.frame(
            player_url = players,
            team_url = team_url,
            league = league,
            team_index = i,
            stringsAsFactors = FALSE
          )
          log_message(paste("Found", length(players), "players"))
        }
        Sys.sleep(3)  # Conservative rate limiting
      }, error = function(e) {
        log_message(paste("Error getting players for team", i, ":", e$message))
        Sys.sleep(5)  # Longer wait on error
      })
    }
  }
  
  # Combine all player URLs
  if (length(all_players) > 0) {
    combined_players <- bind_rows(all_players)
    return(combined_players)
  } else {
    return(data.frame())
  }
}

# Function to get injury history for players (with better error handling)
get_player_injury_histories <- function(player_urls, batch_size = 25) {
  log_message(paste("Getting injury history for", length(player_urls), "players"))
  
  all_injuries <- list()
  total_batches <- ceiling(length(player_urls) / batch_size)
  
  for (batch in 1:total_batches) {
    start_idx <- (batch - 1) * batch_size + 1
    end_idx <- min(batch * batch_size, length(player_urls))
    
    batch_urls <- player_urls[start_idx:end_idx]
    log_message(paste("Processing batch", batch, "of", total_batches, 
                     "(players", start_idx, "to", end_idx, ")"))
    
    tryCatch({
      # Get injuries for batch
      injuries <- tm_player_injury_history(player_urls = batch_urls)
      if (nrow(injuries) > 0) {
        all_injuries[[batch]] <- injuries
        log_message(paste("Found", nrow(injuries), "injuries in batch", batch))
      }
      
      # Save intermediate results more frequently
      if (batch %% 5 == 0) {
        temp_combined <- bind_rows(all_injuries)
        write_csv(temp_combined, "../data/injuries/raw/temp_injuries_backup.csv")
        log_message(paste("Saved intermediate backup with", nrow(temp_combined), "injuries"))
      }
      
      Sys.sleep(8)  # Very conservative rate limiting
      
    }, error = function(e) {
      log_message(paste("Error in batch", batch, ":", e$message))
      Sys.sleep(10)  # Extra long wait on error
    })
  }
  
  # Combine all injury histories
  if (length(all_injuries) > 0) {
    combined_injuries <- bind_rows(all_injuries)
    return(combined_injuries)
  } else {
    return(data.frame())
  }
}

# FINAL FIX - Replace process_injury_data function with correct column names

process_injury_data <- function(injury_data) {
  log_message("Processing injury data...")
  
  if (nrow(injury_data) == 0) {
    log_message("No injury data to process")
    return(data.frame())
  }
  
  # Log actual columns found
  log_message("Available columns in injury data:")
  log_message(paste(colnames(injury_data), collapse = ", "))
  
  # Write the raw data to inspect it
  write_csv(injury_data, "../data/injuries/raw/raw_injury_data_debug.csv")
  log_message("Saved raw injury data for inspection")
  
  # FIXED: Use the actual column names from worldfootballR
  processed <- injury_data %>%
    # Clean dates using the ACTUAL column names
    mutate(
      injury_date_clean = case_when(
        !is.na(injured_since) ~ as.Date(injured_since, format = "%b %d, %Y"),
        TRUE ~ as.Date(NA)
      ),
      return_date_clean = case_when(
        !is.na(injured_until) ~ as.Date(injured_until, format = "%b %d, %Y"),
        TRUE ~ as.Date(NA)
      )
    ) %>%
    
    # Extract season from injury date or season_injured column
    mutate(
      season = case_when(
        # Use season_injured column if available
        !is.na(season_injured) ~ season_injured,
        # Otherwise calculate from injury_date_clean
        !is.na(injury_date_clean) && month(injury_date_clean) >= 8 ~ 
          paste0(year(injury_date_clean), "/", substr(year(injury_date_clean) + 1, 3, 4)),
        !is.na(injury_date_clean) ~ 
          paste0(year(injury_date_clean) - 1, "/", substr(year(injury_date_clean), 3, 4)),
        TRUE ~ TARGET_SEASON_STRING  # Default to target season
      )
    ) %>%
    
    # Filter for target season
    filter(season == TARGET_SEASON_STRING) %>%
    
    # Clean player names
    mutate(
      player_name_clean = str_trim(player_name),
      player_name_normalized = str_replace_all(tolower(player_name_clean), "[^a-z0-9\\s]", ""),
      player_name_normalized = str_squish(player_name_normalized)
    ) %>%
    
    # Categorize injuries using the 'injury' column
    mutate(
      injury_category = case_when(
        str_detect(tolower(injury), "hamstring") ~ "hamstring",
        str_detect(tolower(injury), "knee|acl|mcl|meniscus") ~ "knee",
        str_detect(tolower(injury), "ankle") ~ "ankle",
        str_detect(tolower(injury), "calf") ~ "calf",
        str_detect(tolower(injury), "groin|adductor") ~ "groin",
        str_detect(tolower(injury), "muscle|strain|tear") ~ "muscle",
        str_detect(tolower(injury), "back|spine") ~ "back",
        str_detect(tolower(injury), "thigh|quad") ~ "thigh",
        str_detect(tolower(injury), "shoulder|arm") ~ "upper_body",
        TRUE ~ "other"
      ),
      severity = case_when(
        duration <= 7 ~ "minor",
        duration <= 28 ~ "moderate", 
        duration <= 90 ~ "severe",
        duration > 90 ~ "career-threatening",
        TRUE ~ "unknown"
      )
    ) %>%
    
    # Filter out invalid records
    filter(
      !is.na(duration),
      duration > 0,
      nchar(player_name_clean) > 2
    ) %>%
    
    # Select and rename final columns to match expected schema
    select(
      player_name = player_name_clean,
      player_name_normalized,
      injury_date = injury_date_clean,
      return_date = return_date_clean,
      days_out = duration,
      injury_type = injury,
      injury_category,
      severity,
      season
    )
  
  log_message(paste("Processed", nrow(processed), "valid injury records for", TARGET_SEASON_STRING))
  
  if (nrow(processed) > 0) {
    # Show sample of processed data
    log_message("Sample of processed data:")
    print(head(processed, 3))
    
    # Show summary statistics
    log_message("Summary statistics:")
    log_message(paste("- Total injuries:", nrow(processed)))
    log_message(paste("- Unique players:", n_distinct(processed$player_name)))
    log_message(paste("- Injury categories:", paste(names(table(processed$injury_category)), collapse = ", ")))
    log_message(paste("- Severity levels:", paste(names(table(processed$severity)), collapse = ", ")))
  } else {
    log_message("WARNING: No injuries found for target season 22/23")
    log_message("This might be because:")
    log_message("1. The season_injured column doesn't match '22/23'")
    log_message("2. The injury dates are outside the target season")
    log_message("3. Data quality issues")
    
    # Show what seasons we actually have
    if ("season_injured" %in% colnames(injury_data)) {
      seasons_found <- unique(injury_data$season_injured)
      log_message(paste("Seasons found in data:", paste(seasons_found, collapse = ", ")))
    }
  }
  
  return(processed)
}

# Main execution function with better error handling
main <- function() {
  # Create output directories
  dir.create("../data/injuries/raw", recursive = TRUE, showWarnings = FALSE)
  dir.create("../data/injuries/processed", recursive = TRUE, showWarnings = FALSE)
  
  log_message("=== STARTING INJURY DATA SCRAPING ===")
  log_message(paste("Target season:", TARGET_SEASON_STRING))
  
  # Step 1: Get current injuries (for reference)
  log_message("\n=== STEP 1: Getting current injuries ===")
  tryCatch({
    current_injuries <- get_current_injuries()
    if (nrow(current_injuries) > 0) {
      write_csv(current_injuries, "../data/injuries/raw/current_injuries.csv")
      log_message(paste("Saved", nrow(current_injuries), "current injuries"))
    }
  }, error = function(e) {
    log_message(paste("Could not get current injuries:", e$message))
  })
  
  # Step 2: Get team URLs
  log_message("\n=== STEP 2: Getting team URLs ===")
  team_urls <- get_top5_league_teams()
  
  if (length(team_urls) == 0) {
    log_message("ERROR: No team URLs found. Check your internet connection and worldfootballR installation.")
    return(FALSE)
  }
  
  # Save team URLs
  saveRDS(team_urls, "../data/injuries/raw/team_urls.rds")
  total_teams <- sum(sapply(team_urls, length))
  log_message(paste("Saved URLs for", total_teams, "teams across", length(team_urls), "leagues"))
  
  # Step 3: Get player URLs (limited for initial testing)
  log_message("\n=== STEP 3: Getting player URLs ===")
  
  # Check if we already have player URLs saved
  if (file.exists("../data/injuries/raw/player_urls.csv")) {
    log_message("Loading existing player URLs...")
    all_players <- read_csv("../data/injuries/raw/player_urls.csv", show_col_types = FALSE)
  } else {
    # Start with fewer teams for testing
    all_players <- get_all_player_urls(team_urls, max_teams_per_league = 5)
    if (nrow(all_players) > 0) {
      write_csv(all_players, "../data/injuries/raw/player_urls.csv")
      log_message(paste("Saved", nrow(all_players), "player URLs"))
    } else {
      log_message("ERROR: No player URLs found")
      return(FALSE)
    }
  }
  
  # Step 4: Get injury histories (conservative approach)
  log_message("\n=== STEP 4: Getting injury histories ===")
  
  # Check if we already have injury data
  if (file.exists("../data/injuries/raw/all_injury_histories.csv")) {
    log_message("Loading existing injury histories...")
    injury_histories <- read_csv("../data/injuries/raw/all_injury_histories.csv", show_col_types = FALSE)
  } else {
    # Get unique player URLs (limit for testing)
    unique_players <- unique(all_players$player_url)
    # Start with smaller sample for testing
    test_players <- head(unique_players, 100)
    log_message(paste("Processing", length(test_players), "players for testing"))
    
    # Get injuries in small batches
    injury_histories <- get_player_injury_histories(test_players, batch_size = 10)
    
    if (nrow(injury_histories) > 0) {
      # Save raw data
      write_csv(injury_histories, "../data/injuries/raw/all_injury_histories.csv")
      log_message(paste("Saved", nrow(injury_histories), "injury records"))
    } else {
      log_message("No injury histories found")
      return(FALSE)
    }
  }
  
  # Step 5: Process and clean data
  log_message("\n=== STEP 5: Processing injury data ===")
  processed_injuries <- process_injury_data(injury_histories)
  
  if (nrow(processed_injuries) > 0) {
    # Save processed data
    write_csv(processed_injuries, "../data/injuries/processed/injuries_2022_23.csv")
    log_message(paste("Saved", nrow(processed_injuries), "processed injury records"))
    
    # Create summary statistics
    summary_stats <- list(
      season = TARGET_SEASON_STRING,
      total_injuries = nrow(processed_injuries),
      unique_players = n_distinct(processed_injuries$player_name_clean),
      injuries_by_category = as.list(table(processed_injuries$injury_category)),
      injuries_by_severity = as.list(table(processed_injuries$severity)),
      date_range = list(
        min_date = as.character(min(processed_injuries$injury_date_clean, na.rm = TRUE)),
        max_date = as.character(max(processed_injuries$injury_date_clean, na.rm = TRUE))
      )
    )
    
    # Save summary
    write_json(summary_stats, "../data/injuries/processed/injury_summary.json", pretty = TRUE)
    
    log_message("\n=== SCRAPING COMPLETE ===")
    log_message(paste("Season:", summary_stats$season))
    log_message(paste("Total injuries:", summary_stats$total_injuries))
    log_message(paste("Unique players:", summary_stats$unique_players))
    log_message(paste("Date range:", summary_stats$date_range$min_date, "to", summary_stats$date_range$max_date))
    
    return(TRUE)
  } else {
    log_message("ERROR: No processed injuries found for the target season")
    return(FALSE)
  }
}

# Run the main function
if (!interactive()) {
  success <- main()
  if (!success) {
    quit(status = 1)
  }
} else {
  log_message("Script loaded. Run main() to start scraping.")
}