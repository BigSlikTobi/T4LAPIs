# NFL Data Python Library - Complete Table and Column Reference

> **Generated on:** 2025-07-26 22:14:56  
> **Library:** nfl_data_py  
> **Total Functions Explored:** 24

This document provides a comprehensive overview of all available data tables and their columns in the `nfl_data_py` library.

## Summary

| Status | Count | Functions |
|--------|-------|-----------|
| ✅ Successful | 24 | import_combine_data, import_contracts, import_depth_charts, import_draft_picks, import_draft_values,... |
| ❌ Failed | 0 |  |

## Available Data Tables

### 1. import_combine_data

**Data Shape:** 8,649 rows × 18 columns  
**Function Call:** `nfl.import_combine_data()`

**Description:**  
Import combine results for all position groups

Args:
    years (List[str]): years to get combine data for
    positions (List[str]): list of positions to get data for
    
Returns:
    DataFrame

**Columns (18):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `season` | int32 | 2000 |
| 2 | `draft_year` | float64 | 2000.0 |
| 3 | `draft_team` | object | New York Jets |
| 4 | `draft_round` | float64 | 1.0 |
| 5 | `draft_ovr` | float64 | 13.0 |
| 6 | `pfr_id` | object | AbraJo00 |
| 7 | `cfb_id` | object | None |
| 8 | `player_name` | object | John Abraham |
| 9 | `pos` | object | OLB |
| 10 | `school` | object | South Carolina |
| 11 | `ht` | object | 6-4 |
| 12 | `wt` | float64 | 252.0 |
| 13 | `forty` | float64 | 4.55 |
| 14 | `bench` | float64 | NULL |
| 15 | `vertical` | float64 | NULL |
| 16 | `broad_jump` | float64 | NULL |
| 17 | `cone` | float64 | NULL |
| 18 | `shuttle` | float64 | NULL |

---

### 2. import_contracts

**Data Shape:** 46,841 rows × 25 columns  
**Function Call:** `nfl.import_contracts()`

**Description:**  
Imports historical contract data

Returns:
    DataFrame

**Columns (25):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `player` | object | Joe Burrow |
| 2 | `position` | object | QB |
| 3 | `team` | object | Bengals |
| 4 | `is_active` | bool | True |
| 5 | `year_signed` | int32 | 2023 |
| 6 | `years` | float64 | 5.0 |
| 7 | `value` | float64 | 275.0 |
| 8 | `apy` | float64 | 55.0 |
| 9 | `guaranteed` | float64 | 146.51 |
| 10 | `apy_cap_pct` | float64 | 0.245 |
| 11 | `inflated_value` | float64 | 341.548043 |
| 12 | `inflated_apy` | float64 | 68.309609 |
| 13 | `inflated_guaranteed` | float64 | 181.964377 |
| 14 | `player_page` | object | https://overthecap.com/player/joe-burrow/8741/ |
| 15 | `otc_id` | int32 | 8741 |
| 16 | `gsis_id` | object | 00-0036442 |
| 17 | `date_of_birth` | object | None |
| 18 | `height` | object | 6'4" |
| 19 | `weight` | object | 215 |
| 20 | `college` | object | LSU |
| 21 | `draft_year` | float64 | 2020.0 |
| 22 | `draft_round` | float64 | 1.0 |
| 23 | `draft_overall` | float64 | 1.0 |
| 24 | `draft_team` | object | Bengals |
| 25 | `cols` | object | None |

---

### 3. import_depth_charts

**Data Shape:** 37,327 rows × 15 columns  
**Function Call:** `nfl.import_depth_charts()`

**Description:**  
Imports team depth charts

Args:
    years (List[int]): years to return depth charts for, optional
Returns:
    DataFrame

**Columns (15):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `season` | int32 | 2023 |
| 2 | `club_code` | object | ATL |
| 3 | `week` | float64 | 1.0 |
| 4 | `game_type` | object | REG |
| 5 | `depth_team` | object | 2 |
| 6 | `last_name` | object | Davis |
| 7 | `first_name` | object | Octavious |
| 8 | `football_name` | object | Tae |
| 9 | `formation` | object | Defense |
| 10 | `gsis_id` | object | 00-0034500 |
| 11 | `jersey_number` | object | 33 |
| 12 | `position` | object | ILB |
| 13 | `elias_id` | object | DAV579855 |
| 14 | `depth_position` | object | ILB |
| 15 | `full_name` | object | Tae Davis |

---

### 4. import_draft_picks

**Data Shape:** 12,670 rows × 36 columns  
**Function Call:** `nfl.import_draft_picks()`

**Description:**  
Import draft picks

Args:
    years (List[int]): years to get draft picks for

Returns:
    DataFrame

**Columns (36):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `season` | int32 | 1980 |
| 2 | `round` | int32 | 1 |
| 3 | `pick` | int32 | 1 |
| 4 | `team` | object | DET |
| 5 | `gsis_id` | object |  |
| 6 | `pfr_player_id` | object | SimsBi00 |
| 7 | `cfb_player_id` | object | billy-sims-1 |
| 8 | `pfr_player_name` | object | Billy Sims |
| 9 | `hof` | bool | False |
| 10 | `position` | object | RB |
| 11 | `category` | object | RB |
| 12 | `side` | object | O |
| 13 | `college` | object | Oklahoma |
| 14 | `age` | float64 | 24.0 |
| 15 | `to` | float64 | 1984.0 |
| 16 | `allpro` | int32 | 0 |
| 17 | `probowls` | int32 | 3 |
| 18 | `seasons_started` | int32 | 5 |
| 19 | `w_av` | float64 | 58.0 |
| 20 | `car_av` | float64 | NULL |
| 21 | `dr_av` | float64 | 58.0 |
| 22 | `games` | float64 | 60.0 |
| 23 | `pass_completions` | float64 | 0.0 |
| 24 | `pass_attempts` | float64 | 0.0 |
| 25 | `pass_yards` | float64 | 0.0 |
| 26 | `pass_tds` | float64 | 0.0 |
| 27 | `pass_ints` | float64 | 0.0 |
| 28 | `rush_atts` | float64 | 1131.0 |
| 29 | `rush_yards` | float64 | 5106.0 |
| 30 | `rush_tds` | float64 | 42.0 |
| 31 | `receptions` | float64 | 186.0 |
| 32 | `rec_yards` | float64 | 2072.0 |
| 33 | `rec_tds` | float64 | 5.0 |
| 34 | `def_solo_tackles` | float64 | NULL |
| 35 | `def_ints` | float64 | NULL |
| 36 | `def_sacks` | float64 | NULL |

---

### 5. import_draft_values

**Data Shape:** 0 rows × 6 columns  
**Function Call:** `nfl.import_draft_values()`

**Description:**  
Import draft pick values from variety of models

Args:
    picks (List[int]): subset of picks to return values for
    
Returns:
    DataFrame

**Columns (6):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `pick` | int64 | N/A |
| 2 | `stuart` | float64 | N/A |
| 3 | `johnson` | int64 | N/A |
| 4 | `hill` | float64 | N/A |
| 5 | `otc` | int64 | N/A |
| 6 | `pff` | float64 | N/A |

---

### 6. import_ftn_data

**Data Shape:** 48,225 rows × 29 columns  
**Function Call:** `nfl.import_ftn_data()`

**Description:**  
Imports FTN charting data

FTN Data manually charts plays and has graciously provided a subset of their
charting data to be published via the nflverse. Data is available from the 2022
season onwards a...

**Columns (29):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `ftn_game_id` | int32 | 6164 |
| 2 | `nflverse_game_id` | object | 2023_01_DET_KC |
| 3 | `season` | int32 | 2023 |
| 4 | `week` | int32 | 1 |
| 5 | `ftn_play_id` | int32 | 1009770 |
| 6 | `nflverse_play_id` | int32 | 40 |
| 7 | `starting_hash` | object | 0 |
| 8 | `qb_location` | object | 0 |
| 9 | `n_offense_backfield` | float32 | 0.0 |
| 10 | `n_defense_box` | int32 | 0 |
| 11 | `is_no_huddle` | bool | False |
| 12 | `is_motion` | bool | False |
| 13 | `is_play_action` | bool | False |
| 14 | `is_screen_pass` | bool | False |
| 15 | `is_rpo` | bool | False |
| 16 | `is_trick_play` | float32 | 0.0 |
| 17 | `is_qb_out_of_pocket` | bool | False |
| 18 | `is_interception_worthy` | bool | False |
| 19 | `is_throw_away` | bool | False |
| 20 | `read_thrown` | object | 0 |
| 21 | `is_catchable_ball` | bool | False |
| 22 | `is_contested_ball` | bool | False |
| 23 | `is_created_reception` | bool | False |
| 24 | `is_drop` | bool | False |
| 25 | `is_qb_sneak` | bool | False |
| 26 | `n_blitzers` | int32 | 0 |
| 27 | `n_pass_rushers` | float32 | 0.0 |
| 28 | `is_qb_fault_sack` | bool | False |
| 29 | `date_pulled` | datetime64[us] | 2024-09-06 23:50:14.696374 |

---

### 7. import_ids

**Data Shape:** 12,075 rows × 35 columns  
**Function Call:** `nfl.import_ids()`

**Description:**  
Import mapping table of ids for most major data providers

Args:
    columns (List[str]): list of columns to return
    ids (List[str]): list of specific ids to return
    
Returns:
    DataFrame

**Columns (35):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `mfl_id` | int64 | 17030 |
| 2 | `sportradar_id` | object | 3c76cab3-3df2-43dd-acaa-57e055bd32d0 |
| 3 | `fantasypros_id` | float64 | 24755.0 |
| 4 | `gsis_id` | object | NULL |
| 5 | `pff_id` | float64 | NULL |
| 6 | `sleeper_id` | float64 | 12522.0 |
| 7 | `nfl_id` | object | NULL |
| 8 | `espn_id` | float64 | 4688380.0 |
| 9 | `yahoo_id` | float64 | NULL |
| 10 | `fleaflicker_id` | float64 | NULL |
| 11 | `cbs_id` | float64 | 3168422.0 |
| 12 | `pfr_id` | object | NULL |
| 13 | `cfbref_id` | object | NULL |
| 14 | `rotowire_id` | float64 | 16997.0 |
| 15 | `rotoworld_id` | float64 | NULL |
| 16 | `ktc_id` | float64 | 1730.0 |
| 17 | `stats_id` | float64 | 41786.0 |
| 18 | `stats_global_id` | float64 | 0.0 |
| 19 | `fantasy_data_id` | float64 | NULL |
| 20 | `swish_id` | float64 | NULL |
| 21 | `name` | object | Cam Ward |
| 22 | `merge_name` | object | cam ward |
| 23 | `position` | object | QB |
| 24 | `team` | object | TEN |
| 25 | `birthdate` | object | 2002-05-25 |
| 26 | `age` | float64 | 23.2 |
| 27 | `draft_year` | float64 | 2025.0 |
| 28 | `draft_round` | float64 | 1.0 |
| 29 | `draft_pick` | float64 | 1.0 |
| 30 | `draft_ovr` | float64 | NULL |
| 31 | `twitter_username` | object | NULL |
| 32 | `height` | float64 | 74.0 |
| 33 | `weight` | float64 | 219.0 |
| 34 | `college` | object | Miami (FL) |
| 35 | `db_season` | int64 | 2025 |

---

### 8. import_injuries

**Data Shape:** 5,599 rows × 16 columns  
**Function Call:** `nfl.import_injuries()`

**Description:**  
Imports team injury reports

Args:
    years (List[int]): years to return injury reports for, optional
Returns:
    DataFrame

**Columns (16):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `season` | int32 | 2023 |
| 2 | `game_type` | object | REG |
| 3 | `team` | object | ARI |
| 4 | `week` | int32 | 1 |
| 5 | `gsis_id` | object | 00-0034473 |
| 6 | `position` | object | LB |
| 7 | `full_name` | object | Dennis Gardeck |
| 8 | `first_name` | object | Dennis |
| 9 | `last_name` | object | Gardeck |
| 10 | `report_primary_injury` | object | None |
| 11 | `report_secondary_injury` | object | None |
| 12 | `report_status` | object | None |
| 13 | `practice_primary_injury` | object | Knee |
| 14 | `practice_secondary_injury` | object | None |
| 15 | `practice_status` | object | Full Participation in Practice |
| 16 | `date_modified` | datetime64[us] | 2023-09-08 18:49:43 |

---

### 9. import_ngs_data

**Data Shape:** 620 rows × 29 columns  
**Function Call:** `nfl.import_ngs_data()`

**Description:**  
Imports seasonal NGS data

Args:
    stat_type (str): type of stats to pull (receiving, passing, rushing)
    years (List[int]): years to get PBP data for, optional
Returns:
    DataFrame

**Columns (29):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `season` | int32 | 2023 |
| 2 | `season_type` | object | REG |
| 3 | `week` | int32 | 0 |
| 4 | `player_display_name` | object | Jake Browning |
| 5 | `player_position` | object | QB |
| 6 | `team_abbr` | object | CIN |
| 7 | `avg_time_to_throw` | float64 | 2.7286131687242796 |
| 8 | `avg_completed_air_yards` | float64 | 4.683040935672515 |
| 9 | `avg_intended_air_yards` | float64 | 6.3211489361702125 |
| 10 | `avg_air_yards_differential` | float64 | -1.6381080004976978 |
| 11 | `aggressiveness` | float64 | 16.46090534979424 |
| 12 | `max_completed_air_distance` | float64 | 53.43437563965729 |
| 13 | `avg_air_yards_to_sticks` | float64 | -2.478851063829787 |
| 14 | `attempts` | int32 | 243 |
| 15 | `pass_yards` | int32 | 1936 |
| 16 | `pass_touchdowns` | int32 | 12 |
| 17 | `interceptions` | int32 | 7 |
| 18 | `passer_rating` | float64 | 98.37962962962963 |
| 19 | `completions` | int32 | 171 |
| 20 | `completion_percentage` | float64 | 70.37037037037037 |
| 21 | `expected_completion_percentage` | float64 | 67.08378143026015 |
| 22 | `completion_percentage_above_expectation` | float64 | 3.286588940110221 |
| 23 | `avg_air_distance` | float64 | 20.291303400501107 |
| 24 | `max_air_distance` | float64 | 53.43437563965729 |
| 25 | `player_gsis_id` | object | 00-0035100 |
| 26 | `player_first_name` | object | Jake |
| 27 | `player_last_name` | object | Browning |
| 28 | `player_jersey_number` | int32 | 6 |
| 29 | `player_short_name` | object | J.Browning |

---

### 10. import_officials

**Data Shape:** 1,993 rows × 5 columns  
**Function Call:** `nfl.import_officials()`

**Description:**  
Import game officials

Args:
    years (List[int]): years to get officials for
    
Returns:
    DataFrame

**Columns (5):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `game_id` | object | 2023_01_DET_KC |
| 2 | `off_pos` | object | R |
| 3 | `official_id` | object | HussJo0r |
| 4 | `name` | object | John Hussey |
| 5 | `season` | int64 | 2023 |

---

### 11. import_pbp_data

**Data Shape:** 49,665 rows × 390 columns  
**Function Call:** `nfl.import_pbp_data()`

**Description:**  
Imports play-by-play data

Args:
    years (List[int]): years to get PBP data for
    columns (List[str]): only return these columns
    include_participation (bool): whether to include participation ...

**Columns (390):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `play_id` | float32 | 1.0 |
| 2 | `game_id` | object | 2023_01_ARI_WAS |
| 3 | `old_game_id` | object | 2023091007 |
| 4 | `home_team` | object | WAS |
| 5 | `away_team` | object | ARI |
| 6 | `season_type` | object | REG |
| 7 | `week` | int32 | 1 |
| 8 | `posteam` | object | None |
| 9 | `posteam_type` | object | None |
| 10 | `defteam` | object | None |
| 11 | `side_of_field` | object | None |
| 12 | `yardline_100` | float32 | NULL |
| 13 | `game_date` | object | 2023-09-10 |
| 14 | `quarter_seconds_remaining` | float32 | 900.0 |
| 15 | `half_seconds_remaining` | float32 | 1800.0 |
| 16 | `game_seconds_remaining` | float32 | 3600.0 |
| 17 | `game_half` | object | Half1 |
| 18 | `quarter_end` | float32 | 0.0 |
| 19 | `drive` | float32 | NULL |
| 20 | `sp` | float32 | 0.0 |
| 21 | `qtr` | float32 | 1.0 |
| 22 | `down` | float32 | NULL |
| 23 | `goal_to_go` | int32 | 0 |
| 24 | `time` | object | 15:00 |
| 25 | `yrdln` | object | ARI 35 |
| 26 | `ydstogo` | float32 | 0.0 |
| 27 | `ydsnet` | float32 | NULL |
| 28 | `desc` | object | GAME |
| 29 | `play_type` | object | None |
| 30 | `yards_gained` | float32 | NULL |
| 31 | `shotgun` | float32 | 0.0 |
| 32 | `no_huddle` | float32 | 0.0 |
| 33 | `qb_dropback` | float32 | NULL |
| 34 | `qb_kneel` | float32 | 0.0 |
| 35 | `qb_spike` | float32 | 0.0 |
| 36 | `qb_scramble` | float32 | 0.0 |
| 37 | `pass_length` | object | None |
| 38 | `pass_location` | object | None |
| 39 | `air_yards` | float32 | NULL |
| 40 | `yards_after_catch` | float32 | NULL |
| 41 | `run_location` | object | None |
| 42 | `run_gap` | object | None |
| 43 | `field_goal_result` | object | None |
| 44 | `kick_distance` | float32 | NULL |
| 45 | `extra_point_result` | object | None |
| 46 | `two_point_conv_result` | object | None |
| 47 | `home_timeouts_remaining` | float32 | 3.0 |
| 48 | `away_timeouts_remaining` | float32 | 3.0 |
| 49 | `timeout` | float32 | NULL |
| 50 | `timeout_team` | object | None |
| 51 | `td_team` | object | None |
| 52 | `td_player_name` | object | None |
| 53 | `td_player_id` | object | None |
| 54 | `posteam_timeouts_remaining` | float32 | NULL |
| 55 | `defteam_timeouts_remaining` | float32 | NULL |
| 56 | `total_home_score` | float32 | 0.0 |
| 57 | `total_away_score` | float32 | 0.0 |
| 58 | `posteam_score` | float32 | NULL |
| 59 | `defteam_score` | float32 | NULL |
| 60 | `score_differential` | float32 | NULL |
| 61 | `posteam_score_post` | float32 | NULL |
| 62 | `defteam_score_post` | float32 | NULL |
| 63 | `score_differential_post` | float32 | NULL |
| 64 | `no_score_prob` | float32 | 0.0 |
| 65 | `opp_fg_prob` | float32 | 0.0 |
| 66 | `opp_safety_prob` | float32 | 0.0 |
| 67 | `opp_td_prob` | float32 | 0.0 |
| 68 | `fg_prob` | float32 | 0.0 |
| 69 | `safety_prob` | float32 | 0.0 |
| 70 | `td_prob` | float32 | 0.0 |
| 71 | `extra_point_prob` | float32 | 0.0 |
| 72 | `two_point_conversion_prob` | float32 | 0.0 |
| 73 | `ep` | float32 | 1.474097728729248 |
| 74 | `epa` | float32 | 0.0 |
| 75 | `total_home_epa` | float32 | 0.0 |
| 76 | `total_away_epa` | float32 | 0.0 |
| 77 | `total_home_rush_epa` | float32 | 0.0 |
| 78 | `total_away_rush_epa` | float32 | 0.0 |
| 79 | `total_home_pass_epa` | float32 | 0.0 |
| 80 | `total_away_pass_epa` | float32 | 0.0 |
| 81 | `air_epa` | float32 | NULL |
| 82 | `yac_epa` | float32 | NULL |
| 83 | `comp_air_epa` | float32 | NULL |
| 84 | `comp_yac_epa` | float32 | NULL |
| 85 | `total_home_comp_air_epa` | float32 | 0.0 |
| 86 | `total_away_comp_air_epa` | float32 | 0.0 |
| 87 | `total_home_comp_yac_epa` | float32 | 0.0 |
| 88 | `total_away_comp_yac_epa` | float32 | 0.0 |
| 89 | `total_home_raw_air_epa` | float32 | 0.0 |
| 90 | `total_away_raw_air_epa` | float32 | 0.0 |
| 91 | `total_home_raw_yac_epa` | float32 | 0.0 |
| 92 | `total_away_raw_yac_epa` | float32 | 0.0 |
| 93 | `wp` | float32 | 0.5462617874145508 |
| 94 | `def_wp` | float32 | 0.4537382125854492 |
| 95 | `home_wp` | float32 | 0.5462617874145508 |
| 96 | `away_wp` | float32 | 0.4537382125854492 |
| 97 | `wpa` | float32 | 0.0 |
| 98 | `vegas_wpa` | float32 | 0.0 |
| 99 | `vegas_home_wpa` | float32 | 0.0 |
| 100 | `home_wp_post` | float32 | NULL |
| 101 | `away_wp_post` | float32 | NULL |
| 102 | `vegas_wp` | float32 | 0.7373994588851929 |
| 103 | `vegas_home_wp` | float32 | 0.7373994588851929 |
| 104 | `total_home_rush_wpa` | float32 | 0.0 |
| 105 | `total_away_rush_wpa` | float32 | 0.0 |
| 106 | `total_home_pass_wpa` | float32 | 0.0 |
| 107 | `total_away_pass_wpa` | float32 | 0.0 |
| 108 | `air_wpa` | float32 | NULL |
| 109 | `yac_wpa` | float32 | NULL |
| 110 | `comp_air_wpa` | float32 | NULL |
| 111 | `comp_yac_wpa` | float32 | NULL |
| 112 | `total_home_comp_air_wpa` | float32 | 0.0 |
| 113 | `total_away_comp_air_wpa` | float32 | 0.0 |
| 114 | `total_home_comp_yac_wpa` | float32 | 0.0 |
| 115 | `total_away_comp_yac_wpa` | float32 | 0.0 |
| 116 | `total_home_raw_air_wpa` | float32 | 0.0 |
| 117 | `total_away_raw_air_wpa` | float32 | 0.0 |
| 118 | `total_home_raw_yac_wpa` | float32 | 0.0 |
| 119 | `total_away_raw_yac_wpa` | float32 | 0.0 |
| 120 | `punt_blocked` | float32 | NULL |
| 121 | `first_down_rush` | float32 | NULL |
| 122 | `first_down_pass` | float32 | NULL |
| 123 | `first_down_penalty` | float32 | NULL |
| 124 | `third_down_converted` | float32 | NULL |
| 125 | `third_down_failed` | float32 | NULL |
| 126 | `fourth_down_converted` | float32 | NULL |
| 127 | `fourth_down_failed` | float32 | NULL |
| 128 | `incomplete_pass` | float32 | NULL |
| 129 | `touchback` | float32 | 0.0 |
| 130 | `interception` | float32 | NULL |
| 131 | `punt_inside_twenty` | float32 | NULL |
| 132 | `punt_in_endzone` | float32 | NULL |
| 133 | `punt_out_of_bounds` | float32 | NULL |
| 134 | `punt_downed` | float32 | NULL |
| 135 | `punt_fair_catch` | float32 | NULL |
| 136 | `kickoff_inside_twenty` | float32 | NULL |
| 137 | `kickoff_in_endzone` | float32 | NULL |
| 138 | `kickoff_out_of_bounds` | float32 | NULL |
| 139 | `kickoff_downed` | float32 | NULL |
| 140 | `kickoff_fair_catch` | float32 | NULL |
| 141 | `fumble_forced` | float32 | NULL |
| 142 | `fumble_not_forced` | float32 | NULL |
| 143 | `fumble_out_of_bounds` | float32 | NULL |
| 144 | `solo_tackle` | float32 | NULL |
| 145 | `safety` | float32 | NULL |
| 146 | `penalty` | float32 | NULL |
| 147 | `tackled_for_loss` | float32 | NULL |
| 148 | `fumble_lost` | float32 | NULL |
| 149 | `own_kickoff_recovery` | float32 | NULL |
| 150 | `own_kickoff_recovery_td` | float32 | NULL |
| 151 | `qb_hit` | float32 | NULL |
| 152 | `rush_attempt` | float32 | NULL |
| 153 | `pass_attempt` | float32 | NULL |
| 154 | `sack` | float32 | NULL |
| 155 | `touchdown` | float32 | NULL |
| 156 | `pass_touchdown` | float32 | NULL |
| 157 | `rush_touchdown` | float32 | NULL |
| 158 | `return_touchdown` | float32 | NULL |
| 159 | `extra_point_attempt` | float32 | NULL |
| 160 | `two_point_attempt` | float32 | NULL |
| 161 | `field_goal_attempt` | float32 | NULL |
| 162 | `kickoff_attempt` | float32 | NULL |
| 163 | `punt_attempt` | float32 | NULL |
| 164 | `fumble` | float32 | NULL |
| 165 | `complete_pass` | float32 | NULL |
| 166 | `assist_tackle` | float32 | NULL |
| 167 | `lateral_reception` | float32 | NULL |
| 168 | `lateral_rush` | float32 | NULL |
| 169 | `lateral_return` | float32 | NULL |
| 170 | `lateral_recovery` | float32 | NULL |
| 171 | `passer_player_id` | object | None |
| 172 | `passer_player_name` | object | None |
| 173 | `passing_yards` | float32 | NULL |
| 174 | `receiver_player_id` | object | None |
| 175 | `receiver_player_name` | object | None |
| 176 | `receiving_yards` | float32 | NULL |
| 177 | `rusher_player_id` | object | None |
| 178 | `rusher_player_name` | object | None |
| 179 | `rushing_yards` | float32 | NULL |
| 180 | `lateral_receiver_player_id` | object | None |
| 181 | `lateral_receiver_player_name` | object | None |
| 182 | `lateral_receiving_yards` | float32 | NULL |
| 183 | `lateral_rusher_player_id` | object | None |
| 184 | `lateral_rusher_player_name` | object | None |
| 185 | `lateral_rushing_yards` | float32 | NULL |
| 186 | `lateral_sack_player_id` | object | None |
| 187 | `lateral_sack_player_name` | object | None |
| 188 | `interception_player_id` | object | None |
| 189 | `interception_player_name` | object | None |
| 190 | `lateral_interception_player_id` | object | None |
| 191 | `lateral_interception_player_name` | object | None |
| 192 | `punt_returner_player_id` | object | None |
| 193 | `punt_returner_player_name` | object | None |
| 194 | `lateral_punt_returner_player_id` | object | None |
| 195 | `lateral_punt_returner_player_name` | object | None |
| 196 | `kickoff_returner_player_name` | object | None |
| 197 | `kickoff_returner_player_id` | object | None |
| 198 | `lateral_kickoff_returner_player_id` | object | None |
| 199 | `lateral_kickoff_returner_player_name` | object | None |
| 200 | `punter_player_id` | object | None |
| 201 | `punter_player_name` | object | None |
| 202 | `kicker_player_name` | object | None |
| 203 | `kicker_player_id` | object | None |
| 204 | `own_kickoff_recovery_player_id` | object | None |
| 205 | `own_kickoff_recovery_player_name` | object | None |
| 206 | `blocked_player_id` | object | None |
| 207 | `blocked_player_name` | object | None |
| 208 | `tackle_for_loss_1_player_id` | object | None |
| 209 | `tackle_for_loss_1_player_name` | object | None |
| 210 | `tackle_for_loss_2_player_id` | object | None |
| 211 | `tackle_for_loss_2_player_name` | object | None |
| 212 | `qb_hit_1_player_id` | object | None |
| 213 | `qb_hit_1_player_name` | object | None |
| 214 | `qb_hit_2_player_id` | object | None |
| 215 | `qb_hit_2_player_name` | object | None |
| 216 | `forced_fumble_player_1_team` | object | None |
| 217 | `forced_fumble_player_1_player_id` | object | None |
| 218 | `forced_fumble_player_1_player_name` | object | None |
| 219 | `forced_fumble_player_2_team` | object | None |
| 220 | `forced_fumble_player_2_player_id` | object | None |
| 221 | `forced_fumble_player_2_player_name` | object | None |
| 222 | `solo_tackle_1_team` | object | None |
| 223 | `solo_tackle_2_team` | object | None |
| 224 | `solo_tackle_1_player_id` | object | None |
| 225 | `solo_tackle_2_player_id` | object | None |
| 226 | `solo_tackle_1_player_name` | object | None |
| 227 | `solo_tackle_2_player_name` | object | None |
| 228 | `assist_tackle_1_player_id` | object | None |
| 229 | `assist_tackle_1_player_name` | object | None |
| 230 | `assist_tackle_1_team` | object | None |
| 231 | `assist_tackle_2_player_id` | object | None |
| 232 | `assist_tackle_2_player_name` | object | None |
| 233 | `assist_tackle_2_team` | object | None |
| 234 | `assist_tackle_3_player_id` | object | None |
| 235 | `assist_tackle_3_player_name` | object | None |
| 236 | `assist_tackle_3_team` | object | None |
| 237 | `assist_tackle_4_player_id` | object | None |
| 238 | `assist_tackle_4_player_name` | object | None |
| 239 | `assist_tackle_4_team` | object | None |
| 240 | `tackle_with_assist` | float32 | NULL |
| 241 | `tackle_with_assist_1_player_id` | object | None |
| 242 | `tackle_with_assist_1_player_name` | object | None |
| 243 | `tackle_with_assist_1_team` | object | None |
| 244 | `tackle_with_assist_2_player_id` | object | None |
| 245 | `tackle_with_assist_2_player_name` | object | None |
| 246 | `tackle_with_assist_2_team` | object | None |
| 247 | `pass_defense_1_player_id` | object | None |
| 248 | `pass_defense_1_player_name` | object | None |
| 249 | `pass_defense_2_player_id` | object | None |
| 250 | `pass_defense_2_player_name` | object | None |
| 251 | `fumbled_1_team` | object | None |
| 252 | `fumbled_1_player_id` | object | None |
| 253 | `fumbled_1_player_name` | object | None |
| 254 | `fumbled_2_player_id` | object | None |
| 255 | `fumbled_2_player_name` | object | None |
| 256 | `fumbled_2_team` | object | None |
| 257 | `fumble_recovery_1_team` | object | None |
| 258 | `fumble_recovery_1_yards` | float32 | NULL |
| 259 | `fumble_recovery_1_player_id` | object | None |
| 260 | `fumble_recovery_1_player_name` | object | None |
| 261 | `fumble_recovery_2_team` | object | None |
| 262 | `fumble_recovery_2_yards` | float32 | NULL |
| 263 | `fumble_recovery_2_player_id` | object | None |
| 264 | `fumble_recovery_2_player_name` | object | None |
| 265 | `sack_player_id` | object | None |
| 266 | `sack_player_name` | object | None |
| 267 | `half_sack_1_player_id` | object | None |
| 268 | `half_sack_1_player_name` | object | None |
| 269 | `half_sack_2_player_id` | object | None |
| 270 | `half_sack_2_player_name` | object | None |
| 271 | `return_team` | object | None |
| 272 | `return_yards` | float32 | NULL |
| 273 | `penalty_team` | object | None |
| 274 | `penalty_player_id` | object | None |
| 275 | `penalty_player_name` | object | None |
| 276 | `penalty_yards` | float32 | NULL |
| 277 | `replay_or_challenge` | float32 | 0.0 |
| 278 | `replay_or_challenge_result` | object | None |
| 279 | `penalty_type` | object | None |
| 280 | `defensive_two_point_attempt` | float32 | NULL |
| 281 | `defensive_two_point_conv` | float32 | NULL |
| 282 | `defensive_extra_point_attempt` | float32 | NULL |
| 283 | `defensive_extra_point_conv` | float32 | NULL |
| 284 | `safety_player_name` | object | None |
| 285 | `safety_player_id` | object | None |
| 286 | `season` | int64 | 2023 |
| 287 | `cp` | float32 | NULL |
| 288 | `cpoe` | float32 | NULL |
| 289 | `series` | float32 | 1.0 |
| 290 | `series_success` | float32 | 1.0 |
| 291 | `series_result` | object | First down |
| 292 | `order_sequence` | float32 | 1.0 |
| 293 | `start_time` | object | 9/10/23, 13:02:43 |
| 294 | `time_of_day` | object | None |
| 295 | `stadium` | object | Commanders Field |
| 296 | `weather` | object | Cloudy Temp: 76° F, Humidity: 84%, Wind: S 2 mph |
| 297 | `nfl_api_id` | object | b07c705e-f053-11ed-b4a7-bab79e4492fa |
| 298 | `play_clock` | object | 0 |
| 299 | `play_deleted` | float32 | 0.0 |
| 300 | `play_type_nfl` | object | GAME_START |
| 301 | `special_teams_play` | float32 | 0.0 |
| 302 | `st_play_type` | object | None |
| 303 | `end_clock_time` | object | None |
| 304 | `end_yard_line` | object | None |
| 305 | `fixed_drive` | float32 | 1.0 |
| 306 | `fixed_drive_result` | object | Punt |
| 307 | `drive_real_start_time` | object | None |
| 308 | `drive_play_count` | float32 | NULL |
| 309 | `drive_time_of_possession` | object | None |
| 310 | `drive_first_downs` | float32 | NULL |
| 311 | `drive_inside20` | float32 | NULL |
| 312 | `drive_ended_with_score` | float32 | NULL |
| 313 | `drive_quarter_start` | float32 | NULL |
| 314 | `drive_quarter_end` | float32 | NULL |
| 315 | `drive_yards_penalized` | float32 | NULL |
| 316 | `drive_start_transition` | object | None |
| 317 | `drive_end_transition` | object | None |
| 318 | `drive_game_clock_start` | object | None |
| 319 | `drive_game_clock_end` | object | None |
| 320 | `drive_start_yard_line` | object | None |
| 321 | `drive_end_yard_line` | object | None |
| 322 | `drive_play_id_started` | float32 | NULL |
| 323 | `drive_play_id_ended` | float32 | NULL |
| 324 | `away_score` | int32 | 16 |
| 325 | `home_score` | int32 | 20 |
| 326 | `location` | object | Home |
| 327 | `result` | int32 | 4 |
| 328 | `total` | int32 | 36 |
| 329 | `spread_line` | float32 | 7.0 |
| 330 | `total_line` | float32 | 38.0 |
| 331 | `div_game` | int32 | 0 |
| 332 | `roof` | object | outdoors |
| 333 | `surface` | object |  |
| 334 | `temp` | float32 | NULL |
| 335 | `wind` | float32 | NULL |
| 336 | `home_coach` | object | Ron Rivera |
| 337 | `away_coach` | object | Jonathan Gannon |
| 338 | `stadium_id` | object | WAS00 |
| 339 | `game_stadium` | object | FedExField |
| 340 | `aborted_play` | float32 | 0.0 |
| 341 | `success` | float32 | 0.0 |
| 342 | `passer` | object | None |
| 343 | `passer_jersey_number` | float32 | NULL |
| 344 | `rusher` | object | None |
| 345 | `rusher_jersey_number` | float32 | NULL |
| 346 | `receiver` | object | None |
| 347 | `receiver_jersey_number` | float32 | NULL |
| 348 | `pass` | float32 | 0.0 |
| 349 | `rush` | float32 | 0.0 |
| 350 | `first_down` | float32 | NULL |
| 351 | `special` | float32 | 0.0 |
| 352 | `play` | float32 | 0.0 |
| 353 | `passer_id` | object | None |
| 354 | `rusher_id` | object | None |
| 355 | `receiver_id` | object | None |
| 356 | `name` | object | None |
| 357 | `jersey_number` | float32 | NULL |
| 358 | `id` | object | None |
| 359 | `fantasy_player_name` | object | None |
| 360 | `fantasy_player_id` | object | None |
| 361 | `fantasy` | object | None |
| 362 | `fantasy_id` | object | None |
| 363 | `out_of_bounds` | float32 | 0.0 |
| 364 | `home_opening_kickoff` | float32 | 1.0 |
| 365 | `qb_epa` | float32 | 0.0 |
| 366 | `xyac_epa` | float32 | NULL |
| 367 | `xyac_mean_yardage` | float32 | NULL |
| 368 | `xyac_median_yardage` | float32 | NULL |
| 369 | `xyac_success` | float32 | NULL |
| 370 | `xyac_fd` | float32 | NULL |
| 371 | `xpass` | float32 | NULL |
| 372 | `pass_oe` | float32 | NULL |
| 373 | `nflverse_game_id` | object | 2023_01_ARI_WAS |
| 374 | `possession_team` | object |  |
| 375 | `offense_formation` | object | None |
| 376 | `offense_personnel` | object | None |
| 377 | `defenders_in_box` | float32 | NULL |
| 378 | `defense_personnel` | object | None |
| 379 | `number_of_pass_rushers` | float32 | NULL |
| 380 | `players_on_play` | object |  |
| 381 | `offense_players` | object |  |
| 382 | `defense_players` | object |  |
| 383 | `n_offense` | float32 | 0.0 |
| 384 | `n_defense` | float32 | 0.0 |
| 385 | `ngs_air_yards` | float32 | NULL |
| 386 | `time_to_throw` | float32 | NULL |
| 387 | `was_pressure` | float32 | NULL |
| 388 | `route` | object | None |
| 389 | `defense_man_zone_type` | object | None |
| 390 | `defense_coverage_type` | object | None |

---

### 12. import_players

**Data Shape:** 24,509 rows × 39 columns  
**Function Call:** `nfl.import_players()`

**Description:**  
Import descriptive data for all players

Returns:
    DataFrame

**Columns (39):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `gsis_id` | object | 00-0028830 |
| 2 | `display_name` | object | Isaako Aaitui |
| 3 | `common_first_name` | object | Isaako |
| 4 | `first_name` | object | Isaako |
| 5 | `last_name` | object | Aaitui |
| 6 | `short_name` | object | None |
| 7 | `football_name` | object | None |
| 8 | `suffix` | object | None |
| 9 | `esb_id` | object | AAI622937 |
| 10 | `nfl_id` | object | None |
| 11 | `pfr_id` | object | AaitIs00 |
| 12 | `pff_id` | object | 6998 |
| 13 | `otc_id` | object | 2535 |
| 14 | `espn_id` | object | 14856 |
| 15 | `smart_id` | object | 32004141-4962-2937-61ff-017b1804dec6 |
| 16 | `birth_date` | object | 1987-01-25 |
| 17 | `position_group` | object | DL |
| 18 | `position` | object | NT |
| 19 | `ngs_position_group` | object | None |
| 20 | `ngs_position` | object | None |
| 21 | `height` | float64 | 76.0 |
| 22 | `weight` | float64 | 307.0 |
| 23 | `headshot` | object | https://static.www.nfl.com/image/private/{forma... |
| 24 | `college_name` | object | UNLV |
| 25 | `college_conference` | object | None |
| 26 | `jersey_number` | object | 0 |
| 27 | `rookie_season` | int32 | 2011 |
| 28 | `last_season` | int32 | 2014 |
| 29 | `latest_team` | object | WAS |
| 30 | `status` | object | DEV |
| 31 | `ngs_status` | object | None |
| 32 | `ngs_status_short_description` | object | None |
| 33 | `years_of_experience` | int32 | 2 |
| 34 | `pff_position` | object | DI |
| 35 | `pff_status` | object | None |
| 36 | `draft_year` | float64 | NULL |
| 37 | `draft_round` | float64 | NULL |
| 38 | `draft_pick` | float64 | NULL |
| 39 | `draft_team` | object | None |

---

### 13. import_qbr

**Data Shape:** 82 rows × 23 columns  
**Function Call:** `nfl.import_qbr()`

**Description:**  
Import NFL or college QBR data

Args:
    years (List[int]): list of years to return data for, optional
    level (str): level to pull data, nfl or college, default to nfl
    frequency (str): frequen...

**Columns (23):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `season` | int64 | 2023 |
| 2 | `season_type` | object | Regular |
| 3 | `game_week` | object | Season Total |
| 4 | `team_abb` | object | SF |
| 5 | `player_id` | int64 | 4361741 |
| 6 | `name_short` | object | B. Purdy |
| 7 | `rank` | float64 | 1.0 |
| 8 | `qbr_total` | float64 | 72.8 |
| 9 | `pts_added` | float64 | 37.2 |
| 10 | `qb_plays` | int64 | 530 |
| 11 | `epa_total` | float64 | 76.9 |
| 12 | `pass` | float64 | 66.8 |
| 13 | `run` | float64 | 9.1 |
| 14 | `exp_sack` | int64 | 0 |
| 15 | `penalty` | float64 | 1.0 |
| 16 | `qbr_raw` | float64 | 73.0 |
| 17 | `sack` | float64 | -10.2 |
| 18 | `name_first` | object | Brock |
| 19 | `name_last` | object | Purdy |
| 20 | `name_display` | object | Brock Purdy |
| 21 | `headshot_href` | object | https://a.espncdn.com/i/headshots/nfl/players/f... |
| 22 | `team` | object | 49ers |
| 23 | `qualified` | bool | True |

---

### 14. import_sc_lines

**Data Shape:** 0 rows × 7 columns  
**Function Call:** `nfl.import_sc_lines()`

**Description:**  
Import weekly scoring lines

Args:
    years (List[int]): years to get scoring lines for
   
Returns:
    DataFrame

**Columns (7):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `season` | int64 | N/A |
| 2 | `week` | int64 | N/A |
| 3 | `away_team` | object | N/A |
| 4 | `home_team` | object | N/A |
| 5 | `game_id` | int64 | N/A |
| 6 | `side` | object | N/A |
| 7 | `line` | float64 | N/A |

---

### 15. import_schedules

**Data Shape:** 285 rows × 46 columns  
**Function Call:** `nfl.import_schedules()`

**Description:**  
Import schedules

Args:
    years (List[int]): years to get schedules for
    
Returns:
    DataFrame

**Columns (46):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `game_id` | object | 2023_01_DET_KC |
| 2 | `season` | int64 | 2023 |
| 3 | `game_type` | object | REG |
| 4 | `week` | int64 | 1 |
| 5 | `gameday` | object | 2023-09-07 |
| 6 | `weekday` | object | Thursday |
| 7 | `gametime` | object | 20:20 |
| 8 | `away_team` | object | DET |
| 9 | `away_score` | float64 | 21.0 |
| 10 | `home_team` | object | KC |
| 11 | `home_score` | float64 | 20.0 |
| 12 | `location` | object | Home |
| 13 | `result` | float64 | -1.0 |
| 14 | `total` | float64 | 41.0 |
| 15 | `overtime` | float64 | 0.0 |
| 16 | `old_game_id` | int64 | 2023090700 |
| 17 | `gsis` | float64 | 59173.0 |
| 18 | `nfl_detail_id` | object | NULL |
| 19 | `pfr` | object | 202309070kan |
| 20 | `pff` | float64 | NULL |
| 21 | `espn` | float64 | 401547353.0 |
| 22 | `ftn` | float64 | NULL |
| 23 | `away_rest` | int64 | 7 |
| 24 | `home_rest` | int64 | 7 |
| 25 | `away_moneyline` | float64 | 164.0 |
| 26 | `home_moneyline` | float64 | -198.0 |
| 27 | `spread_line` | float64 | 4.0 |
| 28 | `away_spread_odds` | float64 | -110.0 |
| 29 | `home_spread_odds` | float64 | -110.0 |
| 30 | `total_line` | float64 | 53.0 |
| 31 | `under_odds` | float64 | -110.0 |
| 32 | `over_odds` | float64 | -110.0 |
| 33 | `div_game` | int64 | 0 |
| 34 | `roof` | object | outdoors |
| 35 | `surface` | object | NULL |
| 36 | `temp` | float64 | NULL |
| 37 | `wind` | float64 | NULL |
| 38 | `away_qb_id` | object | 00-0033106 |
| 39 | `home_qb_id` | object | 00-0033873 |
| 40 | `away_qb_name` | object | Jared Goff |
| 41 | `home_qb_name` | object | Patrick Mahomes |
| 42 | `away_coach` | object | Dan Campbell |
| 43 | `home_coach` | object | Andy Reid |
| 44 | `referee` | object | John Hussey |
| 45 | `stadium_id` | object | KAN00 |
| 46 | `stadium` | object | GEHA Field at Arrowhead Stadium |

---

### 16. import_seasonal_data

**Data Shape:** 588 rows × 58 columns  
**Function Call:** `nfl.import_seasonal_data()`

**Description:**  
Imports seasonal player data

Args:
    years (List[int]): years to get seasonal data for
    s_type (str): season type to include in average ('ALL','REG','POST')
Returns:
    DataFrame

**Columns (58):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `player_id` | object | 00-0023459 |
| 2 | `season` | int32 | 2023 |
| 3 | `season_type` | object | REG |
| 4 | `completions` | int32 | 0 |
| 5 | `attempts` | int32 | 1 |
| 6 | `passing_yards` | float64 | 0.0 |
| 7 | `passing_tds` | int32 | 0 |
| 8 | `interceptions` | float64 | 0.0 |
| 9 | `sacks` | float64 | 1.0 |
| 10 | `sack_yards` | float64 | 10.0 |
| 11 | `sack_fumbles` | int32 | 0 |
| 12 | `sack_fumbles_lost` | int32 | 0 |
| 13 | `passing_air_yards` | float64 | 17.0 |
| 14 | `passing_yards_after_catch` | float64 | 0.0 |
| 15 | `passing_first_downs` | float64 | 0.0 |
| 16 | `passing_epa` | float64 | -2.0319598843343556 |
| 17 | `passing_2pt_conversions` | int32 | 0 |
| 18 | `pacr` | float64 | 0.0 |
| 19 | `dakota` | float64 | 0.0 |
| 20 | `carries` | int32 | 0 |
| 21 | `rushing_yards` | float64 | 0.0 |
| 22 | `rushing_tds` | int32 | 0 |
| 23 | `rushing_fumbles` | float64 | 0.0 |
| 24 | `rushing_fumbles_lost` | float64 | 0.0 |
| 25 | `rushing_first_downs` | float64 | 0.0 |
| 26 | `rushing_epa` | float64 | 0.0 |
| 27 | `rushing_2pt_conversions` | int32 | 0 |
| 28 | `receptions` | int32 | 0 |
| 29 | `targets` | int32 | 0 |
| 30 | `receiving_yards` | float64 | 0.0 |
| 31 | `receiving_tds` | int32 | 0 |
| 32 | `receiving_fumbles` | float64 | 0.0 |
| 33 | `receiving_fumbles_lost` | float64 | 0.0 |
| 34 | `receiving_air_yards` | float64 | 0.0 |
| 35 | `receiving_yards_after_catch` | float64 | 0.0 |
| 36 | `receiving_first_downs` | float64 | 0.0 |
| 37 | `receiving_epa` | float64 | 0.0 |
| 38 | `receiving_2pt_conversions` | int32 | 0 |
| 39 | `racr` | float64 | 0.0 |
| 40 | `target_share` | float64 | 0.0 |
| 41 | `air_yards_share` | float64 | 0.0 |
| 42 | `wopr_x` | float64 | 0.0 |
| 43 | `special_teams_tds` | float64 | 0.0 |
| 44 | `fantasy_points` | float64 | 0.0 |
| 45 | `fantasy_points_ppr` | float64 | 0.0 |
| 46 | `games` | int64 | 1 |
| 47 | `tgt_sh` | float64 | 0.0 |
| 48 | `ay_sh` | float64 | 0.0 |
| 49 | `yac_sh` | float64 | 0.0 |
| 50 | `wopr_y` | float64 | 0.0 |
| 51 | `ry_sh` | float64 | 0.0 |
| 52 | `rtd_sh` | float64 | 0.0 |
| 53 | `rfd_sh` | float64 | 0.0 |
| 54 | `rtdfd_sh` | float64 | 0.0 |
| 55 | `dom` | float64 | 0.0 |
| 56 | `w8dom` | float64 | 0.0 |
| 57 | `yptmpa` | float64 | 0.0 |
| 58 | `ppr_sh` | float64 | 0.0 |

---

### 17. import_seasonal_pfr

**Data Shape:** 104 rows × 28 columns  
**Function Call:** `nfl.import_seasonal_pfr()`

**Description:**  
Import PFR advanced season-level statistics

Args:
    s_type (str): must be one of pass, rec, rush
    years (List[int]): years to return data for, optional
Returns:
    DataFrame

**Columns (28):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `player` | object | Tua Tagovailoa |
| 2 | `team` | object | MIA |
| 3 | `pass_attempts` | float64 | 560.0 |
| 4 | `throwaways` | float64 | 14.0 |
| 5 | `spikes` | float64 | 2.0 |
| 6 | `drops` | float64 | 24.0 |
| 7 | `drop_pct` | float64 | 4.4 |
| 8 | `bad_throws` | float64 | 78.0 |
| 9 | `bad_throw_pct` | float64 | 14.3 |
| 10 | `season` | int32 | 2023 |
| 11 | `pfr_id` | object | TagoTu00 |
| 12 | `pocket_time` | float64 | 2.1 |
| 13 | `times_blitzed` | float64 | 116.0 |
| 14 | `times_hurried` | float64 | 39.0 |
| 15 | `times_hit` | float64 | 27.0 |
| 16 | `times_pressured` | float64 | 95.0 |
| 17 | `pressure_pct` | float64 | 15.7 |
| 18 | `batted_balls` | float64 | 9.0 |
| 19 | `on_tgt_throws` | float64 | 430.0 |
| 20 | `on_tgt_pct` | float64 | 79.0 |
| 21 | `rpo_plays` | float64 | 111.0 |
| 22 | `rpo_yards` | float64 | 1073.0 |
| 23 | `rpo_pass_att` | float64 | 105.0 |
| 24 | `rpo_pass_yards` | float64 | 1069.0 |
| 25 | `rpo_rush_att` | float64 | 1.0 |
| 26 | `rpo_rush_yards` | float64 | 4.0 |
| 27 | `pa_pass_att` | float64 | 126.0 |
| 28 | `pa_pass_yards` | float64 | 1145.0 |

---

### 18. import_seasonal_rosters

**Data Shape:** 3,089 rows × 37 columns  
**Function Call:** `nfl.import_seasonal_rosters()`

**Description:**  
Imports roster data as of the end of the season

Args:
    years (List[int]): years to get rosters for
    columns (List[str]): list of columns to return with DataFrame
    
Returns:
    DataFrame

**Columns (37):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `season` | int32 | 2023 |
| 2 | `team` | object | PHI |
| 3 | `position` | object | OL |
| 4 | `depth_chart_position` | object | T |
| 5 | `jersey_number` | float64 | 74.0 |
| 6 | `status` | object | CUT |
| 7 | `player_name` | object | Bernard Williams |
| 8 | `first_name` | object | Bernard |
| 9 | `last_name` | object | Williams |
| 10 | `birth_date` | datetime64[ns] | NaT |
| 11 | `height` | float64 | 80.0 |
| 12 | `weight` | int32 | 286 |
| 13 | `college` | object | None |
| 14 | `player_id` | object | 00-0017724 |
| 15 | `espn_id` | object | None |
| 16 | `sportradar_id` | object | None |
| 17 | `yahoo_id` | object | None |
| 18 | `rotowire_id` | object | None |
| 19 | `pff_id` | object | None |
| 20 | `pfr_id` | object | None |
| 21 | `fantasy_data_id` | object | None |
| 22 | `sleeper_id` | object | None |
| 23 | `years_exp` | int32 | 29 |
| 24 | `headshot_url` | object | None |
| 25 | `ngs_position` | object | None |
| 26 | `week` | int32 | 11 |
| 27 | `game_type` | object | REG |
| 28 | `status_description_abbr` | object | W03 |
| 29 | `football_name` | object | Bernard |
| 30 | `esb_id` | object | WIL148626 |
| 31 | `gsis_it_id` | object | 17623 |
| 32 | `smart_id` | object | 32005749-4c14-8626-f883-08eba7248da6 |
| 33 | `entry_year` | int32 | 1994 |
| 34 | `rookie_year` | float64 | 1994.0 |
| 35 | `draft_club` | object | PHI |
| 36 | `draft_number` | float64 | 14.0 |
| 37 | `age` | float64 | NULL |

---

### 19. import_snap_counts

**Data Shape:** 26,513 rows × 16 columns  
**Function Call:** `nfl.import_snap_counts()`

**Description:**  
Import snap count data for individual players

Args:
    years (List[int]): years to return snap counts for
Returns:
    DataFrame

**Columns (16):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `game_id` | object | 2023_01_ARI_WAS |
| 2 | `pfr_game_id` | object | 202309100was |
| 3 | `season` | int32 | 2023 |
| 4 | `game_type` | object | REG |
| 5 | `week` | int32 | 1 |
| 6 | `player` | object | Saahdiq Charles |
| 7 | `pfr_player_id` | object | CharSa00 |
| 8 | `position` | object | T |
| 9 | `team` | object | WAS |
| 10 | `opponent` | object | ARI |
| 11 | `offense_snaps` | float64 | 71.0 |
| 12 | `offense_pct` | float64 | 1.0 |
| 13 | `defense_snaps` | float64 | 0.0 |
| 14 | `defense_pct` | float64 | 0.0 |
| 15 | `st_snaps` | float64 | 3.0 |
| 16 | `st_pct` | float64 | 0.11 |

---

### 20. import_team_desc

**Data Shape:** 36 rows × 16 columns  
**Function Call:** `nfl.import_team_desc()`

**Description:**  
Import team descriptive data

Returns:
    DataFrame

**Columns (16):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `team_abbr` | object | ARI |
| 2 | `team_name` | object | Arizona Cardinals |
| 3 | `team_id` | int64 | 3800 |
| 4 | `team_nick` | object | Cardinals |
| 5 | `team_conf` | object | NFC |
| 6 | `team_division` | object | NFC West |
| 7 | `team_color` | object | #97233F |
| 8 | `team_color2` | object | #000000 |
| 9 | `team_color3` | object | #ffb612 |
| 10 | `team_color4` | object | #a5acaf |
| 11 | `team_logo_wikipedia` | object | https://upload.wikimedia.org/wikipedia/en/thumb... |
| 12 | `team_logo_espn` | object | https://a.espncdn.com/i/teamlogos/nfl/500/ari.png |
| 13 | `team_wordmark` | object | https://github.com/nflverse/nflverse-pbp/raw/ma... |
| 14 | `team_conference_logo` | object | https://github.com/nflverse/nflverse-pbp/raw/ma... |
| 15 | `team_league_logo` | object | https://raw.githubusercontent.com/nflverse/nflv... |
| 16 | `team_logo_squared` | object | https://github.com/nflverse/nflverse-pbp/raw/ma... |

---

### 21. import_weekly_data

**Data Shape:** 5,653 rows × 53 columns  
**Function Call:** `nfl.import_weekly_data()`

**Description:**  
Imports weekly player data

Args:
    years (List[int]): years to get weekly data for
    columns (List[str]): only return these columns
    downcast (bool): convert float64 to float32, default True
R...

**Columns (53):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `player_id` | object | 00-0023459 |
| 2 | `player_name` | object | A.Rodgers |
| 3 | `player_display_name` | object | Aaron Rodgers |
| 4 | `position` | object | QB |
| 5 | `position_group` | object | QB |
| 6 | `headshot_url` | object | https://static.www.nfl.com/image/upload/f_auto,... |
| 7 | `recent_team` | object | NYJ |
| 8 | `season` | int32 | 2023 |
| 9 | `week` | int32 | 1 |
| 10 | `season_type` | object | REG |
| 11 | `opponent_team` | object | BUF |
| 12 | `completions` | int32 | 0 |
| 13 | `attempts` | int32 | 1 |
| 14 | `passing_yards` | float32 | 0.0 |
| 15 | `passing_tds` | int32 | 0 |
| 16 | `interceptions` | float32 | 0.0 |
| 17 | `sacks` | float32 | 1.0 |
| 18 | `sack_yards` | float32 | 10.0 |
| 19 | `sack_fumbles` | int32 | 0 |
| 20 | `sack_fumbles_lost` | int32 | 0 |
| 21 | `passing_air_yards` | float32 | 17.0 |
| 22 | `passing_yards_after_catch` | float32 | 0.0 |
| 23 | `passing_first_downs` | float32 | 0.0 |
| 24 | `passing_epa` | float32 | -2.0319597721099854 |
| 25 | `passing_2pt_conversions` | int32 | 0 |
| 26 | `pacr` | float32 | 0.0 |
| 27 | `dakota` | float32 | NULL |
| 28 | `carries` | int32 | 0 |
| 29 | `rushing_yards` | float32 | 0.0 |
| 30 | `rushing_tds` | int32 | 0 |
| 31 | `rushing_fumbles` | float32 | 0.0 |
| 32 | `rushing_fumbles_lost` | float32 | 0.0 |
| 33 | `rushing_first_downs` | float32 | 0.0 |
| 34 | `rushing_epa` | float32 | NULL |
| 35 | `rushing_2pt_conversions` | int32 | 0 |
| 36 | `receptions` | int32 | 0 |
| 37 | `targets` | int32 | 0 |
| 38 | `receiving_yards` | float32 | 0.0 |
| 39 | `receiving_tds` | int32 | 0 |
| 40 | `receiving_fumbles` | float32 | 0.0 |
| 41 | `receiving_fumbles_lost` | float32 | 0.0 |
| 42 | `receiving_air_yards` | float32 | 0.0 |
| 43 | `receiving_yards_after_catch` | float32 | 0.0 |
| 44 | `receiving_first_downs` | float32 | 0.0 |
| 45 | `receiving_epa` | float32 | NULL |
| 46 | `receiving_2pt_conversions` | int32 | 0 |
| 47 | `racr` | float32 | NULL |
| 48 | `target_share` | float32 | NULL |
| 49 | `air_yards_share` | float32 | NULL |
| 50 | `wopr` | float32 | NULL |
| 51 | `special_teams_tds` | float32 | 0.0 |
| 52 | `fantasy_points` | float32 | 0.0 |
| 53 | `fantasy_points_ppr` | float32 | 0.0 |

---

### 22. import_weekly_pfr

**Data Shape:** 700 rows × 24 columns  
**Function Call:** `nfl.import_weekly_pfr()`

**Description:**  
Import PFR advanced week-level statistics

Args:
    s_type (str): must be one of pass, rec, rush
    years (List[int]): years to return data for, optional
Returns:
    DataFrame

**Columns (24):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `game_id` | object | 2023_01_DET_KC |
| 2 | `pfr_game_id` | object | 202309070kan |
| 3 | `season` | int32 | 2023 |
| 4 | `week` | int32 | 1 |
| 5 | `game_type` | object | REG |
| 6 | `team` | object | KC |
| 7 | `opponent` | object | DET |
| 8 | `pfr_player_name` | object | Patrick Mahomes |
| 9 | `pfr_player_id` | object | MahoPa00 |
| 10 | `passing_drops` | float64 | 5.0 |
| 11 | `passing_drop_pct` | float64 | 0.135 |
| 12 | `receiving_drop` | float64 | NULL |
| 13 | `receiving_drop_pct` | float64 | NULL |
| 14 | `passing_bad_throws` | float64 | 7.0 |
| 15 | `passing_bad_throw_pct` | float64 | 0.189 |
| 16 | `times_sacked` | float64 | 0.0 |
| 17 | `times_blitzed` | float64 | 5.0 |
| 18 | `times_hurried` | float64 | 1.0 |
| 19 | `times_hit` | float64 | 7.0 |
| 20 | `times_pressured` | float64 | 8.0 |
| 21 | `times_pressured_pct` | float64 | 0.178 |
| 22 | `def_times_blitzed` | float64 | NULL |
| 23 | `def_times_hurried` | float64 | NULL |
| 24 | `def_times_hitqb` | float64 | NULL |

---

### 23. import_weekly_rosters

**Data Shape:** 45,655 rows × 37 columns  
**Function Call:** `nfl.import_weekly_rosters()`

**Description:**  
Imports roster data including mid-season changes

Args:
    years (List[int]): years to get rosters for
    columns (List[str]): list of columns to return with DataFrame
    
Returns:
    DataFrame

**Columns (37):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `season` | int32 | 2023 |
| 2 | `team` | object | PHI |
| 3 | `position` | object | OL |
| 4 | `depth_chart_position` | object | T |
| 5 | `jersey_number` | float64 | 74.0 |
| 6 | `status` | object | CUT |
| 7 | `player_name` | object | Bernard Williams |
| 8 | `first_name` | object | Bernard |
| 9 | `last_name` | object | Williams |
| 10 | `birth_date` | datetime64[ns] | NaT |
| 11 | `height` | float64 | 80.0 |
| 12 | `weight` | int32 | 286 |
| 13 | `college` | object | None |
| 14 | `player_id` | object | 00-0017724 |
| 15 | `espn_id` | object | None |
| 16 | `sportradar_id` | object | None |
| 17 | `yahoo_id` | object | None |
| 18 | `rotowire_id` | object | None |
| 19 | `pff_id` | object | None |
| 20 | `pfr_id` | object | None |
| 21 | `fantasy_data_id` | object | None |
| 22 | `sleeper_id` | object | None |
| 23 | `years_exp` | int32 | 29 |
| 24 | `headshot_url` | object | None |
| 25 | `ngs_position` | object | None |
| 26 | `week` | int32 | 11 |
| 27 | `game_type` | object | REG |
| 28 | `status_description_abbr` | object | W03 |
| 29 | `football_name` | object | Bernard |
| 30 | `esb_id` | object | WIL148626 |
| 31 | `gsis_it_id` | object | 17623 |
| 32 | `smart_id` | object | 32005749-4c14-8626-f883-08eba7248da6 |
| 33 | `entry_year` | int32 | 1994 |
| 34 | `rookie_year` | float64 | 1994.0 |
| 35 | `draft_club` | object | PHI |
| 36 | `draft_number` | float64 | 14.0 |
| 37 | `age` | float64 | NULL |

---

### 24. import_win_totals

**Data Shape:** 0 rows × 9 columns  
**Function Call:** `nfl.import_win_totals()`

**Description:**  
Import win total projections

Args:
    years (List[int]): years to get win totals for
    
Returns:
    DataFrame

**Columns (9):**

| # | Column Name | Data Type | Sample Value |
|---|-------------|-----------|--------------|
| 1 | `game_id` | object | N/A |
| 2 | `market_type` | object | N/A |
| 3 | `abbr` | object | N/A |
| 4 | `lines` | float64 | N/A |
| 5 | `odds` | int64 | N/A |
| 6 | `opening_lines` | float64 | N/A |
| 7 | `opening_odds` | float64 | N/A |
| 8 | `book` | object | N/A |
| 9 | `season` | int64 | N/A |

---

## Usage Examples

### Basic Data Fetching
```python
import nfl_data_py as nfl
import pandas as pd

# Fetch team information
teams = nfl.import_team_desc()

# Fetch player rosters for specific years
rosters = nfl.import_seasonal_rosters([2023, 2024])

# Fetch game schedules
schedules = nfl.import_schedules([2023])

# Fetch weekly player stats
weekly_stats = nfl.import_weekly_data([2023])
```

### Advanced Data Fetching
```python
# Next Gen Stats (requires stat type)
passing_ngs = nfl.import_ngs_data('passing', [2023])
rushing_ngs = nfl.import_ngs_data('rushing', [2023])

# Play-by-play data
pbp = nfl.import_pbp_data([2023])

# Pro Football Reference data (requires stat type)
seasonal_passing = nfl.import_seasonal_pfr('pass', [2023])
weekly_receiving = nfl.import_weekly_pfr('rec', [2023])
```

## Integration with Your Project

These functions can be easily integrated into your existing data pipeline:

```python
from src.core.data.fetch import (
    fetch_team_data,
    fetch_player_data,
    fetch_game_schedule_data,
    fetch_weekly_stats_data
)

# Your existing fetch functions already wrap these:
teams = fetch_team_data()  # -> nfl.import_team_desc()
players = fetch_player_data([2023])  # -> nfl.import_seasonal_rosters([2023])
games = fetch_game_schedule_data([2023])  # -> nfl.import_schedules([2023])
stats = fetch_weekly_stats_data([2023])  # -> nfl.import_weekly_data([2023])
```

---

*This documentation was automatically generated by exploring the nfl_data_py library.*
